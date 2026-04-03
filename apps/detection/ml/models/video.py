"""
model.py — Improved EfficientNet + GRU Deepfake Detector (for Django backend)

Adaptation from https://github.com/firdavsm19/deepfake-detector/tree/main/video

Architecture highlights:
  • EfficientNet-B4 backbone (timm) with optional gradient checkpointing
  • Multi-scale feature fusion from multiple backbone stages
  • Bidirectional GRU for temporal modelling
  • Multi-head self-attention over GRU outputs (temporal attention)
  • Deep classification head with residual dropout
  • Optional frame-level auxiliary head for richer supervision
  • Frequency-domain branch (DCT) optionally fused with spatial features
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ──────────────────────────────────────────────────────────────
# Config (standalone for Django)
# ──────────────────────────────────────────────────────────────

class VideoConfig:
    """Model configuration (standalone version independent from config.py)."""
    # Data
    FACE_SIZE = 224
    NUM_FRAMES = 16

    # Model
    BACKBONE = "efficientnet_b4"
    PRETRAINED = "imagenet"
    USE_GRADIENT_CHECKPOINTING = False
    USE_MULTI_SCALE = True  # Checkpoint trained with multi-scale ([2, 4] stages)
    MULTI_SCALE_STAGES = [2, 4]  # Checkpoint: stages 2 and 4 (56 and 448 channels)
    USE_FREQ_BRANCH = True  # Checkpoint trained WITH frequency branch (512 + 128 = 640)

    # GRU
    GRU_HIDDEN_DIM = 512
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.3
    GRU_BIDIRECTIONAL = True

    # Attention
    USE_TEMPORAL_ATTENTION = True
    ATTENTION_HEADS = 8

    # Classifier
    CLASSIFIER_DIMS = [512, 256, 128]
    CLASSIFIER_DROPOUT = 0.5

    # Auxiliary
    USE_AUX_LOSS = True


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w


class FrequencyBranch(nn.Module):
    """
    Lightweight DCT-based frequency-domain branch.
    Extracts frequency artifacts invisible in pixel space.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    @staticmethod
    def _dct_approx(x: torch.Tensor) -> torch.Tensor:
        """Approximate DCT via FFT magnitude spectrum."""
        x_f = torch.fft.fft2(x)
        return torch.abs(x_f) + 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self._dct_approx(x)
        freq = torch.log(freq)
        freq = self.conv(freq)
        return self.fc(freq.flatten(1))


class MultiScaleFusion(nn.Module):
    """
    Fuse feature maps from multiple EfficientNet stages
    into a single fixed-size vector.
    """

    def __init__(self, stage_dims: List[int], out_dim: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d, out_dim // len(stage_dims), bias=False),
                nn.LayerNorm(out_dim // len(stage_dims)),
            ) for d in stage_dims
        ])
        self.out_dim = out_dim

    def forward(self, stage_feats: List[torch.Tensor]) -> torch.Tensor:
        parts = [proj(f) for proj, f in zip(self.projections, stage_feats)]
        return torch.cat(parts, dim=-1)


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over GRU hidden states.
    Weights important frames differently.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x, _ = self.attn(x, x, x)
        x = self.norm(self.dropout(x) + residual)
        return x


class ClassificationHead(nn.Module):
    """Deep residual classification head."""

    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h, bias=False),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    EfficientNet + GRU deepfake detector for videos.

    Input:  (B, T, C, H, W)  — batch of video clips
    Output: (B,) — clip logits, optional (B, T) frame logits
    """

    def __init__(self, mcfg: VideoConfig = None):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm")

        mcfg = mcfg or VideoConfig()

        # ── 1. EfficientNet Backbone ──────────────────────────
        self.backbone = self._build_backbone(mcfg)

        if mcfg.USE_GRADIENT_CHECKPOINTING:
            try:
                self.backbone.set_grad_checkpointing(True)
            except AttributeError:
                pass  # Not all timm backbones support this

        # Detect feature dimensions
        if mcfg.USE_MULTI_SCALE:
            dummy = torch.zeros(1, 3, mcfg.FACE_SIZE, mcfg.FACE_SIZE)
            with torch.no_grad():
                stage_out = self.backbone(dummy)
            stage_dims = [s.shape[1] for s in stage_out]
            ms_out_dim = 512  # Standard value (will be recalculated from checkpoint)
            self.ms_fusion = MultiScaleFusion(stage_dims, ms_out_dim)
            spatial_dim = ms_out_dim
        else:
            dummy = torch.zeros(1, 3, mcfg.FACE_SIZE, mcfg.FACE_SIZE)
            with torch.no_grad():
                feat = self.backbone(dummy)
            spatial_dim = feat.shape[-1]
            self.ms_fusion = None

        # ── 2. Frequency Branch ───────────────────────────────
        self.use_freq = mcfg.USE_FREQ_BRANCH
        freq_dim = 128
        if self.use_freq:
            self.freq_branch = FrequencyBranch(out_dim=freq_dim)
            spatial_dim += freq_dim

        # ── 3. Feature Projection ─────────────────────────────
        proj_dim = mcfg.GRU_HIDDEN_DIM
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ── 4. Bidirectional GRU ──────────────────────────────
        gru_out = mcfg.GRU_HIDDEN_DIM * (2 if mcfg.GRU_BIDIRECTIONAL else 1)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=mcfg.GRU_HIDDEN_DIM,
            num_layers=mcfg.GRU_NUM_LAYERS,
            batch_first=True,
            dropout=mcfg.GRU_DROPOUT if mcfg.GRU_NUM_LAYERS > 1 else 0,
            bidirectional=mcfg.GRU_BIDIRECTIONAL,
        )

        # ── 5. Temporal Attention ─────────────────────────────
        self.use_attn = mcfg.USE_TEMPORAL_ATTENTION
        if self.use_attn:
            self.temporal_attn = TemporalAttention(
                hidden_dim=gru_out,
                num_heads=mcfg.ATTENTION_HEADS,
                dropout=0.1,
            )

        # ── 6. Aggregation ────────────────────────────────────
        agg_dim = gru_out * 2  # mean + max pooling

        # ── 7. Classification Head ────────────────────────────
        self.head = ClassificationHead(
            in_dim=agg_dim,
            hidden_dims=mcfg.CLASSIFIER_DIMS,
            dropout=mcfg.CLASSIFIER_DROPOUT,
        )

        # ── 8. Auxiliary Frame Head ───────────────────────────
        self.use_aux = mcfg.USE_AUX_LOSS
        if self.use_aux:
            self.aux_head = nn.Linear(gru_out, 1)

        # ── Weight Init ───────────────────────────────────────
        self._init_new_weights()

    @staticmethod
    def _build_backbone(mcfg: VideoConfig) -> nn.Module:
        """
        Build timm backbone with protection against out_indices mismatches.
        Automatically corrects indices if they are incorrect.
        """
        pretrained = (mcfg.PRETRAINED == "imagenet")

        if not mcfg.USE_MULTI_SCALE:
            return timm.create_model(mcfg.BACKBONE, pretrained=pretrained)

        requested_indices = list(mcfg.MULTI_SCALE_STAGES)
        try:
            return timm.create_model(
                mcfg.BACKBONE,
                pretrained=pretrained,
                features_only=True,
                out_indices=tuple(requested_indices),
            )
        except (IndexError, ValueError):
            # Fallback: probe valid indices and adjust
            probe = timm.create_model(
                mcfg.BACKBONE,
                pretrained=False,
                features_only=True,
            )
            valid_indices = list(range(len(probe.feature_info.info)))
            adjusted = [i for i in requested_indices if i in valid_indices]

            if not adjusted:
                # Use last 3 stages as fallback
                adjusted = valid_indices[-3:] if len(valid_indices) >= 3 else valid_indices

            print(
                f"[VideoModel] Requested out_indices={requested_indices} incompatible "
                f"with stages={valid_indices}; using {adjusted}"
            )

            return timm.create_model(
                mcfg.BACKBONE,
                pretrained=pretrained,
                features_only=True,
                out_indices=tuple(adjusted),
            )

    def _init_new_weights(self):
        """Initialize weights for new layers."""
        for m in [self.proj, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from frames.
        x: (B*T, C, H, W)
        returns: (B*T, spatial_dim)
        """
        if self.ms_fusion is not None:
            stages = self.backbone(x)
            spatial = self.ms_fusion(stages)
        else:
            spatial = self.backbone(x)

        if self.use_freq:
            freq = self.freq_branch(x)
            spatial = torch.cat([spatial, freq], dim=-1)

        return spatial

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, C, H, W) video clips
        Returns:
            clip_logits:  (B,) prediction per clip
            frame_logits: (B, T) or None
        """
        B, T, C, H, W = x.shape

        # Spatial features per frame
        x_flat = x.view(B * T, C, H, W)
        feats = self._extract_frame_features(x_flat)
        feats = feats.view(B, T, -1)

        # Project
        feats = self.proj(feats.view(B * T, -1))
        feats = feats.view(B, T, -1)

        # GRU
        gru_out, _ = self.gru(feats)

        # Temporal attention
        if self.use_attn:
            gru_out = self.temporal_attn(gru_out)

        # Auxiliary frame logits
        frame_logits = None
        if self.use_aux and self.training:
            frame_logits = self.aux_head(gru_out).squeeze(-1)

        # Mean + max pooling
        mean_pool = gru_out.mean(dim=1)
        max_pool = gru_out.max(dim=1).values
        agg = torch.cat([mean_pool, max_pool], dim=-1)

        # Classification
        clip_logits = self.head(agg)

        return clip_logits, frame_logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fake probability per clip."""
        self.eval()
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)

    def param_count(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
