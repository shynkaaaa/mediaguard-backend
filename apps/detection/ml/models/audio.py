"""
VoiceDetector model for deepfake audio detection.

Architecture: Wav2Vec2 + AASIST-style graph attention
Source: https://github.com/firdavsm19/deepfake-detector/tree/main/audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class AudioConfig:
    """Configuration for audio model."""
    BACKBONE = "facebook/wav2vec2-base"
    BACKBONE_DIM = 768
    PROJ_DIM = 128
    FREEZE_BACKBONE = True
    SAMPLE_RATE = 16000
    MAX_DURATION_SEC = 4.0
    MAX_LEN = int(SAMPLE_RATE * MAX_DURATION_SEC)  # 64000 samples


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention over sequence of node features."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W(x)
        B, T, D = h.shape
        hi = h.unsqueeze(2).expand(-1, -1, T, -1)
        hj = h.unsqueeze(1).expand(-1, T, -1, -1)
        e = self.leaky(self.a(torch.cat([hi, hj], dim=-1)).squeeze(-1))
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.bmm(alpha, h)
        return F.elu(out)


class AASISTBackend(nn.Module):
    """Spectro-temporal graph attention backend."""

    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.gat1 = GraphAttentionLayer(in_dim, 128)
        self.gat2 = GraphAttentionLayer(128, 64)
        self.pool_attn = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x)
        x = self.gat2(x)
        w = torch.softmax(self.pool_attn(x), dim=1)
        x = (w * x).sum(dim=1)
        return self.head(x)


class VoiceDetector(nn.Module):
    """
    Wav2Vec2 (frozen or fine-tuned) → projection → AASIST backend.

    Input:  (B, T) — raw waveform, 16 kHz, normalised to [-1, 1]
    Output: (B, 2) — logits [real, fake]
    """

    def __init__(self):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers package required for VoiceDetector. "
                "Install with: pip install transformers"
            )

        try:
            self.backbone = Wav2Vec2Model.from_pretrained(AudioConfig.BACKBONE)
        except Exception:
            # Fallback for offline environments; weights can still be restored from checkpoint.
            self.backbone = Wav2Vec2Model(Wav2Vec2Config())
        self.proj = nn.Sequential(
            nn.Linear(AudioConfig.BACKBONE_DIM, AudioConfig.PROJ_DIM),
            nn.LayerNorm(AudioConfig.PROJ_DIM),
            nn.GELU(),
        )
        self.backend = AASISTBackend(in_dim=AudioConfig.PROJ_DIM)

        if AudioConfig.FREEZE_BACKBONE:
            self._freeze_backbone()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if AudioConfig.FREEZE_BACKBONE:
            with torch.no_grad():
                feats = self.backbone(wav).last_hidden_state
        else:
            feats = self.backbone(wav).last_hidden_state
        x = self.proj(feats)
        return self.backend(x)

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
