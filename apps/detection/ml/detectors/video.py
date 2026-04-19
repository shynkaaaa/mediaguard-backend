"""
Deepfake video detector.

Uses DeepfakeDetector (EfficientNet + GRU).
"""

import logging
import os
from pathlib import Path
from typing import List

import torch
import numpy as np

from ..base import BaseDetector, DetectionResult, MediaType
from ..factory import DetectorFactory

logger = logging.getLogger(__name__)

# Path to video model weights
WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "video" / "best_model.pth"

# Video config
NUM_FRAMES = 16
FACE_SIZE = 224


class VideoDetector(BaseDetector):
    """
    Deepfake video detector.

    Architecture: EfficientNet-B4 + GRU + Temporal Attention
    Put the model in: apps/detection/ml/weights/video/best_model.pth
    """

    MODEL_VERSION = "efficientnet-gru-v1"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._transform = None
        self._load_model()

    def _load_model(self):
        if not WEIGHTS_PATH.exists():
            logger.warning(
                "Video model weights not found: %s — detector not ready",
                WEIGHTS_PATH,
            )
            return

        try:
            from ..models.video import DeepfakeDetector as VideoModel
            from torchvision import transforms

            logger.info("Creating VideoModel instance...")
            self._model = VideoModel()

            logger.info("Loading weights from %s...", WEIGHTS_PATH)
            checkpoint = torch.load(str(WEIGHTS_PATH), map_location=self.device)

            # Checkpoint can contain entire state (epoch, optimizer, etc)
            # Extract only model weights
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint

            self._model.load_state_dict(state_dict, strict=False)
            self._model.eval().to(self.device)

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((FACE_SIZE, FACE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            logger.info("VideoDetector loaded from %s on %s", WEIGHTS_PATH, self.device)
        except ImportError as e:
            logger.error("Failed to import VideoModel: %s", e)
            logger.error("Install timm: pip install timm")
            self._model = None
        except Exception as e:
            logger.exception("Failed to load video model: %s", e)
            self._model = None

    def _extract_frames(self, file_path: str) -> List[np.ndarray]:
        """Extract evenly distributed frames from video."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required: pip install opencv-python")

        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise ValueError(f"Could not read video: {file_path}")

        # Uniformly select NUM_FRAMES frames
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        # If there are fewer frames than needed, duplicate the last one
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1] if frames else np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8))

        return frames[:NUM_FRAMES]

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION if self.is_ready else "not-loaded"

    def predict(self, file_path: str) -> DetectionResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_ready:
            logger.warning("VideoDetector in stub mode for file %s", file_path)
            return DetectionResult(
                fake_probability=0.0,
                is_fake=False,
                model_version="stub",
                details={"warning": "video_model_not_ready"},
            )

        try:
            # Extract frames
            frames = self._extract_frames(file_path)
        except Exception as exc:
            logger.exception("Video preprocessing failed, using stub result: %s", exc)
            return DetectionResult(
                fake_probability=0.0,
                is_fake=False,
                model_version="stub",
                details={"warning": "video_preprocessing_failed", "error": str(exc)},
            )

        # Transform frames
        tensors = [self._transform(f) for f in frames]
        video_tensor = torch.stack(tensors).unsqueeze(0)  # (1, T, C, H, W)
        video_tensor = video_tensor.to(self.device)

        with torch.no_grad():
            clip_logits, _ = self._model(video_tensor)  # (1,)
            fake_prob = torch.sigmoid(clip_logits).item()

        return DetectionResult(
            fake_probability=fake_prob,
            is_fake=fake_prob >= 0.5,
            model_version=self.MODEL_VERSION,
            details={"frames_analyzed": NUM_FRAMES},
        )


# Registration in factory
DetectorFactory.register(MediaType.VIDEO, VideoDetector)
