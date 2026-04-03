"""
Image deepfake detector based on EfficientNet-B4.
"""

import logging
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ..base import BaseDetector, DetectionResult, MediaType
from ..config import Config
from ..factory import DetectorFactory
from ..models.image import DeepfakeDetector as ImageModel

logger = logging.getLogger(__name__)

# Path to weights
WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "image" / "best_model.pth"

# ImageNet normalization
_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageDetector(BaseDetector):
    """
    Image deepfake detector.

    Uses EfficientNet-B4 trained on 140k Real vs Fake Faces.
    """

    MODEL_VERSION = "efficientnet-b4-v1"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._load_model()

    def _load_model(self):
        if not WEIGHTS_PATH.exists():
            logger.warning(
                "Weights not found: %s — detector in stub mode (always REAL)",
                WEIGHTS_PATH,
            )
            return

        self._model = ImageModel()
        self._model.load_state_dict(
            torch.load(str(WEIGHTS_PATH), map_location=self.device)
        )
        self._model.eval().to(self.device)
        logger.info("ImageDetector loaded from %s on %s", WEIGHTS_PATH, self.device)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION if self.is_ready else "stub"

    def predict(self, file_path: str) -> DetectionResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Stub mode
        if not self.is_ready:
            return DetectionResult(
                fake_probability=0.0,
                is_fake=False,
                model_version="stub",
                details=None,
            )

        image = Image.open(file_path).convert("RGB")
        tensor = _transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(tensor)
            probs = torch.softmax(output, dim=1)
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()

        return DetectionResult(
            fake_probability=fake_prob,
            is_fake=fake_prob >= 0.5,
            model_version=self.MODEL_VERSION,
            details={"real_probability": real_prob, "fake_probability": fake_prob},
        )


# Register in factory
DetectorFactory.register(MediaType.IMAGE, ImageDetector)
