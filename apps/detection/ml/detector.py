"""
DeepfakeDetector — интерфейс к ML модели EfficientNet-B4.

Модель обучена на 140k Real vs Fake Faces:
https://github.com/firdavsm19/deepfake-detector

Необходимо:
  1. Положить файл весов в apps/detection/ml/checkpoints/best_model.pth
  2. Убедиться что torch и torchvision установлены (см. requirements.txt)
"""

import logging
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import Config
from .model import DeepfakeDetectorModel

logger = logging.getLogger(__name__)

# Путь к весам — папка checkpoints/ рядом с этим файлом
WEIGHTS_PATH = Path(__file__).parent / "checkpoints" / "best_model.pth"

# Тот же препроцессинг что в val_test_transform из оригинального репо
_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        if not WEIGHTS_PATH.exists():
            logger.warning(
                "Файл весов не найден: %s — детектор работает в заглушке-режиме (всегда REAL)",
                WEIGHTS_PATH,
            )
            return
        self.model = DeepfakeDetectorModel()
        self.model.load_state_dict(
            torch.load(str(WEIGHTS_PATH), map_location=self.device)
        )
        self.model.eval().to(self.device)
        logger.info("Модель загружена с %s на %s", WEIGHTS_PATH, self.device)

    def predict(self, file_path: str) -> dict:
        """
        Запускает инференс на изображении.

        Args:
            file_path: Абсолютный путь к загруженному файлу.

        Returns:
            {
                "fake_probability": float,   # 0.0–1.0
                "is_fake":          bool,
                "model_version":    str,
                "details":          dict | None,
            }
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # Если веса не загружены — возвращаем заглушку
        if self.model is None:
            return {"fake_probability": 0.0, "is_fake": False,
                    "model_version": "stub", "details": None}

        image = Image.open(file_path).convert("RGB")
        tensor = _transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()

        return {
            "fake_probability": fake_prob,
            "is_fake": fake_prob >= 0.5,
            "model_version": "efficientnet-b4-v1",
            "details": {"real_probability": real_prob, "fake_probability": fake_prob},
        }

