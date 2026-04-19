"""
Audio deepfake detector.

Uses VoiceDetector (Wav2Vec2 + AASIST).
"""

import logging
import os
from pathlib import Path

import torch

from ..base import BaseDetector, DetectionResult, MediaType
from ..factory import DetectorFactory

logger = logging.getLogger(__name__)

# Path to audio model weights
WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "audio" / "best_model.pth"

# Audio config
SAMPLE_RATE = 16000
MAX_DURATION_SEC = 4.0
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION_SEC)  # 64000 samples


class AudioDetector(BaseDetector):
    """
    Audio deepfake detector.

    Architecture: Wav2Vec2 + AASIST-style graph attention
    Place model at: apps/detection/ml/weights/audio/best_model.pth
    """

    MODEL_VERSION = "wav2vec2-aasist-v1"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._load_model()

    def _load_model(self):
        if not WEIGHTS_PATH.exists():
            logger.warning(
                "Audio model weights not found: %s — detector not ready",
                WEIGHTS_PATH,
            )
            return

        try:
            from ..models.audio import VoiceDetector
            self._model = VoiceDetector()
            self._model.load_state_dict(
                torch.load(str(WEIGHTS_PATH), map_location=self.device)
            )
            self._model.eval().to(self.device)
            logger.info("AudioDetector loaded from %s on %s", WEIGHTS_PATH, self.device)
        except ImportError as e:
            logger.error("Failed to import VoiceDetector: %s", e)
            logger.error("Install transformers: pip install transformers")
        except Exception as e:
            logger.error("Failed to load audio model: %s", e)

    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio."""
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required: pip install librosa")

        # Load with librosa (no TorchCodec required)
        wav, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Convert to torch tensor
        waveform = torch.from_numpy(wav).float()

        # Trim or pad to MAX_LEN
        if waveform.shape[0] > MAX_LEN:
            waveform = waveform[:MAX_LEN]
        elif waveform.shape[0] < MAX_LEN:
            padding = MAX_LEN - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Normalize to [-1, 1]
        waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform  # (T,)

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
            logger.warning("AudioDetector in stub mode for file %s", file_path)
            return DetectionResult(
                fake_probability=0.0,
                is_fake=False,
                model_version="stub",
                details={"warning": "audio_model_not_ready"},
            )

        try:
            waveform = self._load_audio(file_path)
        except Exception as exc:
            logger.exception("Audio preprocessing failed, using stub result: %s", exc)
            return DetectionResult(
                fake_probability=0.0,
                is_fake=False,
                model_version="stub",
                details={"warning": "audio_preprocessing_failed", "error": str(exc)},
            )

        waveform = waveform.unsqueeze(0).to(self.device)  # (1, T)

        with torch.no_grad():
            output = self._model(waveform)  # (1, 2) logits
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
DetectorFactory.register(MediaType.AUDIO, AudioDetector)
