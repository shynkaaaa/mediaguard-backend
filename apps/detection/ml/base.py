"""
Base classes for deepfake detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MediaType(str, Enum):
    """Supported media types."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class DetectionResult:
    """Standardized detection result."""
    fake_probability: float
    is_fake: bool
    model_version: str
    details: Optional[dict] = None
    # For video — results per frame
    frame_results: Optional[list] = None
    # For audio — results per segment
    segment_results: Optional[list] = None

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        result = {
            "fake_probability": self.fake_probability,
            "is_fake": self.is_fake,
            "model_version": self.model_version,
            "details": self.details,
        }
        if self.frame_results is not None:
            result["frame_results"] = self.frame_results
        if self.segment_results is not None:
            result["segment_results"] = self.segment_results
        return result


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.

    Each detector must implement:
    - predict() — main detection method
    - is_ready — readiness check (model loaded)
    - model_version — model version for tracking
    """

    @abstractmethod
    def predict(self, file_path: str) -> DetectionResult:
        """
        Run detection on file.

        Args:
            file_path: Absolute path to media file.

        Returns:
            DetectionResult with analysis results.

        Raises:
            FileNotFoundError: If file not found.
            RuntimeError: If model not loaded.
        """
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check detector readiness (model loaded)."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version for tracking."""
        pass
