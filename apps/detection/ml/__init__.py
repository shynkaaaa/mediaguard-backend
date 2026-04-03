"""
ML module for deepfake detection.

Usage:
    from apps.detection.ml import get_detector, MediaType

    detector = get_detector(MediaType.IMAGE)
    result = detector.predict("/path/to/image.jpg")
"""

from .base import BaseDetector, DetectionResult, MediaType
from .factory import DetectorFactory, get_detector

# Import detectors for auto-registration in factory
from . import detectors  # noqa: F401

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "MediaType",
    "DetectorFactory",
    "get_detector",
]
