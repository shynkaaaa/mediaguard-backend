"""
Detector factory for routing by media type.
"""

import logging
from typing import Dict, Type

from .base import BaseDetector, MediaType

logger = logging.getLogger(__name__)


class DetectorFactory:
    """
    Factory for creating and caching detectors.

    Detectors are created lazily on first request and cached.

    Usage:
        detector = DetectorFactory.get_detector("image")
        result = detector.predict("/path/to/file.jpg")
    """

    _instances: Dict[MediaType, BaseDetector] = {}
    _registry: Dict[MediaType, Type[BaseDetector]] = {}

    @classmethod
    def register(cls, media_type: MediaType, detector_class: Type[BaseDetector]) -> None:
        """
        Register a detector for a media type.

        Args:
            media_type: Media type (image, video, audio).
            detector_class: Detector class.
        """
        cls._registry[media_type] = detector_class
        logger.debug("Registered detector %s for %s", detector_class.__name__, media_type)

    @classmethod
    def get_detector(cls, media_type: MediaType | str) -> BaseDetector:
        """
        Get detector for specified media type.

        Args:
            media_type: Media type (string or MediaType enum).

        Returns:
            Detector instance.

        Raises:
            ValueError: If no detector registered for type.
        """
        if isinstance(media_type, str):
            media_type = MediaType(media_type)

        if media_type not in cls._instances:
            if media_type not in cls._registry:
                raise ValueError(f"No detector registered for media type: {media_type}")

            detector_class = cls._registry[media_type]
            logger.info("Creating detector instance: %s", detector_class.__name__)
            cls._instances[media_type] = detector_class()

        return cls._instances[media_type]

    @classmethod
    def is_registered(cls, media_type: MediaType | str) -> bool:
        """Check if detector is registered for type."""
        if isinstance(media_type, str):
            try:
                media_type = MediaType(media_type)
            except ValueError:
                return False
        return media_type in cls._registry

    @classmethod
    def clear_cache(cls) -> None:
        """Clear detector cache (for tests)."""
        cls._instances.clear()


def get_detector(media_type: MediaType | str) -> BaseDetector:
    """
    Convenient function for quick detector access.

    Args:
        media_type: Media type.

    Returns:
        Detector instance.
    """
    return DetectorFactory.get_detector(media_type)
