"""
Concrete detector implementations.
"""

from .image import ImageDetector
from .video import VideoDetector
from .audio import AudioDetector

__all__ = ["ImageDetector", "VideoDetector", "AudioDetector"]
