"""
PyTorch model architectures for deepfake detection.

- image.py: DeepfakeDetector (EfficientNet-B4) for images
- audio.py: VoiceDetector (Wav2Vec2 + AASIST) for audio
- video.py: DeepfakeDetector (EfficientNet + GRU) for video
"""

from .image import DeepfakeDetector as ImageModel

# Audio and Video models are imported lazily due to heavy dependencies
# (transformers for audio, timm for video)

__all__ = ["ImageModel"]
