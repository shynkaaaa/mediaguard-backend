from rest_framework import serializers
from .models import DetectionTask, DetectionResult


class DetectionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionResult
        fields = ("fake_probability", "is_fake", "details", "model_version", "created_at")


class DetectionTaskSerializer(serializers.ModelSerializer):
    result = DetectionResultSerializer(read_only=True)

    class Meta:
        model = DetectionTask
        fields = ("id", "file", "media_type", "status", "created_at", "updated_at", "result")
        read_only_fields = ("id", "status", "created_at", "updated_at", "result")


class AnalyzeSerializer(serializers.Serializer):
    """Input serializer for POST /analyze/"""

    file = serializers.FileField()

    def validate_file(self, value):
        content_type = getattr(value, "content_type", "")
        if content_type.startswith("image/"):
            self._media_type = "image"
        elif content_type.startswith("video/"):
            self._media_type = "video"
        else:
            raise serializers.ValidationError(
                "Unsupported file type. Upload an image (JPEG/PNG) or video (MP4/AVI/MOV)."
            )
        return value
