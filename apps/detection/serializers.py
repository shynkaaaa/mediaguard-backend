import logging
from rest_framework import serializers
from .models import DetectionTask, DetectionResult

logger = logging.getLogger(__name__)


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


class DetectionTaskStatusSerializer(serializers.ModelSerializer):
    result = DetectionResultSerializer(read_only=True)
    progress = serializers.SerializerMethodField()
    is_finished = serializers.SerializerMethodField()

    class Meta:
        model = DetectionTask
        fields = ("id", "media_type", "status", "progress", "is_finished", "updated_at", "result")
        read_only_fields = fields

    def get_progress(self, obj):
        if obj.status == DetectionTask.Status.PENDING:
            return 5
        if obj.status == DetectionTask.Status.PROCESSING:
            return 60
        return 100

    def get_is_finished(self, obj):
        return obj.status in (DetectionTask.Status.DONE, DetectionTask.Status.FAILED)


class AnalyzeSerializer(serializers.Serializer):
    """Input serializer for POST /analyze/"""

    file = serializers.FileField()
    # Default upload limits by media type.
    MAX_SIZE_BY_TYPE = {
        "image": 10 * 1024 * 1024,   # 10 MB
        "video": 200 * 1024 * 1024,  # 200 MB
        "audio": 100 * 1024 * 1024,  # 100 MB
    }

    def _validate_size(self, file_obj):
        media_type = getattr(self, "_media_type", None)
        if not media_type:
            return

        max_size = self.MAX_SIZE_BY_TYPE.get(media_type)
        if max_size is None:
            return

        file_size = getattr(file_obj, "size", 0) or 0
        if file_size > max_size:
            max_mb = max_size // (1024 * 1024)
            current_mb = round(file_size / (1024 * 1024), 2)
            raise serializers.ValidationError(
                f"File is too large for {media_type}. "
                f"Maximum allowed size is {max_mb} MB, got {current_mb} MB."
            )

    def validate(self, data):
        file_obj = data.get("file")
        if not file_obj:
            return data

        filename = getattr(file_obj, "name", "").lower()
        content_type = getattr(file_obj, "content_type", "") or ""

        logger.debug(f"AnalyzeSerializer validation: filename={filename}, content_type={content_type}")

        # Try to determine type from content_type
        if content_type.startswith("image/"):
            self._media_type = "image"
            logger.debug("Detected as image (content_type)")
        elif content_type.startswith("video/"):
            self._media_type = "video"
            logger.debug("Detected as video (content_type)")
        elif content_type.startswith("audio/"):
            self._media_type = "audio"
            logger.debug("Detected as audio (content_type)")
        else:
            # Fallback: try to determine by file extension if content_type didn't help
            if any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
                self._media_type = "image"
                logger.debug("Detected as image (extension)")
            elif any(filename.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]):
                self._media_type = "video"
                logger.debug("Detected as video (extension)")
            elif any(filename.endswith(ext) for ext in [".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"]):
                self._media_type = "audio"
                logger.debug("Detected as audio (extension)")
            else:
                logger.error(f"Unrecognized file: {filename}, content_type: {content_type}")
                raise serializers.ValidationError(
                    "Unsupported file type. Upload an image (jpg, png...), video (mp4, avi...), or audio (mp3, wav, flac...)."
                )

        self._validate_size(file_obj)

        return data
