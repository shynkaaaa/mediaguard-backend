import uuid
from django.db import models
from django.conf import settings


class DetectionTask(models.Model):
    """Represents a single deepfake analysis request."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    class MediaType(models.TextChoices):
        IMAGE = "image", "Image"
        VIDEO = "video", "Video"
        AUDIO = "audio", "Audio"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="detection_tasks",
    )
    file = models.FileField(upload_to="uploads/%Y/%m/%d/")
    media_type = models.CharField(max_length=10, choices=MediaType.choices)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} — {self.media_type} — {self.status}"


class DetectionResult(models.Model):
    """Stores the ML model output for a DetectionTask."""

    task = models.OneToOneField(DetectionTask, on_delete=models.CASCADE, related_name="result")
    # Probability that the media is a deepfake, 0.0–1.0
    fake_probability = models.FloatField()
    is_fake = models.BooleanField()
    # Optional per-frame or per-region data (JSON)
    details = models.JSONField(null=True, blank=True)
    model_version = models.CharField(max_length=64, default="v1")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        verdict = "FAKE" if self.is_fake else "REAL"
        return f"{self.task.id} → {verdict} ({self.fake_probability:.2%})"
