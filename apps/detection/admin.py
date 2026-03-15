from django.contrib import admin
from .models import DetectionTask, DetectionResult


@admin.register(DetectionTask)
class DetectionTaskAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "media_type", "status", "created_at")
    list_filter = ("status", "media_type")
    search_fields = ("user__username",)
    readonly_fields = ("id", "created_at", "updated_at")


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ("task", "is_fake", "fake_probability", "model_version", "created_at")
    list_filter = ("is_fake", "model_version")
