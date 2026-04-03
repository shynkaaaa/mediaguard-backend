"""
Celery task that runs the deepfake detector on an uploaded file.

The task:
  1. Loads the DetectionTask by ID.
  2. Calls the appropriate ML detector based on media_type.
  3. Saves a DetectionResult and updates the task status.
"""

import logging
from celery import shared_task
from django.db import transaction

from .models import DetectionTask, DetectionResult
from .ml import get_detector

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=10)
def run_detection(self, task_id: str):
    try:
        task = DetectionTask.objects.get(id=task_id)
    except DetectionTask.DoesNotExist:
        logger.error("DetectionTask %s not found", task_id)
        return

    task.status = DetectionTask.Status.PROCESSING
    task.save(update_fields=["status"])

    try:
        detector = get_detector(task.media_type)
        result = detector.predict(task.file.path)

        with transaction.atomic():
            DetectionResult.objects.create(
                task=task,
                fake_probability=result.fake_probability,
                is_fake=result.is_fake,
                details=result.details,
                model_version=result.model_version,
            )
            task.status = DetectionTask.Status.DONE
            task.save(update_fields=["status"])

    except Exception as exc:
        logger.exception("Detection failed for task %s", task_id)
        task.status = DetectionTask.Status.FAILED
        task.save(update_fields=["status"])
        raise self.retry(exc=exc)
