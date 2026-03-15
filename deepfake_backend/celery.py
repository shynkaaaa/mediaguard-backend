from celery import Celery
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepfake_backend.settings")

app = Celery("deepfake_backend")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
