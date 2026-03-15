from django.urls import path
from .views import AnalyzeView, DetectionTaskListView, DetectionTaskDetailView

urlpatterns = [
    path("analyze/", AnalyzeView.as_view(), name="detection-analyze"),
    path("tasks/", DetectionTaskListView.as_view(), name="detection-task-list"),
    path("tasks/<uuid:pk>/", DetectionTaskDetailView.as_view(), name="detection-task-detail"),
]
