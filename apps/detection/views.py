import logging

from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema
from django_ratelimit.decorators import ratelimit
from django.utils.decorators import method_decorator

from .models import DetectionTask, DetectionResult
from .serializers import DetectionTaskSerializer, AnalyzeSerializer
from .ml import get_detector

logger = logging.getLogger(__name__)


@method_decorator(ratelimit(key='user', rate='10/m', method='POST'), name='dispatch')
class AnalyzeView(APIView):
    """
    POST /api/detection/analyze/

    Upload an image or video for deepfake analysis.
    Detection runs synchronously and the result is returned immediately.
    Rate limited to 10 requests per minute per user.
    """

    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(request=AnalyzeSerializer, responses={200: DetectionTaskSerializer})
    def post(self, request):
        serializer = AnalyzeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data["file"]
        media_type = getattr(serializer, "_media_type", "image")

        task = DetectionTask.objects.create(
            user=request.user,
            file=uploaded_file,
            media_type=media_type,
            status=DetectionTask.Status.PROCESSING,
        )

        try:
            detector = get_detector(media_type)
            result = detector.predict(task.file.path)
            DetectionResult.objects.create(
                task=task,
                fake_probability=result.fake_probability,
                is_fake=result.is_fake,
                details=result.details,
                model_version=result.model_version,
            )
            task.status = DetectionTask.Status.DONE
        except Exception as e:
            logger.exception("Detection failed for task %s: %s", task.id, e)
            task.status = DetectionTask.Status.FAILED

        task.save(update_fields=["status"])

        return Response(
            DetectionTaskSerializer(task).data,
            status=status.HTTP_200_OK,
        )


class DetectionTaskListView(generics.ListAPIView):
    """GET /api/detection/tasks/ — list authenticated user's tasks."""

    queryset = DetectionTask.objects.none()
    serializer_class = DetectionTaskSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return DetectionTask.objects.none()
        return DetectionTask.objects.filter(user=self.request.user).select_related("result")


class DetectionTaskDetailView(generics.RetrieveDestroyAPIView):
    """GET / DELETE /api/detection/tasks/{id}/"""

    queryset = DetectionTask.objects.none()
    serializer_class = DetectionTaskSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return DetectionTask.objects.none()
        return DetectionTask.objects.filter(user=self.request.user).select_related("result")
