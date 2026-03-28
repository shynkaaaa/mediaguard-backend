from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny


class RootView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({
            "name": "MediaGuard Backend",
            "version": "1.0.0",
            "status": "running",
            "docs": "/api/docs/",
            "schema": "/api/schema/",
            "endpoints": {
                "auth": "/api/auth/",
                "detection": "/api/detection/",
                "admin": "/admin/",
            }
        })


urlpatterns = [
    path("", RootView.as_view(), name="root"),
    path("admin/", admin.site.urls),
    path("api/schema/", SpectacularAPIView.as_view(), name="api-schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="api-schema"), name="api-docs"),
    path("api/redoc/", SpectacularRedocView.as_view(url_name="api-schema"), name="api-redoc"),
    path("api/auth/", include("apps.users.urls")),
    path("api/detection/", include("apps.detection.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
