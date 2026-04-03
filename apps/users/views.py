from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from drf_spectacular.utils import extend_schema
from django_ratelimit.decorators import ratelimit
from django.utils.decorators import method_decorator

from .models import User
from .serializers import RegisterSerializer, UserProfileSerializer, LogoutSerializer


@method_decorator(ratelimit(key='ip', rate='5/h', method='POST'), name='dispatch')
class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]


@method_decorator(ratelimit(key='ip', rate='10/m', method='POST'), name='dispatch')
class RateLimitedTokenObtainPairView(TokenObtainPairView):
    """JWT login with rate limiting: 10 attempts per minute per IP."""
    pass


@method_decorator(ratelimit(key='ip', rate='20/m', method='POST'), name='dispatch')
class RateLimitedTokenRefreshView(TokenRefreshView):
    """JWT token refresh with rate limiting: 20 attempts per minute per IP."""
    pass


@method_decorator(ratelimit(key='ip', rate='10/m', method='POST'), name='dispatch')
class LogoutView(APIView):
    """Blacklist the refresh token to invalidate the session."""

    @extend_schema(request=LogoutSerializer, responses={204: None, 400: dict})
    def post(self, request):
        try:
            refresh_token = request.data["refresh"]
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception:
            return Response(
                {"detail": "Invalid or expired token."},
                status=status.HTTP_400_BAD_REQUEST,
            )


class ProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer

    def get_object(self):
        return self.request.user


class HealthCheckView(APIView):
    """Check if backend is alive and CORS is working."""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response(
            {
                "status": "ok",
                "message": "Backend is running and ready for connections!",
            },
            status=status.HTTP_200_OK,
        )
