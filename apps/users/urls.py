from django.urls import path
from .views import (
    RegisterView, LogoutView, ProfileView, HealthCheckView,
    RateLimitedTokenObtainPairView, RateLimitedTokenRefreshView
)

urlpatterns = [
    path("health/", HealthCheckView.as_view(), name="auth-health"),
    path("register/", RegisterView.as_view(), name="auth-register"),
    path("login/", RateLimitedTokenObtainPairView.as_view(), name="auth-login"),
    path("token/refresh/", RateLimitedTokenRefreshView.as_view(), name="auth-token-refresh"),
    path("logout/", LogoutView.as_view(), name="auth-logout"),
    path("profile/", ProfileView.as_view(), name="auth-profile"),
]
