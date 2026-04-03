import logging
from django.http import HttpResponseForbidden
from django.core.cache import cache

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """
    Middleware for blocking suspicious requests and protecting against attacks.
    """

    # Suspicious patterns in URL
    SUSPICIOUS_PATTERNS = [
        '/cgi-bin/',
        '/config/',
        '/hudson',
        '/actuator/',
        '/SDK/',
        'wget',
        'chmod',
        'Mozi',
        '/tmp/',
        '.env',
        '.git',
        'phpstorm',
        'XDEBUG',
        '/board.cgi',
        '/wiki',
        'ip-api.com',
        '.well-known/security.txt',
    ]

    # Suspicious User-Agent patterns
    SUSPICIOUS_AGENTS = [
        'masscan',
        'nmap',
        'nikto',
        'sqlmap',
        'python-requests',  # Often used in attacks
    ]

    # Methods that should not be allowed
    BLOCKED_METHODS = ['CONNECT', 'PRI']

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check for suspicious patterns in URL
        path = request.path.lower()
        query = request.META.get('QUERY_STRING', '').lower()

        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.lower() in path or pattern.lower() in query:
                ip = self.get_client_ip(request)
                logger.warning(f"Blocked suspicious request from {ip}: {request.path}")
                self.ban_ip(ip)
                return HttpResponseForbidden("Access Denied")

        # Check User-Agent
        user_agent = request.META.get('HTTP_USER_AGENT', '').lower()
        for agent in self.SUSPICIOUS_AGENTS:
            if agent.lower() in user_agent:
                ip = self.get_client_ip(request)
                logger.warning(f"Blocked suspicious User-Agent from {ip}: {user_agent}")
                self.ban_ip(ip)
                return HttpResponseForbidden("Access Denied")

        # Check request method
        if request.method in self.BLOCKED_METHODS:
            ip = self.get_client_ip(request)
            logger.warning(f"Blocked {request.method} request from {ip}")
            return HttpResponseForbidden("Access Denied")

        # Check if IP is banned
        ip = self.get_client_ip(request)
        if self.is_ip_banned(ip):
            logger.warning(f"Blocked request from banned IP: {ip}")
            return HttpResponseForbidden("Access Denied - IP Banned")

        response = self.get_response(request)
        return response

    def get_client_ip(self, request):
        """Get real client IP considering proxy."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    def ban_ip(self, ip, duration=3600):
        """Temporarily ban IP for duration seconds (default 1 hour)."""
        cache_key = f"banned_ip_{ip}"
        cache.set(cache_key, True, duration)
        logger.warning(f"IP {ip} has been temporarily banned for {duration} seconds")

    def is_ip_banned(self, ip):
        """Check if IP is banned."""
        cache_key = f"banned_ip_{ip}"
        return cache.get(cache_key, False)
