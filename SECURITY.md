# Security

## Implemented Security Features

### Protection Mechanisms

- **Rate Limiting** - Prevents brute-force attacks
  - Login: 10 attempts/minute per IP
  - Registration: 5 attempts/hour per IP
  - API requests: 10 requests/minute per user

- **Malicious Request Blocking** - Automatic detection and blocking
  - Known attack patterns (SQL injection, command injection)
  - Suspicious User-Agents (scanning tools)
  - Temporary IP bans (1 hour) for detected threats

- **Security Headers**
  - XSS protection
  - Content-Type sniffing prevention
  - Clickjacking protection (X-Frame-Options)

- **CORS & CSRF Protection**
  - Configurable allowed origins
  - CSRF tokens for state-changing operations

- **JWT Authentication**
  - Token blacklisting on logout
  - Automatic token refresh
  - Secure cookie settings in production

### Security Monitoring

All security events are logged for analysis.

## Reporting Security Issues

If you discover a security vulnerability, please create an issue or contact directly.

---

**Note:** For production deployment, ensure proper configuration of environment variables (`DEBUG=False`, strong `SECRET_KEY`, restricted `ALLOWED_HOSTS`).
