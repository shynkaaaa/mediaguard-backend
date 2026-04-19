# Deepfake Detection — Django Backend

REST API backend for the Deepfake Detection project.

## Tech Stack
| Layer | Technology |
|---|---|
| Framework | Django 4.2 + Django REST Framework |
| Auth | JWT via `djangorestframework-simplejwt` |
| Async tasks | Celery + Redis(in future) |
| DB | PostgreSQL |

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and set SECRET_KEY and other values

# 4. Start PostgreSQL (+ pgAdmin) in Docker
docker compose up -d postgres pgadmin

# 5. Run migrations
python manage.py migrate

# 6. Create superuser (optional)
python manage.py createsuperuser

# 7. Start Django dev server
python manage.py runserver

# 8. Start Celery worker (separate terminal)
celery -A deepfake_backend worker --loglevel=info

# 9. Open API docs (Swagger UI)
# http://127.0.0.1:8000/api/docs/
```

## Local Infrastructure

1. Start DB + pgAdmin:
```bash
docker compose up -d postgres pgadmin
```

2. In `.env`, set PostgreSQL settings:
```env
DB_NAME=deepfake_db
DB_USER=deepfake_user
DB_PASSWORD=deepfake_password
DB_HOST=localhost
DB_PORT=5432
```

3. Reinstall dependencies (if needed):
```bash
pip install -r requirements.txt
```

4. Apply migrations:
```bash
python manage.py migrate
```

5. Open pgAdmin:
```text
http://localhost:5050
```

6. Stop services when done:
```bash
docker compose down
```

## Backend-Only Local Validation

Step-by-step smoke tests (register/login/profile/detection) are documented in [LOCAL_TESTING.md](LOCAL_TESTING.md).

## API Docs (Swagger / Redoc)

- Swagger UI: [http://127.0.0.1:8000/api/docs/](http://127.0.0.1:8000/api/docs/)
- Redoc: [http://127.0.0.1:8000/api/redoc/](http://127.0.0.1:8000/api/redoc/)
- OpenAPI schema (JSON): [http://127.0.0.1:8000/api/schema/](http://127.0.0.1:8000/api/schema/)

## API Reference

### Authentication  `POST /api/auth/`

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/register/` | ❌ | Create account |
| POST | `/api/auth/login/` | ❌ | Get JWT tokens |
| POST | `/api/auth/token/refresh/` | ❌ | Refresh access token |
| POST | `/api/auth/logout/` | ✅ | Blacklist refresh token |
| GET | `/api/auth/profile/` | ✅ | Get current user |
| PATCH | `/api/auth/profile/` | ✅ | Update profile |

### Detection  `POST /api/detection/`

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/detection/analyze/` | Upload file, start detection |
| GET | `/api/detection/tasks/` | List my tasks |
| GET | `/api/detection/tasks/{id}/status/` | Lightweight status for polling |
| GET | `/api/detection/tasks/{id}/` | Task detail + result |
| DELETE | `/api/detection/tasks/{id}/` | Delete task |

#### Example: upload for analysis
```http
POST /api/detection/analyze/
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

file=<image.jpg or video.mp4>
```
Response `200 OK`:
```json
{
  "id": "3fa85f64-...",
  "file": "/media/uploads/2024/01/01/image.jpg",
  "media_type": "image",
  "status": "done",
  "created_at": "2024-01-01T12:00:00Z",
  "result": {
    "fake_probability": 0.93,
    "is_fake": true,
    "model_version": "v1",
    "details": null
  }
}
```

## Integrating Your ML Model

Open `apps/detection/ml/detector.py` and replace `_run_model()` with your actual inference:

```python
def _run_model(self, file_path: str) -> float:
    # Example with PyTorch:
    img = preprocess(file_path)
    with torch.no_grad():
        prob = self.model(img).sigmoid().item()
    return prob
```

## Project Structure

```
deepfake_backend/
├── manage.py
├── requirements.txt
├── .env.example
├── deepfake_backend/       # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── celery.py
└── apps/
    ├── users/              # auth + profile
    └── detection/          # detection tasks + ML
        └── ml/
            └── detector.py # ← plug your model here
```
