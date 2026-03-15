# Local Testing Guide (No Frontend)

This guide verifies backend health locally using PostgreSQL + Docker, without frontend dependency.

## 1. Start infrastructure

```powershell
docker compose up -d postgres pgadmin
```

Check status:

```powershell
docker compose ps
```

Expected:
- `deepfake-postgres` is `healthy`
- `deepfake-pgadmin` is `running`

## 2. Configure backend env

Use these values in `.env`:

```env
DB_NAME=deepfake_db
DB_USER=deepfake_user
DB_PASSWORD=deepfake_password
DB_HOST=localhost
DB_PORT=5432
```

If your project uses `.venv`:

```powershell
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py check
```

## 3. Run backend

```powershell
python manage.py runserver
```

API base URL: `http://127.0.0.1:8000`

## 4. Smoke test API from PowerShell

Register:

```powershell
$registerBody = @{
  username = "local_user"
  email = "local_user@example.com"
  password = "StrongPass123!"
  password2 = "StrongPass123!"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/auth/register/" -Method Post -ContentType "application/json" -Body $registerBody
```

Login (get JWT):

```powershell
$loginBody = @{
  username = "local_user"
  password = "StrongPass123!"
} | ConvertTo-Json

$tokens = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/auth/login/" -Method Post -ContentType "application/json" -Body $loginBody
$access = $tokens.access
$refresh = $tokens.refresh
```

Profile check (authorized):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/auth/profile/" -Method Get -Headers @{ Authorization = "Bearer $access" }
```

Upload image/video for detection:

```powershell
$headers = @{ Authorization = "Bearer $access" }
$form = @{ file = Get-Item "D:\path\to\sample.jpg" }

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/detection/analyze/" -Method Post -Headers $headers -Form $form
```

List tasks:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/detection/tasks/" -Method Get -Headers @{ Authorization = "Bearer $access" }
```

## 5. Open pgAdmin

URL: `http://localhost:5050`

Login with:
- Email: `admin@local.dev`
- Password: `admin123`

Add server manually:
- Host name/address: `postgres`
- Port: `5432`
- Maintenance DB: `deepfake_db`
- Username: `deepfake_user`
- Password: `deepfake_password`

Why host is `postgres`:
- pgAdmin runs inside Docker network, so it connects to PostgreSQL by service name.

## 6. Shutdown

```powershell
docker compose down
```

To remove DB data too:

```powershell
docker compose down -v
```
