# Deploy to Railway (Django + PostgreSQL)

This guide deploys backend only.

## 1. Push project to GitHub

Railway deploys from a GitHub repo.

## 2. Create PostgreSQL on Railway

1. In Railway: New Project -> Provision PostgreSQL.
2. Open PostgreSQL service and copy connection values:
- PGHOST
- PGPORT
- PGDATABASE
- PGUSER
- PGPASSWORD

## 3. Create Web Service from GitHub repo

1. New Service -> GitHub Repo.
2. Select this repository.
3. Railway should detect `Procfile` automatically.

`Procfile` in this project runs:
- `migrate`
- `collectstatic`
- `gunicorn`

## 4. Set environment variables in Railway (Web service)

Required:
- `SECRET_KEY` = long random string
- `DEBUG` = `False`
- `ALLOWED_HOSTS` = `.up.railway.app`
- `DB_HOST` = `${{Postgres.PGHOST}}`
- `DB_PORT` = `${{Postgres.PGPORT}}`
- `DB_NAME` = `${{Postgres.PGDATABASE}}`
- `DB_USER` = `${{Postgres.PGUSER}}`
- `DB_PASSWORD` = `${{Postgres.PGPASSWORD}}`

Optional (if frontend deployed):
- `CORS_ALLOWED_ORIGINS` = `https://your-frontend-domain.com`
- `CSRF_TRUSTED_ORIGINS` = `https://your-frontend-domain.com,https://*.up.railway.app`

## 5. Trigger deploy

- Open Deployments tab and click Deploy.
- Wait until status is Success.

## 6. Verify backend

1. Open generated Railway URL.
2. Open Swagger docs:
- `/api/docs/`

Example:
- `https://your-service.up.railway.app/api/docs/`

## 7. Common issues

### 502/Crash loop
- Check logs for missing env vars (`SECRET_KEY`, DB_*).
- Ensure PostgreSQL service is running.

### Static files missing (admin broken styles)
- `collectstatic` runs in `Procfile`; check deploy logs for errors.

### CORS/CSRF errors
- Add exact frontend origin to:
  - `CORS_ALLOWED_ORIGINS`
  - `CSRF_TRUSTED_ORIGINS`

### Large ML dependencies
- `torch` can increase build time and memory usage.
- Use paid Railway plan or optimize model package strategy if build fails.
