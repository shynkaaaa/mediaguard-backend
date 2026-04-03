#!/bin/bash
# Production start script for MediaGuard Backend

set -e

echo "Starting MediaGuard Backend in PRODUCTION mode..."

# Activate virtual environment
source .venv/bin/activate

# Run migrations
echo "Running database migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

# Start gunicorn with configuration
echo "Starting Gunicorn server..."
exec gunicorn -c gunicorn.conf.py deepfake_backend.wsgi:application
