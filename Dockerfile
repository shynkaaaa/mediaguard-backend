FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create directory for model weights
RUN mkdir -p /app/apps/detection/ml/checkpoints

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "deepfake_backend.wsgi:application"]
