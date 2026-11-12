# Multi-purpose Dockerfile for Movie Recommendation System
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY . .

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Default command: run FastAPI backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

