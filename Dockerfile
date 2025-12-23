# =======================================
# Dockerfile for Spartan Serverless Framework
# =======================================

# =======================================
# 1️⃣ Base image
# =======================================
FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Sydel Palinlin <sydel.palinlin@gmail.com>" \
      version="1.0.0" \
      description="Spartan Serverless Framework - Swiss Army knife for serverless development" \
      org.opencontainers.image.title="Spartan Serverless Framework" \
      org.opencontainers.image.description="A powerful scaffold for serverless applications on GCP" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Sydel Palinlin" \
      org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# =======================================
# 2️⃣ System dependencies
# =======================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =======================================
# 3️⃣ Install Poetry and dependencies
# =======================================
RUN pip install --upgrade pip poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

# =======================================
# 4️⃣ Copy app source
# =======================================
COPY . .

# =======================================
# 4.1️⃣ Environment-specific configuration
# =======================================
ARG BUILD_ENV=production
COPY .env.${BUILD_ENV} .env

# =======================================
# 5️⃣ Set up runtime user (security)
# =======================================
RUN useradd -m spartan
USER spartan

# =======================================
# 6️⃣ Expose port and define entrypoint
# =======================================
EXPOSE 8080
CMD ["python", "main.py"]
