# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project ───────────────────────────────────────────────────────────────
COPY . .

# ── Create required directories ────────────────────────────────────────────────
RUN mkdir -p \
        data/raw \
        data/processed \
        models/saved \
        logs \
        reports/plots

# ── Python path ────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app

# ── Default: run the batch pipeline ───────────────────────────────────────────
CMD ["python", "-m", "src.pipeline.batch_pipeline"]
