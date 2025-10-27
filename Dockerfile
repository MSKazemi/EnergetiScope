# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# system deps only if you truly need them at runtime; keep minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- dependency layer (cached) ----
# copy ONLY requirements first to leverage caching
COPY requirements.txt /app/requirements.txt
RUN pip install --user -r /app/requirements.txt

# ---- app code ----
# copy the app directory preserving structure
COPY app/ /app/

# create non-root user
RUN useradd -m -u 10001 appuser
USER appuser

# default command
CMD ["python", "k8s_collect.py", "-h"]
