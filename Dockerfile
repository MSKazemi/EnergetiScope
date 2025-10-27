FROM python:3.11-slim

# System deps (parquet, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- dependency layer (cached) ----
# copy ONLY requirements first to leverage caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- app code ----
# copy the app directory preserving structure
COPY app/* /app/

# Default: just print help
CMD ["python", "k8s_collect.py", "-h"]


# ===================================================================
# FROM python:3.11-slim

# # System deps (parquet, build tools)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential git curl && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy code
# COPY k8s_collect.py k8s_encode.py kepler_labels.py join_features_labels.py train_power.py predict_k8s.py models.py /app/
# # If you keep a service (see §4) also copy predict_service.py

# # NOTE: fix the filename typo: requirments.txt -> requirements.txt
# COPY requirments.txt /app/requirements.txt

# # Install deps
# RUN pip install --no-cache-dir -r requirements.txt

# # Default: just print help
# CMD ["python", "k8s_collect.py", "-h"]

# ====================================================================
# # syntax=docker/dockerfile:1.6
# FROM python:3.11-slim AS runtime

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PATH="/home/appuser/.local/bin:${PATH}"

# # system deps only if you truly need them at runtime; keep minimal
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl ca-certificates \
#  && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # ---- dependency layer (cached) ----
# # copy ONLY requirements first to leverage caching
# COPY requirements.txt /app/requirements.txt
# RUN pip install --user -r /app/requirements.txt

# # ---- app code ----
# # copy the app directory preserving structure
# COPY app/ /app/

# # create non-root user
# RUN useradd -m -u 10001 appuser
# USER appuser

# # default command
# CMD ["python", "k8s_collect.py", "-h"]
