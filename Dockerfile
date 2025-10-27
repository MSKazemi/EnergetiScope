FROM python:3.11-slim

# System deps (parquet, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code
COPY ./app/* /app/
# If you keep a service (see §4) also copy predict_service.py

# NOTE: fix the filename typo: requirments.txt -> requirements.txt
COPY requirments.txt /app/requirements.txt

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Default: just print help
CMD ["python", "k8s_collect.py", "-h"]
