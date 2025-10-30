# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Add libgomp1 for scikit-learn/pandas/sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/data /app/artifacts

# Copy only requirements first to leverage layer caching
COPY requirements.txt /app/requirements.txt

# Install system-wide (not --user) so they’re visible to any user
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Now copy the app
COPY app/ /app/

# Create non-root user and give it ownership
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Run your script (remove -h unless you only want help text)
CMD ["python", "k8s_collect.py"]
