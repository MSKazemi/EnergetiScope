#!/bin/bash

python3 k8s_collect.py watch \
--kinds Deployment Job CronJob Pod \
--emit-initial \
--suppress-tls-warnings > ./data/inference_requests.ndjson

python3 kepler_labels.py \
  --prom http://prometheus.n1.local \
  --owner-source auto \
  --mode window \
  --start $(date -u -d '5 days ago' +%s) \
  --end   $(date -u +%s) \
  --out ./data/kepler_labels.parquet
  --suppress-tls-warnings



python3 join_features_labels.py \
  --features ./data/features.parquet \
  --labels   ./data/kepler_labels.parquet \
  --out      ./data/train_rows.parquet
