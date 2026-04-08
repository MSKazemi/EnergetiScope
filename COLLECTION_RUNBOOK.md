# Benchmark Collection Runbook (2026-04-08)

## What was done

### Phase 1 — New Manifests (completed)
Added 5 diverse real-world workload manifests to `manifests/bench/`:

| Manifest | Kind | Image | CPU | Mem |
|---|---|---|---|---|
| `deploy-redis-500m-256mi.yaml` | Deployment | `redis:7-alpine` | 500m | 256Mi |
| `deploy-nginx-250m-128mi.yaml` | Deployment | `nginx:alpine` | 250m | 128Mi |
| `deploy-postgres-500m-512mi.yaml` | Deployment | `postgres:15-alpine` | 500m | 512Mi |
| `job-python-cpu-500m-256mi.yaml` | Job | `python:3.11-slim` | 500m | 256Mi |
| `deploy-memcached-250m-256mi.yaml` | Deployment | `memcached:alpine` | 250m | 256Mi |

Updated `eval/collect_bench.sh` to include these 5 new manifests.

### Phase 2 — Data Collection (running)
Started in tmux session `bench` on 2026-04-08.

**Command:**
```bash
./eval/collect_bench.sh \
  --prom http://localhost:32090 \
  --reps 3 \
  --warmup 300 \
  --cooldown 60
```

**Runtime:** ~5.5 hours (18 manifests × 3 reps × ~6 min each)

**Monitor progress:**
```bash
tmux attach -t bench
# Ctrl+B, D to detach again
```

**Log file:** `bench_collection.log` (tee'd from tmux)

**Output locations:**
- Features: `data/bench/features/*.ndjson`
- Labels: `data/bench/labels/*.parquet`
- Combined features: `data/bench/combined_features.ndjson`

---

## What to do after collection finishes

### Step 1: Verify collection output
```bash
# Check feature files exist for new workloads
ls data/bench/features/ | grep -E 'redis|nginx|postgres|python|memcached'

# Count total feature files (expect 54 = 18 manifests × 3 reps)
ls data/bench/features/*.ndjson | wc -l

# Count total label files
ls data/bench/labels/*.parquet | wc -l
```

### Step 2: Combine labels
```bash
python3 eval/combine_labels.py \
  --input-dir data/bench/labels \
  --out data/bench/labels.parquet
```

### Step 3: SBERT Ablation — Condition A (numeric only)
```bash
# Fit encoder WITHOUT SBERT
python3 app/k8s_encode.py fit \
  --input data/bench/combined_features.ndjson \
  --out data/artifacts/encoder_no_sbert.joblib \
  --no-sbert

# Transform
python3 app/k8s_encode.py transform \
  --input data/bench/combined_features.ndjson \
  --encoder data/artifacts/encoder_no_sbert.joblib \
  --out data/bench/features_no_sbert.parquet

# Join with labels
python3 app/join_features_labels.py \
  --features data/bench/features_no_sbert.parquet \
  --labels data/bench/labels.parquet \
  --out data/bench/train_rows_no_sbert.parquet

# Train
python3 app/train_power.py \
  --train data/bench/train_rows_no_sbert.parquet \
  --target energy_step_j \
  --out data/artifacts/knn_no_sbert.joblib
```

### Step 4: SBERT Ablation — Condition B (numeric + SBERT)
```bash
# Fit encoder WITH SBERT
python3 app/k8s_encode.py fit \
  --input data/bench/combined_features.ndjson \
  --out data/artifacts/encoder_sbert.joblib

# Transform
python3 app/k8s_encode.py transform \
  --input data/bench/combined_features.ndjson \
  --encoder data/artifacts/encoder_sbert.joblib \
  --out data/bench/features_sbert.parquet

# Join
python3 app/join_features_labels.py \
  --features data/bench/features_sbert.parquet \
  --labels data/bench/labels.parquet \
  --out data/bench/train_rows_sbert.parquet

# Train
python3 app/train_power.py \
  --train data/bench/train_rows_sbert.parquet \
  --target energy_step_j \
  --out data/artifacts/knn_sbert.joblib
```

### Step 5: Evaluate both conditions
```bash
# Evaluate numeric-only
python3 eval/eval_existing_model.py \
  --features data/bench/features_no_sbert.parquet \
  --labels data/bench/labels.parquet \
  | tee eval/results_no_sbert.txt

# Evaluate numeric+SBERT
python3 eval/eval_existing_model.py \
  --features data/bench/features_sbert.parquet \
  --labels data/bench/labels.parquet \
  | tee eval/results_sbert.txt
```

### Step 6: Update paper
Add two new rows to Table II in `paper/main.tex`:

| Evaluation | n | MAE | R² | MAPE |
|---|---|---|---|---|
| Bench CV — numeric only (expanded) | ? | ? | ? | ? |
| Bench CV — numeric + SBERT | ? | ? | ? | ? |

Update §V-A with expanded workload count and SBERT ablation discussion.

---

## Troubleshooting

- **Zero energy detected:** Check Kepler pod logs: `kubectl logs -n kepler -l app.kubernetes.io/name=kepler`
- **Pod stuck in Pending:** Node may be overloaded — check `kubectl describe pod -n energetiscope-bench`
- **Collection script crashed:** Rerun with `--reps 1` for the failed manifest only, or restart from where it left off
- **tmux session gone:** Check `bench_collection.log` to see where it stopped, then rerun remaining manifests
