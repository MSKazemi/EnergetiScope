#!/usr/bin/env bash
# =============================================================================
# EnergetiScope Benchmark Data Collection Script
#
# Deploys each manifest, waits for steady state, collects features + Kepler
# labels, then cleans up. Repeats for each repetition.
#
# Usage:
#   ./eval/collect_bench.sh [OPTIONS]
#
# Options:
#   --prom URL          Prometheus base URL (default: http://localhost:9090)
#   --reps N            Repetitions per manifest (default: 3)
#   --warmup N          Seconds to wait for steady state (default: 300)
#   --cooldown N        Seconds between manifests (default: 60)
#   --out-dir PATH      Output directory (default: data/bench)
#   --ns NAMESPACE      Benchmark namespace (default: energetiscope-bench)
#   --dry-run           Print commands without executing
#
# Requirements:
#   - kubectl configured and pointing at your cluster
#   - Kepler installed and scraping to Prometheus
#   - Python env with project dependencies (run from project root)
#
# Example:
#   ./eval/collect_bench.sh --prom http://prometheus.lpt.local --reps 3
# =============================================================================
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
PROM_URL="http://localhost:9090"
REPS=3
WARMUP=300      # 5 minutes steady state
COOLDOWN=60     # 1 minute between manifests
OUT_DIR="data/bench"
NS="energetiscope-bench"
DRY_RUN=false
MANIFEST_DIR="manifests/bench"

# ── arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --prom)      PROM_URL="$2";  shift 2 ;;
    --reps)      REPS="$2";      shift 2 ;;
    --warmup)    WARMUP="$2";    shift 2 ;;
    --cooldown)  COOLDOWN="$2";  shift 2 ;;
    --out-dir)   OUT_DIR="$2";   shift 2 ;;
    --ns)        NS="$2";        shift 2 ;;
    --dry-run)   DRY_RUN=true;   shift ;;
    *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
  esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ts_now() { date +%s; }

wait_for_pods_ready() {
  local label="$1"
  local timeout="${2:-120}"
  log "Waiting for pods with label $label to be Ready (timeout: ${timeout}s)..."
  run kubectl wait pod \
    -n "$NS" \
    -l "$label" \
    --for=condition=Ready \
    --timeout="${timeout}s" 2>/dev/null || true
}

wait_for_job_running() {
  local job_name="$1"
  log "Waiting for Job $job_name to start running..."
  for i in $(seq 1 30); do
    local status
    status=$(kubectl get job "$job_name" -n "$NS" \
      -o jsonpath='{.status.active}' 2>/dev/null || echo "0")
    if [[ "$status" == "1" ]]; then
      log "Job $job_name is running."
      return 0
    fi
    sleep 5
  done
  log "[WARN] Job $job_name did not start within 150s — continuing anyway."
}

collect_features() {
  local out_file="$1"
  log "Collecting manifest features → $out_file"
  run python3 app/k8s_collect.py watch \
    --kinds Deployment Job CronJob \
    --namespaces "$NS" \
    --emit-initial \
    --output "$out_file" &
  local pid=$!
  sleep 5          # give the watcher time to emit initial state
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

collect_labels() {
  local start_ts="$1"
  local end_ts="$2"
  local out_file="$3"
  log "Collecting Kepler labels [$start_ts → $end_ts] → $out_file"
  run python3 app/kepler_labels.py \
    --prom "$PROM_URL" \
    --start "$start_ts" \
    --end "$end_ts" \
    --mode job \
    --ns "$NS" \
    --out "$out_file"
}

cleanup_manifest() {
  local manifest="$1"
  log "Deleting $manifest..."
  run kubectl delete -f "$manifest" --ignore-not-found=true
  # wait for pods to terminate
  run kubectl wait pod \
    -n "$NS" \
    -l "app=energetiscope-bench" \
    --for=delete \
    --timeout=60s 2>/dev/null || true
}

# ── manifest list (ordered by CPU asc for reproducibility) ────────────────────
MANIFESTS=(
  "$MANIFEST_DIR/deploy-idle-100m-128mi.yaml"
  "$MANIFEST_DIR/deploy-cpu-250m-256mi.yaml"
  "$MANIFEST_DIR/deploy-cpu-500m-256mi.yaml"
  "$MANIFEST_DIR/deploy-mem-500m-1gi.yaml"
  "$MANIFEST_DIR/deploy-cpu-1000m-512mi.yaml"
  "$MANIFEST_DIR/deploy-mem-1000m-2gi.yaml"
  "$MANIFEST_DIR/deploy-cpu-2000m-512mi.yaml"
  "$MANIFEST_DIR/job-cpu-250m-256mi.yaml"
  "$MANIFEST_DIR/job-cpu-500m-256mi.yaml"
  "$MANIFEST_DIR/job-cpu-1000m-512mi.yaml"
  "$MANIFEST_DIR/job-cpu-2000m-1gi.yaml"
  "$MANIFEST_DIR/cronjob-cpu-500m-256mi.yaml"
  "$MANIFEST_DIR/cronjob-cpu-1000m-512mi.yaml"
)

# ── setup ─────────────────────────────────────────────────────────────────────
log "=== EnergetiScope Benchmark Collection ==="
log "Prometheus: $PROM_URL"
log "Repetitions per manifest: $REPS"
log "Warmup: ${WARMUP}s | Cooldown: ${COOLDOWN}s"
log "Output: $OUT_DIR"
log "Manifests: ${#MANIFESTS[@]}"
echo ""

run mkdir -p "$OUT_DIR/features" "$OUT_DIR/labels"

# Apply namespace first
run kubectl apply -f "$MANIFEST_DIR/00-namespace.yaml"

TOTAL_CONFIGS=$(( ${#MANIFESTS[@]} * REPS ))
CURRENT=0

# ── main loop ─────────────────────────────────────────────────────────────────
for MANIFEST in "${MANIFESTS[@]}"; do
  MANIFEST_NAME=$(basename "$MANIFEST" .yaml)

  for REP in $(seq 1 "$REPS"); do
    CURRENT=$(( CURRENT + 1 ))
    log "--- [$CURRENT/$TOTAL_CONFIGS] $MANIFEST_NAME  rep=$REP ---"

    # Generate unique ID for this run
    RUN_ID="${MANIFEST_NAME}_rep${REP}_$(ts_now)"
    FEAT_FILE="$OUT_DIR/features/${RUN_ID}.ndjson"
    LABEL_FILE="$OUT_DIR/labels/${RUN_ID}.parquet"

    # 1. Apply manifest
    log "Applying $MANIFEST..."
    run kubectl apply -f "$MANIFEST"

    # 2. Wait for pod ready (different logic for Jobs vs Deployments)
    if [[ "$MANIFEST_NAME" == job-* ]]; then
      wait_for_job_running "$(grep 'name: bench-job' "$MANIFEST" | head -1 | awk '{print $2}')"
    elif [[ "$MANIFEST_NAME" == cronjob-* ]]; then
      log "CronJob applied — waiting 90s for first job to trigger..."
      sleep 90
    else
      wait_for_pods_ready "app=energetiscope-bench" 120
    fi

    # 3. Collect manifest features (snapshot of current K8s state)
    collect_features "$FEAT_FILE"

    # 4. Record start timestamp and wait for steady state
    START_TS=$(ts_now)
    log "Steady-state warmup: ${WARMUP}s (started at $START_TS)..."
    run sleep "$WARMUP"
    END_TS=$(ts_now)

    # 5. Collect Kepler labels for the warmup window
    collect_labels "$START_TS" "$END_TS" "$LABEL_FILE"

    # 6. Delete manifest + cooldown
    cleanup_manifest "$MANIFEST"
    log "Cooldown: ${COOLDOWN}s..."
    run sleep "$COOLDOWN"

    log "Done: features=$FEAT_FILE  labels=$LABEL_FILE"
    echo ""
  done
done

# ── combine all collected data ────────────────────────────────────────────────
log "=== Collection complete. Combining data... ==="

COMBINED_NDJSON="$OUT_DIR/combined_features.ndjson"
run cat "$OUT_DIR/features/"*.ndjson > "$COMBINED_NDJSON"
log "Combined features: $COMBINED_NDJSON ($(wc -l < "$COMBINED_NDJSON") lines)"

log ""
log "=== Next steps ==="
log "1. Fit encoder:"
log "   python3 app/k8s_encode.py fit --input $COMBINED_NDJSON --out data/encoder_bench.joblib --no-sbert"
log ""
log "2. Transform features:"
log "   python3 app/k8s_encode.py transform --input $COMBINED_NDJSON --encoder data/encoder_bench.joblib --out $OUT_DIR/features.parquet"
log ""
log "3. Combine labels:"
log "   python3 eval/combine_labels.py --input-dir $OUT_DIR/labels --out $OUT_DIR/labels.parquet"
log ""
log "4. Join + train:"
log "   python3 app/join_features_labels.py --features $OUT_DIR/features.parquet --labels $OUT_DIR/labels.parquet --out $OUT_DIR/train_rows.parquet"
log "   python3 app/train_power.py --train $OUT_DIR/train_rows.parquet --target energy_step_j --out data/artifacts/knn_bench.joblib"
log ""
log "5. Evaluate:"
log "   python3 eval/eval_existing_model.py --features $OUT_DIR/features.parquet --labels $OUT_DIR/labels.parquet"
