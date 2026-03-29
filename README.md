# EnergetiScope — Kubernetes Workload Energy Prediction

EnergetiScope predicts energy consumption for Kubernetes workloads from their specs before they run. It collects workload templates, encodes them into features, trains a KNN regressor, and serves predictions via a FastAPI service. Ground-truth labels come from Kepler and Prometheus.

## Features

- **Kubernetes watcher** — streams `Deployment`/`Job`/`CronJob`/`Pod` specs to structured `InferenceRequest` JSON
- **Feature encoder** — SBERT-optional text embeddings + numeric/categorical features → Parquet
- **Label builder** — exports `avg_power_w` and `energy_step_j`/`total_energy_j` from Kepler/Prometheus
- **Training** — KNN baseline with group-aware cross-validation; artifact saved as `.joblib`
- **Serving** — FastAPI service with `/predict`, `/predict/from-yaml`, and `/infer/from-yaml` endpoints
- **Kubernetes manifests** — batch Jobs for the full pipeline and Deployments for API + collector

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                 │
│                                                       │
│  Collector ──POST InferenceRequest──► FastAPI /predict│
│  (k8s_collect.py)                    uses encoder.joblib
│                                      uses knn_energy.joblib
│  Kepler ──metrics──► Prometheus                       │
└─────────────────────────────────────────────────────┘
         ▲ artifacts
Batch Jobs: collect → encode → label → join → train
         └──────────────────────────────► PVC (energetiscope-data)
```

## Quickstart

### Prerequisites

- Python ≥ 3.11
- Kubernetes cluster access (optional for local dev)
- Prometheus + Kepler for label generation (see `kepler-setup.md` and `prometheus-setup.md`)
- Docker and `kubectl`/`helm` for cluster deployment

### Local install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the API locally

```bash
uvicorn app/predict_service:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

## API Reference

All endpoints are on the FastAPI service (`app/predict_service.py`). Artifacts are loaded at startup from env vars.

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/predict` | POST | `InferenceRequest` JSON | `PredictOut` JSON |
| `/infer/from-yaml` | POST | Kubernetes YAML (`text/plain`) | Array of `InferenceRequest` JSON |
| `/predict/from-yaml` | POST | Kubernetes YAML (`text/plain`) | `PredictOut` JSON |

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `ENCODER_PATH` | `/artifacts/encoder.joblib` | Encoder artifact path |
| `MODEL_PATH` | `/artifacts/knn_energy.joblib` | Trained model path |

**Response shape:**

```json
{
  "pred_energy_step_j": 123.4,
  "workload_kind": "Deployment",
  "workload_name": "nginx",
  "namespace": "default",
  "spec_hash": "6c7c..."
}
```

### Examples

**From a live Deployment (inside cluster):**

```bash
kubectl get deploy nginx -n default -o yaml | \
  curl -sS -X POST \
    http://energetiscope-predict.default.svc.cluster.local:8000/predict/from-yaml \
    -H 'Content-Type: text/plain' --data-binary @- | jq .
```

**From an `InferenceRequest` JSON:**

```bash
curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "workload_kind": "Deployment",
    "workload_name": "nginx",
    "namespace": "default",
    "pod_spec": {"containers": [{"name": "nginx", "image": "nginx:1.25"}]}
  }' | jq .
```

**From Python:**

```python
import requests

yaml_text = open("example/example.yaml").read()
r = requests.post(
    "http://localhost:8000/predict/from-yaml",
    data=yaml_text,
    headers={"Content-Type": "text/plain"},
    timeout=30,
)
print(r.json())  # {'pred_energy_step_j': ..., 'workload_kind': 'Deployment', ...}
```

## Pipeline: collect → encode → label → join → train → predict

Run the full pipeline locally with the following steps:

```bash
# 1. Collect workload specs from your cluster
python app/k8s_collect.py watch \
  --kinds Deployment Job CronJob Pod \
  --emit-initial \
  --suppress-tls-warnings \
  --output data/in.ndjson

# 2. Fit encoder (--no-sbert speeds up initial runs)
python app/k8s_encode.py fit \
  --input data/in.ndjson \
  --out artifacts/encoder.joblib \
  --no-sbert

# 3. Transform features
python app/k8s_encode.py transform \
  --input data/in.ndjson \
  --encoder artifacts/encoder.joblib \
  --out data/features.parquet

# 4. Export labels from Kepler/Prometheus
python app/kepler_labels.py \
  --prom http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090 \
  --mode job \
  --start $(date -u -d '6 hours ago' +%s) \
  --end   $(date -u +%s) \
  --out data/labels.parquet

# 5. Join features and labels
python app/join_features_labels.py \
  --features data/features.parquet \
  --labels   data/labels.parquet \
  --out      data/train_rows.parquet

# 6. Train KNN model
python app/train_power.py \
  --train data/train_rows.parquet \
  --target total_energy_j \
  --out artifacts/knn_energy.joblib

# 7. Predict for current workloads
python app/predict_k8s.py \
  --encoder artifacts/encoder.joblib \
  --model   artifacts/knn_energy.joblib \
  --input   data/in.ndjson
```

## Docker

```bash
docker build -t energetiscope:latest .

docker run --rm -p 8000:8000 \
  -e ENCODER_PATH=/app/artifacts/encoder.joblib \
  -e MODEL_PATH=/app/artifacts/knn_energy.joblib \
  -v "$(pwd)/app/artifacts:/app/artifacts:ro" \
  energetiscope:latest \
  uvicorn app/predict_service:app --host 0.0.0.0 --port 8000
```

## Kubernetes Deployment

Install Prometheus and Kepler first (see `prometheus-setup.md` and `kepler-setup.md`), then apply the manifests in order:

```bash
# Namespace and PVC
kubectl apply -f k8s/jobs/00-ns-pvc.yaml

# RBAC for reading cluster objects
kubectl apply -f k8s/jobs/01-rbac.yaml

# Job 1: collect + encode
kubectl apply -f k8s/jobs/02-job1-collect.yaml

# Job 2: build labels from Kepler
kubectl apply -f k8s/jobs/03-job2-label.yaml

# Job 3: join features and labels
kubectl apply -f k8s/jobs/04-job3-dataset.yaml

# Job 4: train model
kubectl apply -f k8s/jobs/05-job4-train.yaml

# Deploy the predictor API
kubectl apply -f k8s/deploy-energetiscope-predict.yaml

# Deploy the collector
kubectl apply -f k8s/deploy-energetiscope-collector.yaml
```

> **Note:** Update the Prometheus endpoint in `03-job2-label.yaml` and the storage class in `00-ns-pvc.yaml` to match your cluster setup.

**In-cluster service DNS:** `http://energetiscope-predict.<namespace>.svc.cluster.local:8000`

### Example: CronJob to periodically pre-score a Deployment

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: energetiscope-prescore
  namespace: default
spec:
  schedule: "*/10 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: energetiscope-reader
          restartPolicy: OnFailure
          containers:
          - name: prescore
            image: bitnami/kubectl:1.29
            command: ["/bin/sh", "-c"]
            args:
            - |
              kubectl get deploy nginx-deployment -n default -o yaml | \
              curl -sS -X POST \
                http://energetiscope-predict.default.svc.cluster.local:8000/predict/from-yaml \
                -H 'Content-Type: text/plain' --data-binary @- | jq .
```

The `energetiscope-reader` ServiceAccount needs `get` on `deployments` in the target namespace.

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ENCODER_PATH` | `/artifacts/encoder.joblib` | Encoder artifact path |
| `MODEL_PATH` | `/artifacts/knn_energy.joblib` | Trained model path |
| `KUBECONFIG` | autodetect | Out-of-cluster kubeconfig path |
| `K8S_CA_FILE` | — | CA certificate file for TLS |
| `VERIFY_SSL` | client default | Force SSL verification on/off |

In Kubernetes, manifests mount a PVC at `/app/artifacts` and set the above env vars automatically.

## Prometheus / Kepler Sanity Check

Ensure Prometheus has data for these metrics before running label export:

- `kube_pod_owner`
- `kepler_container_power_watt`
- `kepler_container_joules_total`

## Roadmap

### Research

| # | Task | Status |
|---|------|--------|
| R1 | Run experiments and record real MAE / R² values | pending |
| R2 | Add validation datasets and holdout benchmarks | pending |
| R3 | Feature ablation study (numeric / categorical / SBERT) | pending |
| R4 | Model registry and automated retraining CronJob | pending |
| R5 | Runtime metrics integration (CPU%, memory% at steady state) | pending |
| R6 | Evaluate gradient-boosted trees or MLP alongside KNN | pending |
| R7 | Cross-cluster transfer learning evaluation | pending |
| R8 | Sidecar/service-mesh detection (`_count_sidecars()`) | pending |
| R9 | Optional mutating admission webhook | pending |

### Paper (`paper/`)

| # | Task | Status |
|---|------|--------|
| P1 | Fill in real CV MAE ± std and R² in §IV | pending |
| P2 | Create architecture figure (`paper/figs/architecture.pdf`) | pending |
| P3 | Fill in author / affiliation block | pending |
| P4 | Choose and format for target venue (IEEE CLOUD, ICT4S, IPDPS, EuroSys) | pending |
| P5 | Verify and expand related work citations | done |
| P6 | Add feature ablation results table in §IV | pending |
| P7 | Reviewer-mode validation pass | pending |
| P8 | SoA section expansion | done |
| P9 | Build final PDF (`cd paper && make`) | pending |

## Additional Files

| File | Description |
|---|---|
| `kepler-setup.md` | Helm install commands for Kepler |
| `prometheus-setup.md` | Helm install commands for kube-prometheus-stack |
| `example/example.yaml` | Sample Kubernetes manifest for local testing |
| `TECHNICAL_DOCUMENTATION.md` | In-depth technical reference |
| `ROADMAP.md` | Detailed project progress tracking |

## License

> TODO: Add a `LICENSE` file (recommended: Apache-2.0 or MIT).

## Cite

```bibtex
@misc{EnergetiScope2025,
  title  = {EnergetiScope: Machine Learning-Based Energy Prediction for Kubernetes Workloads},
  author = {TODO},
  year   = {2025},
  doi    = {TODO}
}
```

## Acknowledgments

- [Kepler](https://github.com/sustainable-computing-io/kepler) — eBPF-based energy metrics for Kubernetes
- [Prometheus](https://prometheus.io) / [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts)

## Contact

`<CONTACT_NAME>` — `<EMAIL>`
