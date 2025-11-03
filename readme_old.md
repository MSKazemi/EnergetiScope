kubectl -n energy delete job build-labels --ignore-not-found
kubectl -n energy apply -f k8s/jobs/03-job2-label.yaml
kubectl -n energy logs -f job/build-labels -f




✦ Based on the contents of the files:

  deploy-podpower-collector.yaml: This file defines a Kubernetes Deployment that runs a data collector. Its role is to watch for changes to Deployment, Job, and CronJob
  resources within the energy namespace and send that information to the podpower-predict service.

  deploy-podpower-predict.yaml: This file defines three resources:
   1. A Deployment that runs a prediction service using uvicorn. It loads a machine learning model and an encoder to make predictions.
   2. A Service of type ClusterIP that exposes the prediction service within the Kubernetes cluster.
   3. An Ingress resource that exposes the prediction service outside of the cluster at the hostname podpower.local.

  In summary, the collector gathers data from the cluster and the predict service uses that data to make prediction



### Sanity Checks

Do you actually have Kepler metrics in Prometheus?

```
kepler_container_power_watt
kepler_container_joules_total
```
### Quick sanity checks (only if something fails)

* In the Prom UI, verify these return data:

  * `kube_pod_owner` (or at least `kube_pod_labels`, `kube_pod_info`)
  * `kepler_container_power_watt`
  * `kepler_container_joules_total`
* If `kube_pod_owner` is empty, the scri




## ML Pipeline Full Loop

* Collect → inference_requests.ndjson (done)
* Encode → encoder.joblib, features.parquet (done/next)
* Label from Kepler → kepler_labels.parquet
* Join → train_rows.parquet
* Train → knn_power.joblib (+ CV metrics printed) (knn_energy.joblib)
* Predict → JSON with pred_avg_power_w (and later energy if you choose that target)

#### Next Steps

When you’re happy with offline accuracy, we can wrap steps (2)+(5)+(6) into a tiny FastAPI service and (optionally) a mutating admission webhook that writes recommended requests/limits or annotations with predicted power.

## Freeze your label spec (what you predict)

Pick one (you can do both later):

- A. Job-completion labels (best for batch Jobs)
  - Target = avg_power_w over the pod’s life and/or total_energy_j.
  - Join key = (namespace, workload_kind, workload_name, _spec_hash, pod_uid).

- B. Rolling window labels (best for long-running Deployments)
  - Target = windowed average power (e.g., 60s), optionally per pod then aggregated to workload.
  - Join key = (namespace, workload_kind, workload_name, _spec_hash, time_bucket).

You already have _spec_hash in the encoded features. Keep it—it’s perfect for stable joins per template.


## Collector & Feature Extraction

Run against a file (dry run):
```bash
python3 k8s_collect.py from-file ./my-deployment.yaml
# → prints one JSON line (InferenceRequest)

# Watch the cluster (advisory mode):
python3 k8s_collect.py watch --kinds Deployment Job CronJob Pod
python3 k8s_collect.py watch --kinds Deployment Job CronJob --namespaces default team-a
python3 k8s_collect.py watch --kinds Deployment Job CronJob --namespaces kubeintellect

# → streams JSON lines you can pipe to your predictor or a message bus
All namespaces & initial dump (so you immediately see data):

python k8s_collect.py watch --kinds Deployment Job CronJob --emit-initial
```

### Only kubeintellect, suppress TLS warning (dev):

```bash
python3 k8s_collect.py watch watch \
--kinds Deployment ReplicaSet Job CronJob Pod \
--emit-initial \
--suppress-tls-warnings > ./data/inference_requests.ndjson
```


```bash
python3 k8s_collect.py watch \
--kinds Deployment Job CronJob Pod \
--namespaces kubeintellect \
--emit-initial \
--suppress-tls-warnings > ./data/inference_requests.ndjson
```

## 2) Export labels from Kepler (Prometheus) → Parquet

```bash
   python3 app/kepler_labels.py \
     --prom http://prometheus.lpt.local \
     --owner-source auto \
     --mode window \
     --start $(date -u -d '6 hours ago' +%s) \
     --end   $(date -u +%s) \
     --out ./data/labels.parquet
```

With `--owner-source auto`, it will:
* Try `kube_pod_owner` first.
* If missing, query the live Kubernetes API to map each pod to its owner Deployment/Job/CronJob.

****### Window vs Job labels

- **window mode**: emits one row per timestamp per pod. Columns include `ts`, `avg_power_w`, and per-interval `energy_step_j`. There is no `total_energy_j`.
- **job mode**: aggregates each pod's lifetime into a single row. It computes mean `avg_power_w` and sums `energy_step_j` into `total_energy_j`.

Pick the mode based on your training target:

- **If training with `--target total_energy_j`**, generate labels with `--mode job` so the column exists.
- **If training with `--target energy_step_j` or `avg_power_w`**, `--mode window` is fine; or aggregate windows to create totals before training if you want `total_energy_j`.

## 2) Join features ↔ labels
Proceed to the `join_features_labels.py` step for training.

```bash
python join_features_labels.py \
  --features ./data/features.parquet \
  --labels   ./data/labels.parquet \
  --out      ./data/train_rows.parquet
```

## 3) Join features ↔ labels

```bash
python join_features_labels.py \
  --features ./data/features.parquet \
  --labels   ./data/labels.parquet \
  --out      ./data/train_rows.parquet
```

## 4) Train a baseline regressor (KNN)

```bash
python3 train_power.py \
  --train ./data/train_rows.parquet \
  --target total_energy_j \
  --out ./artifacts/knn_energy.joblib
```




## 4) Predict for current workloads

Use your latest NDJSON (or collect again), then:

```bash
python predict_k8s.py \
  --encoder ./artifacts/encoder.joblib \
  --model   ./artifacts/knn_energy.joblib \
  --input   ./data/inference_requests.ndjson
```


## 5) Predict (offline or service)

```bash
python predict_k8s.py \
  --encoder ./artifacts/encoder.joblib \
  --model ./artifacts/knn_energy.joblib \
  --input ./data/inference_requests.ndjson
```



This prints JSON like:

```json
[
  {"namespace":"kubeintellect","workload_kind":"Deployment","workload_name":"postgres","_spec_hash":"...","pred_avg_power_w": 12.7},
  ...
]
```

## 5) (Optional) Save predictions back to K8s as annotations

If you want a quick advisory loop, you can write a tiny patcher that adds:
`power.k8spcp.io/pred-avg-watt: "<value>"` to the `Deployment.spec.template.metadata.annotations`. I can give you that snippet if you want it.

---

### Quick Troubleshooting

* If `train_rows.parquet` has **0 rows**, the join keys didn’t match. Use a broader join (drop `_spec_hash`) or ensure your features and labels cover the same workloads/time.
* If `predict_k8s.py` errors on encoder/model paths, double-check that `./artifacts/encoder.joblib` and `./artifacts/knn_energy.joblib` exist.
* If SBERT model download is blocked, keep using `--no-sbert` for now; you can re-fit later with SBERT turned on.

Want me to add the optional **annotation patcher** (writes predictions back to the Deployment/CronJob/Job template) next?



---
### Example:
 
