# EnergetiScope

**Predict how much energy a Kubernetes workload will use — from its YAML, before you deploy it.**

EnergetiScope learns the relationship between a workload's *specification* (image, resource
requests, kind, labels) and its *measured* energy draw, using [Kepler](https://github.com/sustainable-computing-io/kepler)'s
eBPF power metrics as ground truth. Give it a manifest, get joules back — without running the
workload first.

It is for platform and SRE teams doing energy-aware scheduling, capacity planning, or carbon
reporting, who already have Kepler and want a *predictive* signal rather than only a
retrospective one.

```console
$ curl -sS -X POST localhost:8000/predict/from-yaml \
    -H 'Content-Type: text/plain' --data-binary @deployment.yaml
{"pred_energy_step_j":3423.8538,"workload_kind":"Deployment","workload_name":"nginx",
 "namespace":"default","spec_hash":"5f834bc13195e9ed"}
```

On our benchmark, gradient-boosted trees reach **R² = 0.917, MAPE 11.5%** under
workload-grouped cross-validation (n = 1,422 measurements over 10 workload groups).
Reproduce that number yourself in one command — it is the quickstart below.

---

## Install

```bash
git clone https://github.com/MSKazemi/EnergetiScope.git
cd EnergetiScope
pip install -r requirements.txt      # or: make install  (uses uv)
```

Requires **Python ≥ 3.11**. No cluster needed to try it — the benchmark dataset ships with
the repo.

## Quickstart — reproduce the benchmark (about a minute)

```bash
python eval/eval_bench_grouped.py
```

```
rows=1422  workload groups=10  target=energy_step_j  mean=10624.0 J  sd=4321.3 J
feature dim=9  non-constant columns=4

model                             MAE (J)     RMSE      R2    MAPE |   wl-MAE   wl-R2  wl-MAPE
----------------------------------------------------------------------------------------------
KNN (k=5, cosine)            2027 +/- 1285     2648   0.625   23.3% |     1612   0.773    20.9%
Linear Regression            1449 +/- 633      1704   0.844   15.1% |      911   0.933    14.9%
Gradient Boosted Trees        982 +/- 347      1242   0.917   11.5% |      767   0.956     9.9%
Random Forest                1092 +/- 428      1377   0.898   13.1% |      889   0.938    11.7%

mean-predictor MAE = 3523 J (R2 = 0 by definition)
```

Folds are split by workload (`GroupKFold`) and metrics are computed on pooled out-of-fold
predictions, so repeated measurements of one workload never straddle a fold. The
mean-predictor row is printed as the honest floor to compare against.

## Serve predictions

```bash
PYTHONPATH=app \
ENCODER_PATH=data/bench/encoder.joblib \
MODEL_PATH=data/artifacts/knn_bench.joblib \
  uvicorn predict_service:app --port 8000
```

> `PYTHONPATH=app` is required — the service imports its encoder as a top-level module.

```bash
curl -sS -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "namespace": "default",
  "workload_kind": "Deployment",
  "workload_name": "nginx",
  "containers": [
    {"name": "nginx", "image": "nginx:1.25", "req_cpu_mcpu": 250, "req_mem_mib": 256}
  ]
}'
```

```json
{"pred_energy_step_j":3423.8538,"workload_kind":"Deployment","workload_name":"nginx",
 "namespace":"default","spec_hash":"5f834bc13195e9ed"}
```

Interactive docs at <http://localhost:8000/docs>.

## How it works

```
k8s_collect.py            watch the cluster   → InferenceRequest NDJSON
k8s_encode.py             fit / transform     → feature vectors + encoder.joblib
kepler_labels.py          query Prometheus    → measured energy per workload
join_features_labels.py   merge               → train_rows.parquet
train_power.py            GroupKFold CV       → model.joblib
predict_service.py        FastAPI             → joules from a manifest
```

Each workload becomes one vector combining optional **SBERT** embeddings of image names and
labels, **numeric** resource requests/limits, and **one-hot categorical** runtime/node class.
Regressors available via `--model {gbt,rf,linear,knn}`; `gbt` is the default and the most
accurate on our benchmark.

## API

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/predict` | POST | `InferenceRequest` JSON | prediction JSON |
| `/infer/from-yaml` | POST | Kubernetes YAML (`text/plain`) | `InferenceRequest[]` |
| `/predict/from-yaml` | POST | Kubernetes YAML (`text/plain`) | prediction JSON |

`InferenceRequest` requires `namespace`, `workload_kind`, `workload_name`, and `containers`
(each with `name` and `image`; optional `req_cpu_mcpu`, `req_mem_mib`, `lim_cpu_mcpu`,
`lim_mem_mib`). See [`app/models.py`](app/models.py).

| Variable | Default | Description |
|---|---|---|
| `ENCODER_PATH` | `/artifacts/encoder.joblib` | Encoder artifact |
| `MODEL_PATH` | `/artifacts/knn_energy.joblib` | Trained model |
| `KUBECONFIG` | autodetect | Out-of-cluster kubeconfig |
| `K8S_CA_FILE` | — | CA certificate for TLS |
| `VERIFY_SSL` | client default | Force SSL verification on/off |

## Limitations — read before trusting a number

This is **research code from an in-progress paper**. The boundaries matter more than the
headline:

- **Models are hardware-specific.** The bundled model was trained on a single bare-metal node
  using Intel RAPL. Energy is as much a property of the machine as of the workload — retrain
  on your own cluster rather than reusing ours.
- **Kepler needs RAPL.** On most cloud VMs (including AKS) the RAPL MSRs are not exposed, so
  Kepler falls back to an estimation model and may attribute **zero** energy to pods. Ground
  truth requires bare metal or a host exposing RAPL.
- **The benchmark is small and narrow.** 1,422 measurements, but only **10 workload groups**,
  and only **4 of 9 feature columns actually vary** — the categorical block is largely
  degenerate. R² = 0.917 is a real number on this data; it is not evidence of generality.
- **`join_features_labels.py` currently over-joins.** Its `train_rows*.parquet` outputs
  contain many-to-many duplicate rows (14,906 rows where the correct inner merge yields
  1,422) and models trained on them score *worse than the mean predictor*. Use
  `eval/eval_bench_grouped.py`, which merges correctly, until this is fixed
  ([issue](https://github.com/MSKazemi/EnergetiScope/issues)).
- **Specification-only.** Two workloads with identical specs but different real load look
  identical to the model. It predicts declared intent, not runtime behaviour.

## Repository layout

| Path | Contents |
|---|---|
| `app/` | Pipeline stages and the FastAPI service |
| `eval/` | Evaluation and reproduction scripts |
| `data/bench/` | Benchmark features, labels, and encoders |
| `k8s/` | Manifests: batch Jobs for the pipeline, Deployments for API + collector |
| `manifests/bench/` | Workload manifests used to generate the benchmark |

Cluster setup and deployment: [`cluster-setup.md`](cluster-setup.md),
[`k8s-collect-guide.md`](k8s-collect-guide.md),
[`COLLECTION_RUNBOOK.md`](COLLECTION_RUNBOOK.md),
[`TECHNICAL_DOCUMENTATION.md`](TECHNICAL_DOCUMENTATION.md).

> External-validation scripts in `eval/` expect the third-party dataset
> [Zenodo 14332659](https://doi.org/10.5281/zenodo.14332659) in `data/external/`. It is not
> redistributed here — download it separately.

## Contributing

Contributions are very welcome — especially **energy measurements from hardware other than
ours**, which is the single most valuable thing anyone can add.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[good first issues](https://github.com/MSKazemi/EnergetiScope/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Citing

See [`CITATION.cff`](CITATION.cff), or the "Cite this repository" button on GitHub.

## License

[Apache-2.0](LICENSE) — see [`NOTICE`](NOTICE).

## Acknowledgments

- [Kepler](https://github.com/sustainable-computing-io/kepler) — eBPF energy metrics for Kubernetes
- [Prometheus](https://prometheus.io) / [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts)
- [sentence-transformers](https://www.sbert.net/) — SBERT text embeddings
