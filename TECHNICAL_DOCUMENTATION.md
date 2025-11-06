# k8sPodPCP: Technical and Scientific Documentation

## System Overview

k8sPodPCP (Kubernetes Pod Power and Energy Prediction) is a machine learning-based system for predicting energy consumption of Kubernetes workloads from their declarative specifications. The system employs a supervised learning approach, utilizing workload specifications as features and ground-truth energy measurements from Kepler as labels to train a k-nearest neighbors (KNN) regression model.

## Architecture and Workflow

The system operates through a five-stage pipeline: (1) workload specification collection, (2) feature encoding, (3) ground-truth label extraction, (4) dataset construction, and (5) model training. Once trained, the model is served via a RESTful API for real-time energy predictions.

### Stage 1: Workload Specification Collection

The collection module (`k8s_collect.py`) extracts Kubernetes workload specifications from the cluster API. It supports multiple workload types including Deployments, Jobs, CronJobs, and Pods. The module watches the Kubernetes API for ADDED and MODIFIED events, transforming each workload's pod template into a structured `InferenceRequest` JSON schema.

The `InferenceRequest` schema captures:
- **Workload metadata**: namespace, workload kind (Deployment/Job/CronJob/Pod), workload name
- **Container specifications**: image names, commands, arguments, resource requests and limits (CPU in millicores, memory in MiB)
- **Infrastructure context**: node type, runtime class, GPU count
- **Orchestration metadata**: labels, annotations, init container count, sidecar count
- **Volume types**: enumeration of volume types (emptyDir, hostPath, PVC, etc.)

The collector normalizes workload hierarchies by mapping ReplicaSets and Pods to their parent Deployments, ensuring consistent feature representation across workload lifecycle stages.

### Stage 2: Feature Encoding

The encoding module (`k8s_encode.py`) transforms structured `InferenceRequest` objects into numerical feature vectors suitable for machine learning. The encoding process employs a multi-modal approach:

**Numeric Features**: Resource specifications are aggregated across containers by summation:
- CPU requests and limits (millicores)
- Memory requests and limits (MiB)
- GPU count, init container count, sidecar count

These numeric features are standardized using `StandardScaler` to normalize distributions.

**Categorical Features**: Infrastructure attributes are one-hot encoded:
- Runtime class (e.g., kata, gVisor, default)
- Node instance type (e.g., m5.large, c5.xlarge)

**Text Embeddings** (optional): Semantic text features are extracted from:
- Workload names and kinds
- Container image names
- Command and argument strings
- Labels and annotations

When enabled, the system uses Sentence-BERT (specifically `all-MiniLM-L6-v2`) to generate 384-dimensional embeddings. These embeddings capture semantic relationships between workload specifications, enabling the model to identify similar workloads based on functional similarity rather than exact string matches.

The final feature vector is a concatenation of standardized numeric features, one-hot encoded categorical features, and optional SBERT embeddings. A specification hash is computed for caching and deduplication purposes.

### Stage 3: Ground-Truth Label Extraction

The label extraction module (`kepler_labels.py`) queries Prometheus for energy measurements collected by Kepler. Kepler is a Kubernetes eBPF-based tool that estimates power consumption at the container level using hardware performance counters and system-level metrics.

The module performs the following operations:

1. **Energy Query**: Retrieves cumulative energy measurements (`kepler_container_joules_total`) for each container namespace and pod name over a specified time window.

2. **Step Energy Calculation**: Computes per-step energy consumption by calculating the first-order difference of cumulative energy values:
   ```
   energy_step_j = diff(kepler_container_joules_total).clip(lower=0.0)
   ```
   This yields energy consumed during each Prometheus query step interval (default: 60 seconds).

3. **Power Aggregation** (optional): Computes average power in watts from `kepler_container_power_watt` metrics.

4. **Workload Mapping**: Maps pods to their owner workloads (Deployments, Jobs, etc.) using either Prometheus `kube_pod_owner` metrics or direct Kubernetes API queries.

5. **Aggregation Modes**:
   - **Window mode**: Preserves time-series data with per-step energy values
   - **Job mode**: Aggregates energy over the entire job lifetime, computing total energy consumption

The resulting labels include `energy_step_j` (energy per time step in Joules), `avg_power_w` (average power in Watts), and `total_energy_j` (total energy for job-mode aggregations).

### Stage 4: Dataset Construction

The join module (`join_features_labels.py`) performs an inner join between encoded features and ground-truth labels. The join is performed on:
- Namespace
- Canonical workload name (normalized to handle ReplicaSet-to-Deployment mappings)

This produces a training dataset where each row contains:
- Feature vector (as a NumPy array)
- Target variable (`energy_step_j`, `avg_power_w`, or `total_energy_j`)
- Metadata (namespace, workload kind, workload name, specification hash)

### Stage 5: Model Training

The training module (`train_power.py`) implements a KNN regressor with cosine similarity metric. The training process employs group-aware cross-validation to prevent data leakage:

1. **Grouping**: Workloads are grouped by `workload_name` to ensure that all samples from the same workload appear exclusively in either the training or validation set.

2. **Cross-Validation**: `GroupKFold` is used with a maximum of 5 folds, limited by the number of unique workloads. This ensures that the model's performance reflects generalization to unseen workloads rather than memorization of workload-specific patterns.

3. **Model Selection**: The number of neighbors (default: 5) is configurable. Cosine similarity is used as the distance metric, which is particularly effective for high-dimensional feature vectors with optional SBERT embeddings.

4. **Evaluation Metrics**: The model is evaluated using:
   - Mean Absolute Error (MAE): Average absolute difference between predicted and actual energy
   - R² Score: Coefficient of determination measuring explained variance

5. **Final Model**: After cross-validation, a final model is trained on the entire dataset and serialized using joblib.

## Prediction Service

The prediction service (`predict_service.py`) is a FastAPI application that serves trained models via REST endpoints. The service:

1. **Initialization**: Loads the encoder and model artifacts at startup from configurable paths (default: `/artifacts/encoder.joblib` and `/artifacts/knn_energy.joblib`).

2. **Prediction Endpoint** (`POST /predict`): Accepts an `InferenceRequest` JSON, transforms it through the encoder, and returns:
   - `pred_energy_step_j`: Predicted energy consumption for one time step (default: 60 seconds) in Joules
   - Metadata: workload kind, workload name, namespace, specification hash

3. **YAML Endpoints**: Provides convenience endpoints (`/predict/from-yaml`, `/infer/from-yaml`) that accept raw Kubernetes YAML manifests and perform the necessary parsing and transformation.

## Technical Specifications

### Energy Prediction Period

The `pred_energy_step_j` value represents predicted energy consumption for a single time step. The step duration is determined by the Prometheus query step parameter used during label collection (default: 60 seconds). This can be configured via the `--step` argument in `kepler_labels.py` (e.g., `--step 30s`, `--step 5m`).

### Feature Vector Dimensions

The feature vector dimensionality depends on configuration:
- **Numeric features**: 9 dimensions (CPU/memory requests/limits, GPU count, init/sidecar counts)
- **Categorical features**: Variable dimensions based on unique values of runtime class and node type
- **SBERT embeddings**: 384 dimensions (when enabled) or 0 (when disabled)

Total dimensionality typically ranges from 20-50 dimensions without SBERT, and 400-450 dimensions with SBERT enabled.

### Model Characteristics

- **Algorithm**: K-Nearest Neighbors Regression
- **Distance Metric**: Cosine similarity
- **Neighbors**: Configurable (default: 5)
- **Training Strategy**: Group-aware cross-validation by workload name
- **Serialization**: joblib format for compatibility with scikit-learn

### Data Flow

```
Kubernetes API → k8s_collect.py → InferenceRequest (NDJSON)
                                           ↓
                                    k8s_encode.py
                                           ↓
                                    Feature Vectors (Parquet)
                                           ↓
Prometheus/Kepler → kepler_labels.py → Energy Labels (Parquet)
                                           ↓
                                    join_features_labels.py
                                           ↓
                                    Training Dataset (Parquet)
                                           ↓
                                    train_power.py
                                           ↓
                                    Trained Model (joblib)
                                           ↓
                                    predict_service.py (FastAPI)
                                           ↓
                                    Energy Predictions (JSON)
```

## Integration with Kubernetes Ecosystem

The system integrates with several Kubernetes ecosystem components:

- **Kepler**: Provides ground-truth energy measurements via eBPF-based power estimation
- **Prometheus**: Serves as the metrics aggregation and query layer
- **Kubernetes API**: Source of workload specifications and pod-to-workload mappings
- **Persistent Volumes**: Stores encoder and model artifacts for service deployment

## Limitations and Considerations

1. **Temporal Resolution**: Predictions are for discrete time steps. For continuous energy estimation over arbitrary periods, multiply `pred_energy_step_j` by the number of steps.

2. **Workload Similarity**: The KNN approach relies on similarity between workloads. Novel workload types with no similar training examples may yield less accurate predictions.

3. **Cluster-Specific**: Models trained on one cluster may not generalize to clusters with different hardware configurations, node types, or runtime environments.

4. **Ground-Truth Dependency**: The system requires Kepler to be installed and properly configured. Without ground-truth labels, model training cannot proceed.

5. **Feature Completeness**: Predictions are based solely on declarative specifications. Runtime behavior, actual resource utilization, and external factors (network I/O, storage I/O) are not directly captured in the feature set.

## Use Cases

The system is designed for:
- **Scheduler Integration**: Pre-scheduling energy estimation for workload placement decisions
- **Cost Optimization**: Predicting energy costs before workload deployment
- **Capacity Planning**: Estimating cluster energy requirements for workload sets
- **Research**: Studying relationships between workload specifications and energy consumption

## Future Enhancements

Potential improvements include:
- Integration of runtime metrics (actual CPU/memory utilization) as features
- Model registry and automated retraining pipelines
- Support for additional prediction targets (peak power, energy efficiency metrics)
- Ensemble methods combining multiple model types
- Transfer learning approaches for cross-cluster generalization

