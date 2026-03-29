# Cluster Setup: Prometheus + Kepler

This guide covers installing both **Prometheus** (metrics collection) and **Kepler** (power monitoring) on your Kubernetes cluster. Both are required for EnergetiScope's label generation pipeline.

> **Important:** Install Prometheus first, then Kepler. Kepler's ServiceMonitor must match the Prometheus Operator's selector label.

---

## 1. Installing Prometheus

The `kube-prometheus-stack` Helm chart provides Prometheus and Grafana.

### Add the Helm repository

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

### Install kube-prometheus-stack

The following installs the stack into the `monitoring` namespace with persistent storage. Alertmanager and Pushgateway are disabled as they are not needed for this project.

**Note:** Adjust `grafana.ingress.hosts` and `prometheus.ingress.hosts` to match your environment.

```bash
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --create-namespace --namespace monitoring \
  --set kube-state-metrics.enabled=true \
  --set alertmanager.enabled=false \
  --set prometheus-pushgateway.enabled=false \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.accessModes[0]=ReadWriteOnce \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=8Gi \
  --set prometheus.service.type=ClusterIP \
  --set prometheus.service.port=9090 \
  --set grafana.service.type=ClusterIP \
  --set grafana.ingress.enabled=true \
  --set grafana.ingress.ingressClassName=nginx \
  --set grafana.ingress.hosts[0]=grafana.lpt.local \
  --set prometheus.ingress.enabled=true \
  --set prometheus.ingress.ingressClassName=nginx \
  --set prometheus.ingress.hosts[0]=prometheus.lpt.local \
  --atomic --timeout 15m
```

### Verify Prometheus

```bash
kubectl get pods -n monitoring
```

Access the Prometheus and Grafana UIs at the hostnames you configured.

---

## 2. Installing Kepler

[Kepler](https://kepler.sh/) collects power consumption metrics via eBPF, which Prometheus scrapes to build the training labels for the energy prediction model.

### Add the Kepler Helm repository

```bash
helm repo add kepler https://sustainable-computing-io.github.io/kepler-helm-chart
helm repo update
```

### Determine the Prometheus ServiceMonitor selector label

The Kepler `ServiceMonitor` label must match the Prometheus Operator's selector. Check what your cluster uses:

```bash
# Check the ServiceMonitor selector
kubectl -n monitoring get prometheus -o jsonpath='{range .items[*]}{.metadata.name}{" => "}{.spec.serviceMonitorSelector.matchLabels}{"\n"}{end}'

# Check the Helm release label
kubectl -n monitoring get prometheus -o jsonpath='{range .items[*]}{.metadata.labels.helm\.sh/release}{"\n"}{end}'
```

### Install Kepler

If the label is `release: prometheus`:

```bash
kubectl create ns kepler || true

helm upgrade --install kepler kepler/kepler \
  --namespace kepler \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.labels.release=prometheus
```

If the label is `release: prometheus-stack`:

```bash
kubectl create ns kepler || true

helm upgrade --install kepler kepler/kepler \
  --namespace kepler \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.labels.release=prometheus-stack
```

### Verify Kepler

```bash
kubectl get pods -n kepler
```

Confirm Kepler's metrics are visible in Prometheus (search for `kepler_container_joules_total`).

---

## 3. Sanity Check

Before running the label export pipeline, verify Prometheus has data for these metrics:

- `kube_pod_owner`
- `kepler_container_power_watt`
- `kepler_container_joules_total`

```bash
# Port-forward Prometheus for local access
kubectl port-forward svc/prometheus-operated 9090:9090 -n monitoring &

# Quick check
curl -s 'http://localhost:9090/api/v1/query?query=kepler_container_joules_total' | python3 -m json.tool | head -20
```

---

## Known Issue: Azure VMs and RAPL

Azure VMs do not expose RAPL MSR registers or hardware performance counters. On AKS, Kepler falls back to its `Regressor/AbsPower` model, which may attribute 0 energy to pods started after Kepler. See `DISCUSSION_LOG.md` Section 10 for details.

**Recommendation:** Use bare-metal nodes or VMs with RAPL passthrough for accurate per-pod energy measurement.
