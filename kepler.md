# Add Kepler chart repo (listed on Artifact Hub)
# Install Kepler with ServiceMonitor enabled and labeled for our "prometheus" release

```bash
helm repo add kepler https://sustainable-computing-io.github.io/kepler-helm-chart
helm repo update

kubectl create ns kepler || true
helm upgrade --install kepler kepler/kepler \
  -n kepler \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.labels.release=prometheus
```