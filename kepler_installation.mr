helm repo add kepler https://sustainable-computing-io.github.io/kepler-helm-chart
helm repo update


# enable (or re-enable) Kepler's ServiceMonitor
helm upgrade --install kepler kepler/kepler -n kepler \
  --set serviceMonitor.enabled=true


kubectl get servicemonitors -n kepler
# should show: kepler
