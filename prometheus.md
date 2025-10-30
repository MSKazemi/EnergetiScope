# Prometheus
```bash

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

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