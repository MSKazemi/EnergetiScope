# Installing Kepler for Power Monitoring

This guide shows how to install the [Kepler](https://kepler.sh/) power monitoring tool using its Helm chart. These instructions assume you have Helm installed and configured for your Kubernetes cluster.

Kepler is used to collect power consumption metrics, which are then scraped by Prometheus to build the training labels for the power prediction model.

## Installation Steps

1.  **Add the Kepler Helm repository:**

    ```bash
    helm repo add kepler https://sustainable-computing-io.github.io/kepler-helm-chart
    helm repo update
    ```

2.  **Create the namespace and install Kepler:**

    The following command creates the `energy` namespace (if it doesn't exist) and installs Kepler into it.

    We enable the `ServiceMonitor` so that Prometheus can automatically discover and scrape Kepler's metrics. The `release=prometheus` label is added to the `ServiceMonitor` to match the selector used by the Prometheus Operator in this project.

    ```bash
    kubectl create ns energy || true

    helm upgrade --install kepler kepler/kepler \
      --namespace energy \
      --set serviceMonitor.enabled=true \
      --set serviceMonitor.labels.release=prometheus
    ```

## Verification

After installation, you can check that the Kepler pods are running:

```bash
kubectl get pods -n energy
```

You should also be able to see Kepler's metrics in your Prometheus dashboard.
