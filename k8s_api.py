from kubernetes import client, config
from functools import lru_cache
from k8s_collect import list_and_emit_initial

# ------------------- Kubernetes Config Loader -------------------
def _load_k8s_config():
    try:
        logger.info("Trying to load local kubeconfig...")
        config.load_kube_config()
        logger.info("Loaded local kubeconfig.")
    except ConfigException:
        logger.warning("Local kubeconfig not found. Trying in-cluster config...")
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config.")
        except ConfigException as e:
            logger.error("Failed to load any Kubernetes config.")
            raise KubernetesConfigurationError("Kubernetes configuration could not be loaded.") from e

# ------------------- Kubernetes API Clients -------------------
@lru_cache()
def get_core_v1_api() -> client.CoreV1Api:
    _load_k8s_config()
    logger.debug("Returning CoreV1Api client.")
    return client.CoreV1Api()

@lru_cache()
def get_apps_v1_api() -> client.AppsV1Api:
    _load_k8s_config()
    logger.debug("Returning AppsV1Api client.")
    return client.AppsV1Api()

@lru_cache()
def get_networking_v1_api() -> client.NetworkingV1Api:
    _load_k8s_config()
    logger.debug("Returning NetworkingV1Api client.")
    return client.NetworkingV1Api()

def list_ingress_in_namespace(namespace: str):
    # Imports are INSIDE the function for robustness in the REPL environment
    import json
    from kubernetes import client, config
    try:
        # Attempt to load Kubernetes config (local or in-cluster)
        try:
            config.load_kube_config()
        except config.ConfigException:
            try:
                config.load_incluster_config()
            except config.ConfigException as e:
                return {"status": "error", "message": f"Could not load Kubernetes config: {str(e)}", "error_type": "K8sConfigError"}

        networking_v1 = client.NetworkingV1Api()
        ingress_list = networking_v1.list_namespaced_ingress(namespace=namespace, timeout_seconds=10)
        ingresses = [
            {
                "name": ingress.metadata.name,
                "namespace": ingress.metadata.namespace,
                "hosts": [rule.host for rule in ingress.spec.rules] if ingress.spec.rules else [],
                "creation_timestamp": ingress.metadata.creation_timestamp.isoformat() if ingress.metadata.creation_timestamp else None
            }
            for ingress in ingress_list.items
        ]
        return {"status": "success", "data": ingresses} # Return a dictionary
    except client.exceptions.ApiException as e:
        error_message = f"K8s API error: {e.reason} (status: {e.status})"
        # Attempt to get more details from the error body
        if e.body:
            try:
                error_details = json.loads(e.body)
                error_message += f" Details: {error_details.get('message', e.body)}"
            except json.JSONDecodeError:
                pass # If body is not JSON, use the original ApiException message
        return {"status": "error", "message": error_message, "error_type": "ApiException"}
    except ImportError as ie:
        return {"status": "error", "message": f"Import error inside function: {str(ie)}", "error_type": "ImportError"}
    except Exception as e:
        return {"status": "error", "message": str(e), "error_type": type(e).__name__}
# Pydantic Input Schema:


def list_pods_in_namespace(namespace: str):
    # Imports are INSIDE the function for robustness in the REPL environment
    import json
    from kubernetes import client, config

    try:
        # Load Kubernetes configuration
        try:
            config.load_kube_config()
        except config.ConfigException:
            try:
                config.load_incluster_config()
            except config.ConfigException as e:
                return {
                    "status": "error",
                    "message": f"Could not load Kubernetes config: {str(e)}",
                    "error_type": "K8sConfigError"
                }

        core_v1 = client.CoreV1Api()
        pod_list = core_v1.list_namespaced_pod(namespace=namespace, timeout_seconds=10)

        pods = [
            {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "node_name": pod.spec.node_name,
                "phase": pod.status.phase,
                "pod_ip": pod.status.pod_ip,
                "host_ip": pod.status.host_ip,
                "creation_timestamp": pod.metadata.creation_timestamp.isoformat()
                    if pod.metadata.creation_timestamp else None,
                "containers": [c.name for c in pod.spec.containers]
            }
            for pod in pod_list.items
        ]

        return {"status": "success", "data": pods}

    except client.exceptions.ApiException as e:
        error_message = f"K8s API error: {e.reason} (status: {e.status})"
        if e.body:
            try:
                error_details = json.loads(e.body)
                error_message += f" Details: {error_details.get('message', e.body)}"
            except json.JSONDecodeError:
                pass
        return {"status": "error", "message": error_message, "error_type": "ApiException"}

    except ImportError as ie:
        return {
            "status": "error",
            "message": f"Import error inside function: {str(ie)}",
            "error_type": "ImportError"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }




print(list_ingress_in_namespace(namespace='kube-system'))
print(list_pods_in_namespace(namespace='kube-system'))