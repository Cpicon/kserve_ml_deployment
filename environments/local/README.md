# Local KServe Playground

This folder contains helper scripts for spinning up a **Kind + Istio + Knative + KServe** stack on your laptop and a minimal example (the classic *sklearn-iris* model) you can use to verify everything is working.

---
## 1. Prerequisites

Make sure the following CLIs are installed and on your `$PATH`:

* `docker` (Kind uses Docker under the hood)
* `kind`  – <https://kind.sigs.k8s.io>
* `kubectl` – <https://kubernetes.io/docs/tasks/tools/>
* `helm` – <https://helm.sh>
* `jq` – <https://stedolan.github.io/jq/>

---
## 2. Create the cluster

```bash
# from the project root
bash environments/local/setup_kind.sh
```

The script creates a cluster called **kserve-deployment** with ports 80/443 exposed on your host so that Istio can later bind to them.

---
## 3. Install Istio, Knative & KServe

```bash
bash environments/local/install_kserve_knative.sh
```

The script:
1. Installs Istio 1.23.2
2. Installs Knative Serving 1.18
3. Installs cert-manager 1.16
4. Installs KServe v0.15.2
5. Detects that you are running on Kind and patches the Istio gateway via `setup_ingress_routing.sh` so that `http://localhost` & `https://localhost`  reach the gateway.

> Need a clean slate? Re-run with `--clean` to wipe previous installs:
> ```bash
> bash environments/local/install_kserve_knative.sh --clean
> ```

---
## 4. Deploy the demo model

```bash
kubectl create namespace kserve-test   # once
kubectl apply -n kserve-test -f environments/local/test/sklearn-iris.yaml
```

Wait until the InferenceService is ready:

```bash
kubectl get inferenceservice sklearn-iris -n kserve-test -w
```

You should eventually see `READY: True`.

---
## 5. Smoke-test with the helper script

```bash
bash environments/local/test/test_inference.sh \
     sklearn-iris kserve-test environments/local/test/iris-input.json
```

The script sends a prediction request and prints the result (should be class *Iris-setosa*, *Iris-versicolor* or *Iris-virginica*).

---
## 6. Quick tunnel to the Swagger UI (no DNS tweaks)

Knative's revision Service has no selector, so we cannot port-forward it directly. The easiest workaround is to forward **the Pod itself**:

```bash
# 1. Grab the name of the running revision pod
POD=$(kubectl -n kserve-test get pod \
      -l serving.knative.dev/revision=sklearn-iris-predictor-00001 \
      -o jsonpath='{.items[0].metadata.name}')

# 2. Forward local port 8080 to the model container's port 8080
kubectl -n kserve-test port-forward pod/$POD 8080:8080
```

Open your browser at <http://localhost:8080/docs> – the interactive Swagger UI is now available without touching `/etc/hosts`, DNS or the Istio gateway.

> If the pod has been scaled to zero you may need to hit the InferenceService once (e.g. rerun the test script) so that Knative spins the pod up before you port-forward.

---
## 7. Cleanup

```bash
kind delete cluster --name kserve-deployment
```

---
## 8. Production Considerations

Deploying a KServe stack in a production environment requires careful planning and configuration beyond a local setup. Below are critical considerations to address before going live.

### 1. Kubernetes Cluster

- **Managed Kubernetes:** Use a managed Kubernetes service like Amazon EKS, Google GKE, or Azure AKS. These services handle the control plane's reliability, scaling, and security, which is complex to manage yourself.
- **Node Pools:** Use dedicated node pools for different components. For example, have a separate node pool for the ingress gateways, another for the KServe/Knative control plane, and dedicated node pools (potentially with GPUs) for the models themselves. This isolates workloads and allows for more granular scaling and resource management.

### 2. Ingress and DNS

- **Public DNS:** The `istio-ingressgateway` service will be exposed via a cloud provider's LoadBalancer. You must configure your public DNS to point a wildcard domain (e.g., `*.models.yourcompany.com`) to the public IP address of this LoadBalancer.
- **TLS Certificates:** Use `cert-manager` with a `ClusterIssuer` configured for a production-ready Certificate Authority like Let's Encrypt. This will enable automatic provisioning and renewal of TLS certificates for all your `InferenceService` endpoints, ensuring secure communication.

### 3. High Availability (HA)

- **Control Plane:** For all installed components (Istio, Knative, KServe), run multiple replicas of their control plane pods (e.g., `istiod`, `kserve-controller-manager`). This prevents a single pod failure from causing an outage.
- **Ingress Gateways:** Run multiple replicas of the `istio-ingressgateway` to ensure your entrypoint is fault-tolerant.

### 4. Security

- **Istio Security:** The default Istio installation is permissive. Implement Istio `AuthorizationPolicies` to enforce strict, least-privilege access control between your services and from external traffic.
- **Network Policies:** Use Kubernetes `NetworkPolicies` to restrict traffic between pods at the network level, providing an additional layer of defense.
- **Secrets Management:** Use a dedicated secrets management solution like HashiCorp Vault or a cloud provider's secret manager (e.g., AWS Secrets Manager) to handle sensitive information like model storage credentials. Avoid storing plaintext secrets in Git.

### 5. Model Storage

- **Persistent & Shared Storage:** You MUST use a production-grade, shared, and persistent storage solution for your models, such as AWS S3, Google Cloud Storage, or a self-hosted MinIO cluster. This ensures that models are not lost and can be accessed from any node where a model pod might be scheduled.
- **Credentials:** Create a Kubernetes secret containing the credentials for your model storage and reference this secret in your `InferenceService` definitions.

### 6. Resource Management

- **Requests and Limits:** Profile your models to understand their CPU, memory, and GPU consumption under load. Set appropriate resource `requests` and `limits` for your `InferenceService` pods to ensure stable performance and prevent resource contention on your nodes.
- **Autoscaling:** Fine-tune Knative's autoscaling parameters (e.g., `target` concurrency, `scale-down-delay`) to match your traffic patterns and cost-efficiency goals.

By addressing these areas, you can build a robust, scalable, and secure production environment for your machine learning models.

Enjoy experimenting! Feel free to adjust the scripts or drop additional model manifests inside the `test/` folder. 