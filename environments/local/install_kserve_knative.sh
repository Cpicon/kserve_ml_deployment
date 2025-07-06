#!/bin/bash

# Script to install KServe and its dependencies for a serverless deployment.
# Usage: ./install_kserve_knative.sh [--clean]

set -eo pipefail

# --- Configuration ---
export ISTIO_VERSION=1.23.2
export KNATIVE_OPERATOR_VERSION=v1.18.0
export KNATIVE_SERVING_VERSION=1.18.0
export KSERVE_VERSION=v0.15.2
export CERT_MANAGER_VERSION=v1.16.1

# --- Parse Arguments ---
CLEAN_INSTALL=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_INSTALL=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--clean]"
            exit 1
            ;;
    esac
done

# --- Version Check Functions ---
check_istio_version() {
    if helm list -n istio-system 2>/dev/null | grep -q "istio-base"; then
        local installed_version=$(helm list -n istio-system -o json 2>/dev/null | jq -r '.[] | select(.name=="istio-base") | .app_version')
        if [[ "$installed_version" == "$ISTIO_VERSION" ]]; then
            echo "true"
        else
            echo "false"
        fi
    else
        echo "false"
    fi
}

check_knative_version() {
    if kubectl get knativeserving knative-serving -n knative-serving &>/dev/null; then
        local installed_version=$(kubectl get knativeserving knative-serving -n knative-serving -o jsonpath='{.spec.version}' 2>/dev/null)
        if [[ "$installed_version" == "$KNATIVE_SERVING_VERSION" ]]; then
            echo "true"
        else
            echo "false"
        fi
    else
        echo "false"
    fi
}

check_certmanager_version() {
    if helm list -n cert-manager 2>/dev/null | grep -q "cert-manager"; then
        local installed_version=$(helm list -n cert-manager -o json 2>/dev/null | jq -r '.[] | select(.name=="cert-manager") | .app_version')
        if [[ "$installed_version" == "$CERT_MANAGER_VERSION" ]]; then
            echo "true"
        else
            echo "false"
        fi
    else
        echo "false"
    fi
}

check_kserve_version() {
    if helm list -n kserve 2>/dev/null | grep -q "kserve"; then
        local installed_chart=$(helm list -n kserve -o json 2>/dev/null | jq -r '.[] | select(.name=="kserve") | .chart')
        # Extract version from chart name (e.g., "kserve-v0.15.2" -> "v0.15.2")
        local installed_version="${installed_chart#kserve-}"
        if [[ "$installed_version" == "$KSERVE_VERSION" ]]; then
            echo "true"
        else
            echo "false"
        fi
    else
        echo "false"
    fi
}

# --- Cleanup Function ---
cleanup() {
    echo "--- Cleaning up previous installations ---"
    
    # Istio cleanup
    echo "Cleaning up Istio..."
    helm uninstall --ignore-not-found istio-ingressgateway -n istio-system 2>/dev/null || true
    helm uninstall --ignore-not-found istiod -n istio-system 2>/dev/null || true
    helm uninstall --ignore-not-found istio-base -n istio-system 2>/dev/null || true
    kubectl delete namespace istio-system --ignore-not-found=true --timeout=30s 2>/dev/null || true
    # Wait for namespace to be fully deleted
    kubectl wait --for=delete namespace/istio-system --timeout=60s 2>/dev/null || true
    echo "Istio cleanup complete."

    # Cert-Manager cleanup
    echo "Cleaning up Cert-Manager..."
    helm uninstall --ignore-not-found cert-manager -n cert-manager 2>/dev/null || true
    kubectl delete namespace cert-manager --ignore-not-found=true --timeout=30s 2>/dev/null || true
    # Clean up Cert-Manager CRDs
    kubectl delete crd certificaterequests.cert-manager.io --ignore-not-found=true 2>/dev/null || true
    kubectl delete crd certificates.cert-manager.io --ignore-not-found=true 2>/dev/null || true
    kubectl delete crd challenges.acme.cert-manager.io --ignore-not-found=true 2>/dev/null || true
    kubectl delete crd clusterissuers.cert-manager.io --ignore-not-found=true 2>/dev/null || true
    kubectl delete crd issuers.cert-manager.io --ignore-not-found=true 2>/dev/null || true
    kubectl delete crd orders.acme.cert-manager.io --ignore-not-found=true 2>/dev/null || true
    # Wait for namespace to be fully deleted
    kubectl wait --for=delete namespace/cert-manager --timeout=60s 2>/dev/null || true
    echo "Cert-Manager cleanup complete."

    # Knative cleanup
    echo "Cleaning up Knative..."
    kubectl delete KnativeServing knative-serving -n knative-serving --ignore-not-found=true --timeout=30s 2>/dev/null || true
    helm uninstall --ignore-not-found knative-operator -n knative-serving 2>/dev/null || true
    helm uninstall --ignore-not-found knative-operator -n knative-operator 2>/dev/null || true
    kubectl delete namespace knative-serving --ignore-not-found=true --timeout=30s 2>/dev/null || true
    kubectl delete namespace knative-operator --ignore-not-found=true --timeout=30s 2>/dev/null || true
    # Clean up Knative CRDs
    kubectl delete crd knativeeventings.operator.knative.dev --ignore-not-found=true 2>/dev/null || true
    kubectl delete crd knativeservings.operator.knative.dev --ignore-not-found=true 2>/dev/null || true
    # Wait for namespaces to be fully deleted
    kubectl wait --for=delete namespace/knative-serving --timeout=60s 2>/dev/null || true
    kubectl wait --for=delete namespace/knative-operator --timeout=60s 2>/dev/null || true
    echo "Knative cleanup complete."

    # KServe cleanup
    echo "Cleaning up KServe..."
    helm uninstall --ignore-not-found kserve -n kserve 2>/dev/null || true
    helm uninstall --ignore-not-found kserve-crd -n kserve 2>/dev/null || true
    kubectl delete namespace kserve --ignore-not-found=true --timeout=30s 2>/dev/null || true
    # Wait for namespace to be fully deleted
    kubectl wait --for=delete namespace/kserve --timeout=60s 2>/dev/null || true
    echo "KServe cleanup complete."
    echo "------------------------------------------"
}

# Call cleanup function if --clean flag is passed
if [[ "$CLEAN_INSTALL" == "true" ]]; then
    cleanup
    
    # Give time for all resources to be fully cleaned up
    echo "Waiting for resources to be fully cleaned up..."
    sleep 5
fi

# --- Prerequisite Check ---
if ! command -v helm &> /dev/null; then
    echo "ERROR: helm is not installed. Please install helm to continue." >&2
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl is not installed. Please install kubectl to continue." >&2
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is not installed. Please install jq to continue." >&2
    exit 1
fi

# --- Istio Installation ---
if [[ $(check_istio_version) == "true" ]]; then
    echo "✅ Istio ${ISTIO_VERSION} is already installed. Skipping..."
else
    echo "--- Installing Istio ${ISTIO_VERSION} ---"
    helm repo add istio https://istio-release.storage.googleapis.com/charts --force-update >/dev/null 2>&1
    helm install istio-base istio/base -n istio-system --wait --set defaultRevision=default --create-namespace --version ${ISTIO_VERSION} >/dev/null 2>&1
    helm install istiod istio/istiod -n istio-system --wait --version ${ISTIO_VERSION} >/dev/null 2>&1
    helm install istio-ingressgateway istio/gateway -n istio-system --version ${ISTIO_VERSION} >/dev/null 2>&1

    echo "Verifying Istio installation..."
    if kubectl wait --for=condition=Ready pod -l app=istio-ingressgateway -n istio-system --timeout=300s >/dev/null; then
        echo "✅ Istio ingress gateway is ready." 
    else
        echo "❌ ERROR: Istio ingress gateway failed to become ready." >&2
        exit 1
    fi
fi

# --- Knative Installation ---
if [[ $(check_knative_version) == "true" ]]; then
    echo "✅ Knative Serving ${KNATIVE_SERVING_VERSION} is already installed. Skipping..."
else
    echo "--- Installing Knative Serving ${KNATIVE_SERVING_VERSION} ---"
    helm install knative-operator --namespace knative-serving --create-namespace --wait \
      https://github.com/knative/operator/releases/download/knative-${KNATIVE_OPERATOR_VERSION}/knative-operator-${KNATIVE_OPERATOR_VERSION}.tgz

    kubectl apply -f - <<EOF
apiVersion: operator.knative.dev/v1beta1
kind: KnativeServing
metadata:
  name: knative-serving
  namespace: knative-serving
spec:
  version: "${KNATIVE_SERVING_VERSION}"
  config:
    domain:
      "example.com": ""
EOF

    echo "Waiting for Knative operator to reconcile..."
    sleep 10

    echo "Verifying Knative installation..."
    # First wait for the KnativeServing resource to be ready
    if kubectl wait --for=condition=Ready knativeserving/knative-serving -n knative-serving --timeout=300s >/dev/null 2>&1; then
        echo "KnativeServing resource is ready, checking pods..."
        # Now wait for the controller and webhook pods
        if kubectl wait --for=condition=Ready pod -l app=controller -n knative-serving --timeout=300s >/dev/null 2>&1 && \
           kubectl wait --for=condition=Ready pod -l app=webhook -n knative-serving --timeout=300s >/dev/null 2>&1; then
            echo "✅ Knative Serving is ready."
        else
            echo "❌ ERROR: Knative Serving pods failed to become ready." >&2
            exit 1
        fi
    else
        echo "❌ ERROR: KnativeServing resource failed to become ready." >&2
        exit 1
    fi
fi

# --- Cert-Manager Installation ---
if [[ $(check_certmanager_version) == "true" ]]; then
    echo "✅ Cert-Manager ${CERT_MANAGER_VERSION} is already installed. Skipping..."
else
    echo "--- Installing Cert-Manager ${CERT_MANAGER_VERSION} ---"
    helm repo add jetstack https://charts.jetstack.io --force-update >/dev/null 2>&1
    helm install \
      cert-manager jetstack/cert-manager \
      --namespace cert-manager \
      --create-namespace \
      --version ${CERT_MANAGER_VERSION} \
      --set installCRDs=true >/dev/null 2>&1

    echo "Verifying Cert-Manager installation..."
    if kubectl wait --for=condition=Ready pod --all -n cert-manager --timeout=300s >/dev/null; then
        echo "✅ Cert-Manager is ready."
    else
        echo "❌ ERROR: Cert-Manager failed to become ready." >&2
        exit 1
    fi
fi

# --- KServe Installation ---
if [[ $(check_kserve_version) == "true" ]]; then
    echo "✅ KServe ${KSERVE_VERSION} is already installed. Skipping..."
else
    echo "--- Installing KServe ${KSERVE_VERSION} ---"
    helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version ${KSERVE_VERSION} --namespace kserve --create-namespace --wait >/dev/null 2>&1
    helm install kserve oci://ghcr.io/kserve/charts/kserve --version ${KSERVE_VERSION} --namespace kserve --wait >/dev/null 2>&1

    echo "Verifying KServe installation..."
    if kubectl wait --for=condition=Ready pod -l control-plane=kserve-controller-manager -n kserve --timeout=300s >/dev/null; then
        echo "✅ KServe is ready."
    else
        echo "❌ ERROR: KServe failed to become ready." >&2
        exit 1
    fi
fi

echo ""
echo "🎉 All components are ready!"
echo ""
echo "Installed versions:"
echo "  - Istio: ${ISTIO_VERSION}"
echo "  - Knative Serving: ${KNATIVE_SERVING_VERSION}"
echo "  - Cert-Manager: ${CERT_MANAGER_VERSION}"
echo "  - KServe: ${KSERVE_VERSION}"

# Set up ingress routing for Kind if we're running in a Kind cluster
if kubectl config current-context | grep -q "kind-"; then
    echo ""
    echo "Detected Kind cluster. Setting up ingress routing..."
    if [[ -f "$(dirname "$0")/setup_ingress_routing.sh" ]]; then
        bash "$(dirname "$0")/setup_ingress_routing.sh"
    fi
fi
