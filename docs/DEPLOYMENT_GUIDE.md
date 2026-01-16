# Deployment Guide

This guide covers deploying LLM inference servers on OpenShift with GPU support.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Direct OpenShift Deployment](#direct-openshift-deployment)
3. [ACM Managed Cluster Deployment](#acm-managed-cluster-deployment)
4. [Verifying Deployment](#verifying-deployment)
5. [Exposing Services](#exposing-services)
6. [Cleanup](#cleanup)

---

## Prerequisites

### 1. OpenShift CLI

```bash
# Install oc CLI
# macOS
brew install openshift-cli

# Linux
curl -LO https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/openshift-client-linux.tar.gz
tar xvf openshift-client-linux.tar.gz
sudo mv oc kubectl /usr/local/bin/
```

### 2. Cluster Access

```bash
# Login to your OpenShift cluster
oc login https://api.your-cluster.example.com:6443 -u admin

# Verify access
oc whoami
oc get nodes
```

### 3. GPU Node Verification

```bash
# Check for GPU nodes
oc get nodes -l nvidia.com/gpu.present=true

# Verify GPU resources
oc describe node <gpu-node-name> | grep nvidia.com/gpu
```

### 4. Hugging Face Token (Optional)

For gated models, set up HF token:
```bash
# Create secret (if needed)
oc create secret generic hf-token \
  --from-literal=token=hf_xxxxx \
  -n llm-bench
```

---

## Direct OpenShift Deployment

### Step 1: Create Namespace and Service Account

```bash
# Apply prerequisites
oc apply -f manifests/prerequisites/00-namespace.yaml
oc apply -f manifests/prerequisites/01-serviceaccount.yaml

# Verify
oc get namespace llm-bench
oc get sa -n llm-bench
```

### Step 2: Apply Security Context Constraints (CRITICAL)

**This step is REQUIRED for GPU access on OpenShift!**

```bash
# Apply privileged SCC binding
oc apply -f manifests/prerequisites/02-scc-privileged.yaml

# Verify
oc get rolebinding -n llm-bench
```

Without this, pods will crash with `CrashLoopBackOff`.

### Step 3: Deploy LLM Backend

Choose one of the following backends:

#### Option A: vLLM (Recommended)

```bash
oc apply -f manifests/vllm/vllm-mistral.yaml

# Watch deployment
oc get pods -n llm-bench -w
```

#### Option B: TGI (4-bit Quantized)

```bash
oc apply -f manifests/tgi/tgi-mistral.yaml

# Watch deployment
oc get pods -n llm-bench -w
```

#### Option C: Triton with vLLM Backend

```bash
oc apply -f manifests/triton/triton-vllm.yaml

# Watch deployment
oc get pods -n llm-bench -w
```

### Step 4: Wait for Model Download

First deployment takes time to download the model (~14GB):

```bash
# Watch logs
oc logs -f deployment/vllm-server -n llm-bench

# Expected output:
# Downloading model 'mistralai/Mistral-7B-Instruct-v0.2'...
# INFO: Started server process
# INFO: Waiting for application startup.
# INFO: Application startup complete.
```

---

## ACM Managed Cluster Deployment

For deploying to managed clusters via Red Hat Advanced Cluster Management:

### Step 1: Verify ACM Hub Access

```bash
# Login to Hub cluster
oc login https://api.hub-cluster.example.com:6443

# List managed clusters
oc get managedclusters
```

### Step 2: Deploy Prerequisites via ManifestWork

```bash
# Apply to managed cluster (e.g., gpu1)
oc apply -f manifests-acm/prerequisites/acm-deploy-prerequisites.yaml
oc apply -f manifests-acm/prerequisites/acm-scc-privileged.yaml

# Verify ManifestWork status
oc get manifestwork -n gpu1
```

### Step 3: Deploy LLM Backend

```bash
# Deploy vLLM to managed cluster
oc apply -f manifests-acm/vllm/acm-vllm-mistral.yaml

# Check status with feedback
oc get manifestwork llm-bench-vllm -n gpu1 -o yaml
```

### Step 4: Monitor Deployment

```bash
# Check ManifestWork status
oc get manifestwork -n gpu1 -o jsonpath='{.items[*].status.conditions}'

# Get deployment feedback
oc get manifestwork llm-bench-vllm -n gpu1 \
  -o jsonpath='{.status.resourceStatus.manifests[0].statusFeedback.values}'
```

---

## Verifying Deployment

### Check Pod Status

```bash
# List pods
oc get pods -n llm-bench

# Expected output:
# NAME                           READY   STATUS    RESTARTS   AGE
# vllm-server-xxxxx-yyyyy        1/1     Running   0          10m
```

### Check Logs

```bash
# vLLM
oc logs deployment/vllm-server -n llm-bench | tail -50

# TGI
oc logs deployment/tgi-server -n llm-bench | tail -50

# Triton
oc logs deployment/triton-server -n llm-bench | tail -50
```

### Test Health Endpoint

```bash
# Get route URL
ROUTE=$(oc get route vllm-route -n llm-bench -o jsonpath='{.spec.host}')

# Test health
curl -k https://$ROUTE/health
# Expected: {"status":"healthy"}
```

### Test Inference

```bash
# vLLM
curl -k https://$ROUTE/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

---

## Exposing Services

### OpenShift Route (Default)

Routes are created automatically by the manifests:

```bash
# List routes
oc get routes -n llm-bench

# Get URL
oc get route vllm-route -n llm-bench -o jsonpath='{.spec.host}'
```

### NodePort (Alternative)

```bash
# Patch service to NodePort
oc patch svc vllm-service -n llm-bench \
  -p '{"spec": {"type": "NodePort"}}'

# Get NodePort
oc get svc vllm-service -n llm-bench -o jsonpath='{.spec.ports[0].nodePort}'
```

### Port Forward (Local Testing)

```bash
# Forward local port to service
oc port-forward svc/vllm-service 8000:8000 -n llm-bench

# Test locally
curl http://localhost:8000/health
```

---

## Cleanup

### Remove Single Backend

```bash
# Delete vLLM deployment
oc delete -f manifests/vllm/vllm-mistral.yaml
```

### Remove All Resources

```bash
# Delete all LLM resources
oc delete namespace llm-bench
```

### ACM Cleanup

```bash
# Delete ManifestWorks from Hub
oc delete manifestwork -n gpu1 -l app.kubernetes.io/part-of=llm-benchmark
```

---

## Deployment Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Create namespace | ~5s | Instant |
| Apply SCC | ~5s | Instant |
| Pod scheduled | ~30s | Depends on cluster |
| Model download | 5-15 min | ~14GB download |
| Model loading | 2-5 min | GPU memory allocation |
| **Total** | **8-20 min** | First deployment |

Subsequent deployments (with cached model) take 3-5 minutes.

---

## Next Steps

- [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) - Run performance benchmarks
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
