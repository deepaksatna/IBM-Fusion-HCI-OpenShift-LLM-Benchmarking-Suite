# Troubleshooting Guide

Common issues and solutions for LLM inference on OpenShift.

## Table of Contents

1. [CrashLoopBackOff - GPU Access Denied](#crashloopbackoff---gpu-access-denied)
2. [OOMKilled - Out of Memory](#oomkilled---out-of-memory)
3. [ImagePullBackOff](#imagepullbackoff)
4. [503 Service Unavailable](#503-service-unavailable)
5. [Slow Model Loading](#slow-model-loading)
6. [CUDA Errors](#cuda-errors)
7. [TGI Specific Issues](#tgi-specific-issues)
8. [Triton Specific Issues](#triton-specific-issues)
9. [ACM ManifestWork Issues](#acm-manifestwork-issues)

---

## CrashLoopBackOff - GPU Access Denied

### Symptoms

```
NAME                           READY   STATUS             RESTARTS   AGE
vllm-server-xxxxx-yyyyy        0/1     CrashLoopBackOff   5          10m
```

Logs show:
```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```
or
```
Error: unable to open /dev/nvidia0
```

### Root Cause

OpenShift Security Context Constraints (SCC) blocking GPU device access.

### Solution

**Apply privileged SCC to the service account:**

```bash
# Check if SCC is applied
oc get rolebinding -n llm-bench | grep privileged

# If missing, apply it
oc apply -f manifests/prerequisites/02-scc-privileged.yaml

# Verify
oc describe rolebinding llm-bench-privileged -n llm-bench
```

**Alternative: Direct oc command**

```bash
oc adm policy add-scc-to-user privileged \
  -z bench-sa -n llm-bench
```

**Restart the pod after applying SCC:**

```bash
oc delete pod -l app=vllm-server -n llm-bench
```

---

## OOMKilled - Out of Memory

### Symptoms

```
NAME                           READY   STATUS    RESTARTS   AGE
vllm-server-xxxxx-yyyyy        0/1     OOMKilled   3        5m
```

### Root Cause

Model + KV cache exceeds GPU memory (common with 20GB MIG partitions).

### Solutions

#### Solution 1: Reduce Context Length

```yaml
# vLLM
args:
  - "--max-model-len"
  - "2048"    # Reduce from 4096

# TGI
args:
  - "--max-total-tokens"
  - "2048"
```

#### Solution 2: Enable Quantization (TGI)

```yaml
# TGI with 4-bit quantization
args:
  - "--quantize"
  - "bitsandbytes-nf4"
```

#### Solution 3: Reduce GPU Memory Utilization

```yaml
# vLLM
args:
  - "--gpu-memory-utilization"
  - "0.80"    # Reduce from 0.90
```

#### Solution 4: Use Smaller Model

Try TinyLlama for testing:
```yaml
args:
  - "--model"
  - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

---

## ImagePullBackOff

### Symptoms

```
NAME                           READY   STATUS             RESTARTS   AGE
vllm-server-xxxxx-yyyyy        0/1     ImagePullBackOff   0          5m
```

### Solution

#### Check Image Name

```bash
# Verify image exists
podman pull vllm/vllm-openai:v0.6.4.post1
```

#### Check Image Pull Secret (for private registries)

```bash
# Create pull secret
oc create secret docker-registry regcred \
  --docker-server=nvcr.io \
  --docker-username=\$oauthtoken \
  --docker-password=<NGC-API-KEY> \
  -n llm-bench

# Link to service account
oc secrets link bench-sa regcred --for=pull -n llm-bench
```

---

## 503 Service Unavailable

### Symptoms

```bash
curl https://$ROUTE/v1/completions
# Returns: 503 Service Unavailable
```

### Causes and Solutions

#### 1. Pod Not Ready

```bash
# Check pod status
oc get pods -n llm-bench
oc describe pod <pod-name> -n llm-bench

# Wait for readiness
oc wait --for=condition=Ready pod -l app=vllm-server -n llm-bench --timeout=600s
```

#### 2. Model Still Loading

```bash
# Check logs for loading progress
oc logs deployment/vllm-server -n llm-bench | grep -i "loading\|download"
```

#### 3. Health Check Failing

```bash
# Test health endpoint directly
oc exec deployment/vllm-server -n llm-bench -- curl -s localhost:8000/health
```

#### 4. Route Configuration Issue

```bash
# Check route status
oc get route vllm-route -n llm-bench -o yaml

# Verify service endpoints
oc get endpoints vllm-service -n llm-bench
```

---

## Slow Model Loading

### Symptoms

Pod stays in `Running` but not `Ready` for >15 minutes.

### Causes

1. **First download** - ~14GB model needs to download
2. **Slow network** - Hugging Face CDN connectivity
3. **Disk I/O** - Slow storage for model caching

### Solutions

#### Check Download Progress

```bash
oc logs -f deployment/vllm-server -n llm-bench
# Look for: "Downloading model files..."
```

#### Pre-cache Model (Advanced)

Create PVC with pre-downloaded model:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: llm-bench
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 50Gi
```

---

## CUDA Errors

### CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:** See [OOMKilled](#oomkilled---out-of-memory)

### CUDA Device Not Found

```
RuntimeError: No CUDA GPUs are available
```

**Solution:**

1. Verify GPU node label:
```bash
oc get nodes -l nvidia.com/gpu.present=true
```

2. Check GPU operator:
```bash
oc get pods -n nvidia-gpu-operator
```

3. Verify GPU resources:
```bash
oc describe node <gpu-node> | grep nvidia
```

### CUDA Version Mismatch

```
CUDA driver version is insufficient for CUDA runtime version
```

**Solution:** Use compatible container image:
```yaml
# Match CUDA version to driver
image: vllm/vllm-openai:v0.6.4.post1  # CUDA 12.x
```

---

## TGI Specific Issues

### Model Doesn't Fit (No Quantization)

```
Error: Model does not fit in GPU memory
```

**Solution:** Enable quantization:
```yaml
args:
  - "--quantize"
  - "bitsandbytes-nf4"
```

### Flash Attention Error

```
ImportError: flash_attn not available
```

**Solution:** Already handled in official TGI image. If custom image:
```yaml
env:
  - name: DISABLE_FLASH_ATTN
    value: "true"
```

---

## Triton Specific Issues

### Model Repository Error

```
Error: Model repository not found at /models
```

**Solution:** Verify ConfigMap is mounted:
```bash
oc get configmap triton-model-config -n llm-bench -o yaml
oc describe pod <triton-pod> -n llm-bench | grep -A5 Mounts
```

### vLLM Backend Not Loading

```
Error: Failed to load model 'mistral' with backend 'vllm'
```

**Solution:** Check model.json configuration:
```bash
oc get configmap triton-model-config -n llm-bench -o jsonpath='{.data.model\.json}'
```

---

## ACM ManifestWork Issues

### ManifestWork Not Applied

```bash
oc get manifestwork -n gpu1
# Shows: Applied: False
```

**Solution:**

1. Check ManifestWork status:
```bash
oc describe manifestwork llm-bench-vllm -n gpu1
```

2. Verify managed cluster connection:
```bash
oc get managedcluster gpu1 -o jsonpath='{.status.conditions}'
```

### Feedback Not Updating

```bash
oc get manifestwork llm-bench-vllm -n gpu1 -o jsonpath='{.status.resourceStatus}'
# Empty or stale
```

**Solution:** Add feedbackRules to manifestConfigs:
```yaml
manifestConfigs:
  - resourceIdentifier:
      group: apps
      resource: deployments
      name: vllm-server
      namespace: llm-bench
    feedbackRules:
      - type: JSONPaths
        jsonPaths:
          - name: readyReplicas
            path: .status.readyReplicas
```

---

## Diagnostic Commands

### Quick Health Check

```bash
# All-in-one diagnostic
oc get pods,svc,routes,pvc -n llm-bench
oc describe pod -l backend=vllm -n llm-bench | grep -A20 Events
```

### GPU Status

```bash
# Check GPU allocation
oc describe node <gpu-node> | grep -A10 "Allocated resources"

# Check GPU operator pods
oc get pods -n nvidia-gpu-operator
```

### Network Connectivity

```bash
# Test from within cluster
oc debug deployment/vllm-server -n llm-bench -- curl -s localhost:8000/health

# Test route externally
curl -k https://$(oc get route vllm-route -n llm-bench -o jsonpath='{.spec.host}')/health
```

### Resource Usage

```bash
# Check resource consumption
oc top pod -n llm-bench
oc adm top node <gpu-node>
```

---

## Getting Help

If issues persist:

1. Collect diagnostic info:
```bash
oc describe pod <pod-name> -n llm-bench > pod-describe.txt
oc logs <pod-name> -n llm-bench > pod-logs.txt
oc get events -n llm-bench > events.txt
```

2. Check OpenShift/Kubernetes version compatibility
3. Verify GPU operator version
4. Review HuggingFace model compatibility
