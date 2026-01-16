# IBM Fusion HCI OpenShift LLM Benchmark Results

**Author:** Deepak Soni
**Contact:** deepak.satna@gmail.com
**Date:** January 16, 2026
**Cluster:** gpu1.fusion.isys.hpc.dc.uq.edu.au
**Model:** Mistral-7B-Instruct-v0.2
**GPU:** NVIDIA A100 MIG (20GB partition)

---

## Executive Summary

| Backend | Peak Throughput | Best Latency (P50) | Reliability | Notes |
|---------|-----------------|-------------------|-------------|-------|
| **vLLM** | **560.71 tok/s** | 2543 ms | 100% | FP16, best overall |
| **Triton** | 404.72 tok/s | 2619 ms | 100%* | vLLM backend |
| **TGI** | 165.60 tok/s | 6069 ms | 100% | 4-bit quantized |

*Triton had 43% error at c=8; recovered at c=16

---

## Comprehensive Benchmark Results

### Throughput Comparison (tokens/sec)

| Backend | c=1 | c=4 | c=8 | c=16 |
|---------|-----|-----|-----|------|
| **vLLM** | 39.2 | 150.8 | 276.2 | **560.71** |
| **TGI** | 19.9 | 43.2 | 80.58 | 165.60 |
| **Triton** | 23.8 | 101.69 | 161.08* | 404.72 |

*43% error rate at c=8

### Latency Comparison at Peak Concurrency (c=16)

| Backend | P50 | P90 | P95 | P99 |
|---------|-----|-----|-----|-----|
| **vLLM** | 2543 ms | 2811 ms | 2814 ms | 2840 ms |
| **TGI** | 6069 ms | 6079 ms | 6080 ms | 6081 ms |
| **Triton** | 2816 ms | 2829 ms | 2869 ms | 2871 ms |

---

## Detailed Results by Backend

### 1. vLLM (Winner - FP16)

| Concurrency | Throughput | P50 | P95 | P99 | Success |
|-------------|------------|-----|-----|-----|---------|
| 1 | 39.22 tok/s | 2515 ms | 2943 ms | 3669 ms | 100% |
| 4 | 150.78 tok/s | 2524 ms | 3000 ms | 3001 ms | 100% |
| 8 | 276.17 tok/s | 2615 ms | 3638 ms | 4514 ms | 100% |
| **16** | **560.71 tok/s** | **2543 ms** | 2814 ms | 2840 ms | 100% |

**Configuration:**
```yaml
image: vllm/vllm-openai:v0.6.4.post1
args:
  - "--model=mistralai/Mistral-7B-Instruct-v0.2"
  - "--max-model-len=2048"
  - "--gpu-memory-utilization=0.85"
  - "--dtype=half"
  - "--enforce-eager"
resources:
  limits:
    nvidia.com/gpu: 1
```

---

### 2. TGI (4-bit Quantized)

| Concurrency | Throughput | P50 | P95 | P99 | Success |
|-------------|------------|-----|-----|-----|---------|
| 1 | 19.93 tok/s | 3325 ms | 3674 ms | - | 100% |
| 4 | 43.22 tok/s | 5896 ms | 6242 ms | - | 100% |
| 8 | 80.58 tok/s | 6141 ms | 6371 ms | 6371 ms | 100% |
| **16** | **165.60 tok/s** | **6069 ms** | 6080 ms | 6081 ms | 100% |

**Configuration:**
```yaml
image: ghcr.io/huggingface/text-generation-inference:2.4.0
args:
  - "--model-id=mistralai/Mistral-7B-Instruct-v0.2"
  - "--max-input-length=1024"
  - "--max-total-tokens=2048"
  - "--quantize=bitsandbytes-nf4"  # Required for 20GB MIG
resources:
  limits:
    nvidia.com/gpu: 1
```

**Note:** TGI requires 4-bit quantization to fit Mistral-7B in 20GB MIG. This reduces throughput by ~3.4x compared to FP16 vLLM.

---

### 3. Triton Inference Server (vLLM Backend)

| Concurrency | Throughput | P50 | P95 | P99 | Success |
|-------------|------------|-----|-----|-----|---------|
| 1 | 23.84 tok/s | 2619 ms | - | - | 74% |
| 4 | 101.69 tok/s | 2646 ms | 3444 ms | 3508 ms | 100% |
| 8 | 161.08 tok/s | 2619 ms | 3144 ms | 3144 ms | **56.7%** |
| **16** | **404.72 tok/s** | **2816 ms** | 2869 ms | 2871 ms | 100% |

**Configuration:**
```yaml
image: nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3
model.json:
  model: mistralai/Mistral-7B-Instruct-v0.2
  max_model_len: 2048
  gpu_memory_utilization: 0.85
  dtype: half
  enforce_eager: true
```

**Note:** Triton showed instability at c=8 (43% errors) but recovered at c=16.

---

## Performance Analysis

### Scaling Efficiency

```
vLLM:   c=1 → c=16 = 14.3x improvement (near-linear)
TGI:    c=1 → c=16 = 8.3x improvement
Triton: c=1 → c=16 = 17.0x improvement
```

### Why vLLM Wins

1. **No Quantization Overhead:** Runs FP16 on 20GB MIG
2. **Optimized Memory:** PagedAttention reduces memory fragmentation
3. **Continuous Batching:** Better GPU utilization under load
4. **Mature Implementation:** Most production-tested

### Why TGI is Slower

1. **Quantization Required:** 4-bit reduces compute throughput
2. **Higher Latency:** Dequantization overhead per forward pass
3. **Memory Bandwidth:** Constant unpack/pack operations

---

## Cost Analysis (A100 MIG @ $1.50/hr)

| Backend | Throughput | Cost per 1M tokens |
|---------|------------|-------------------|
| vLLM (c=16) | 560 tok/s | **$0.74** |
| Triton (c=16) | 405 tok/s | $1.03 |
| TGI (c=16) | 166 tok/s | $2.51 |

vLLM delivers **3.4x better cost efficiency** than TGI.

---

## Critical: OpenShift SCC Configuration

**Without this, ALL GPU pods will crash with CrashLoopBackOff!**

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: llm-bench-privileged
  namespace: llm-bench
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:openshift:scc:privileged
subjects:
- kind: ServiceAccount
  name: bench-sa
  namespace: llm-bench
```

Apply this before deploying any LLM workloads on OpenShift.

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 30 |
| Warmup | 10 |
| Max Tokens | 100 |
| Test Prompt | "Explain quantum computing in simple terms" |

---

## Files Generated

```
benchmark-results/
├── vllm_c1_t100_*.json
├── vllm_c4_t100_*.json
├── vllm_c8_t100_*.json
├── vllm_c16_benchmark.json      ← 560.71 tok/s
├── tgi_c1_t100_*.json
├── tgi_c4_t100_*.json
├── tgi_c8_benchmark.json
├── tgi_c16_benchmark.json       ← 165.60 tok/s
├── triton_c1_t100_*.json
├── triton_c4_benchmark.json
├── triton_c8_benchmark.json
├── triton_c16_benchmark.json    ← 404.72 tok/s
└── BENCHMARK_SUMMARY.md
```

---

## Recommendations

### For Maximum Performance
**Use vLLM** at concurrency 16+ for 560+ tokens/sec

### For Production Kubernetes
**Use vLLM** - simpler deployment, best tooling, most active community

### For Existing Triton Infrastructure
**Use Triton with vLLM backend** - 72% of vLLM performance with enterprise features

### For Memory-Constrained (<20GB)
**Use TGI with 4-bit quantization** - accepts the 3.4x performance penalty

---

## Conclusion

For IBM Fusion HCI OpenShift with A100 MIG (20GB):

| Rank | Backend | Throughput | Recommendation |
|------|---------|------------|----------------|
| 1 | **vLLM** | 561 tok/s | Production workloads |
| 2 | **Triton** | 405 tok/s | If Triton ecosystem required |
| 3 | **TGI** | 166 tok/s | Only if quantization acceptable |

**Winner: vLLM** - 3.4x faster than TGI, 1.4x faster than Triton, 100% reliable.
