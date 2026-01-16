# Benchmark Guide

This guide covers running LLM inference benchmarks and generating performance visualizations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Benchmark Client Usage](#benchmark-client-usage)
4. [Running Full Benchmark Suite](#running-full-benchmark-suite)
5. [Understanding Results](#understanding-results)
6. [Generating Visualizations](#generating-visualizations)
7. [Customizing Benchmarks](#customizing-benchmarks)

---

## Prerequisites

### 1. Python Environment

```bash
cd benchmarks/client

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Backend Access

```bash
# Get route URLs
VLLM_URL=$(oc get route vllm-route -n llm-bench -o jsonpath='{.spec.host}')
TGI_URL=$(oc get route tgi-route -n llm-bench -o jsonpath='{.spec.host}')
TRITON_URL=$(oc get route triton-route -n llm-bench -o jsonpath='{.spec.host}')

# Test health endpoints
curl -k https://$VLLM_URL/health
curl -k https://$TGI_URL/health
curl -k https://$TRITON_URL/v2/health/ready
```

### 3. Update Endpoint Configuration

Edit `benchmarks/client/inference_client.py` to match your cluster:

```python
BACKEND_CONFIGS = {
    "vllm": {
        "endpoint": "https://vllm-route-llm-bench.apps.YOUR-CLUSTER.example.com/v1/completions",
        "health": "https://vllm-route-llm-bench.apps.YOUR-CLUSTER.example.com/health",
        "format": "openai",
    },
    # ... update other backends
}
```

---

## Quick Start

### Single Backend Test

```bash
cd benchmarks/client

# Quick test with vLLM (10 iterations)
python3 inference_client.py --backend vllm --iterations 10 --concurrency 1

# Expected output:
# ============================================================
#   Benchmark: VLLM
# ============================================================
#   Iterations: 10
#   Warmup: 10
#   Max Tokens: 128
#   Concurrency: 1
#
# Backend ready.
# Warming up (10 requests)...
# Warmup complete.
# Running benchmark (10 requests)...
#
# ============================================================
#   Results: VLLM
# ============================================================
#
# Requests:
#   Successful: 10
#   Failed:     0
#   Error Rate: 0.0%
```

---

## Benchmark Client Usage

### Basic Options

```bash
python3 inference_client.py --help

Options:
  --backend {vllm,triton,tgi}  Backend to benchmark (required)
  --iterations INT             Number of requests (default: 100)
  --warmup INT                 Warmup requests (default: 10)
  --max-tokens INT             Max tokens to generate (default: 128)
  --concurrency INT            Concurrent workers (default: 1)
  --output-file FILE           Save JSON results
  --quiet                      Suppress progress output
```

### Example Commands

```bash
# Standard benchmark
python3 inference_client.py --backend vllm --iterations 100 --concurrency 8

# High concurrency test
python3 inference_client.py --backend vllm --iterations 50 --concurrency 16

# Short output (50 tokens)
python3 inference_client.py --backend vllm --iterations 100 --max-tokens 50

# Save results to file
python3 inference_client.py --backend vllm --iterations 30 --concurrency 16 \
  --output-file results/data/vllm_c16.json
```

---

## Running Full Benchmark Suite

### All Backends, Multiple Concurrency Levels

```bash
#!/bin/bash
# run_benchmarks.sh

BACKENDS="vllm triton tgi"
CONCURRENCIES="1 4 8 16"
ITERATIONS=30
MAX_TOKENS=100

for backend in $BACKENDS; do
    for c in $CONCURRENCIES; do
        echo ">>> Testing $backend at concurrency=$c <<<"

        python3 inference_client.py \
            --backend $backend \
            --iterations $ITERATIONS \
            --max-tokens $MAX_TOKENS \
            --concurrency $c \
            --output-file results/data/${backend}_c${c}_benchmark.json

        echo ""
        sleep 5  # Cool down between tests
    done
done
```

### Run Script

```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

---

## Understanding Results

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Tokens/sec** | Token generation throughput | Higher is better |
| **P50 Latency** | Median response time | Lower is better |
| **P95 Latency** | 95th percentile latency | Lower is better |
| **Success Rate** | % of successful requests | 100% ideal |

### JSON Output Format

```json
{
  "backend": "vllm",
  "config": {
    "iterations": 30,
    "warmup": 10,
    "max_tokens": 100,
    "concurrency": 16
  },
  "statistics": {
    "successful_requests": 30,
    "failed_requests": 0,
    "error_rate_percent": 0.0,
    "total_time_sec": 5.35,
    "latency_ms": {
      "min": 2500.98,
      "max": 2839.53,
      "mean": 2661.58,
      "median": 2543.22,
      "p50": 2543.22,
      "p90": 2810.87,
      "p95": 2814.39,
      "p99": 2839.53
    },
    "throughput": {
      "requests_per_sec": 5.61,
      "tokens_per_sec": 560.71
    },
    "tokens": {
      "total_generated": 3000,
      "avg_per_request": 100.0
    }
  },
  "raw_latencies": [...],
  "raw_tokens": [...]
}
```

### Interpreting Throughput

| Tokens/sec | Rating | Suitable For |
|------------|--------|--------------|
| >500 | Excellent | Production high-traffic |
| 200-500 | Good | Production moderate traffic |
| 100-200 | Fair | Development/testing |
| <100 | Poor | Investigation needed |

---

## Generating Visualizations

### Generate All Plots

```bash
cd benchmarks/visualization

# Run visualization script
python3 generate_plots.py

# Output:
# ============================================================
#   IBM Fusion HCI OpenShift LLM Benchmark Visualization
# ============================================================
#
# Loading benchmark data...
#   Loaded 13 benchmark results
#
# Generating visualizations...
#   Saved: throughput_comparison.png
#   Saved: tokens_per_second.png
#   Saved: latency_comparison.png
#   Saved: scaling_efficiency.png
#   Saved: performance_heatmap.png
#   Saved: benchmark_summary_dashboard.png
```

### Generated Plots

| Plot | Description |
|------|-------------|
| `benchmark_summary_dashboard.png` | Comprehensive summary with all key metrics |
| `tokens_per_second.png` | Bar chart comparing throughput across backends |
| `throughput_comparison.png` | Line chart showing throughput scaling |
| `latency_comparison.png` | 4-panel latency analysis |
| `performance_heatmap.png` | Heatmap of metrics by backend and concurrency |
| `scaling_efficiency.png` | How well each backend scales with concurrency |

### Customize Visualization

Edit `generate_plots.py` to:
- Change colors: Modify `COLORS` dict
- Add new metrics: Add functions similar to `plot_throughput_comparison()`
- Change output format: Modify `plt.savefig()` parameters

---

## Customizing Benchmarks

### Custom Prompts

Create a JSON file with prompts:

```json
[
    "Explain machine learning in simple terms.",
    "Write a Python function to sort a list.",
    "What is the capital of France?",
    "Describe the solar system."
]
```

Use with:
```bash
python3 inference_client.py --backend vllm --prompts-file custom_prompts.json
```

### Different Output Lengths

Test with varying token lengths:

```bash
# Short responses
python3 inference_client.py --backend vllm --max-tokens 50 \
  --output-file results/data/vllm_t50.json

# Long responses
python3 inference_client.py --backend vllm --max-tokens 200 \
  --output-file results/data/vllm_t200.json
```

### Stress Testing

High concurrency stress test:

```bash
python3 inference_client.py --backend vllm \
  --iterations 100 \
  --concurrency 32 \
  --warmup 20 \
  --max-tokens 100
```

---

## Expected Results

Based on IBM Fusion HCI with A100 MIG (20GB):

### vLLM (FP16)

| Concurrency | Tokens/sec | P50 Latency |
|-------------|------------|-------------|
| 1 | ~39 | ~2515 ms |
| 4 | ~151 | ~2524 ms |
| 8 | ~276 | ~2615 ms |
| 16 | ~561 | ~2543 ms |

### TGI (4-bit Quantized)

| Concurrency | Tokens/sec | P50 Latency |
|-------------|------------|-------------|
| 1 | ~20 | ~3325 ms |
| 4 | ~43 | ~5896 ms |
| 8 | ~81 | ~6141 ms |
| 16 | ~166 | ~6069 ms |

### Triton (vLLM Backend)

| Concurrency | Tokens/sec | P50 Latency |
|-------------|------------|-------------|
| 1 | ~24 | ~2619 ms |
| 4 | ~102 | ~2646 ms |
| 8 | ~161 | ~2619 ms |
| 16 | ~405 | ~2816 ms |

---

## Troubleshooting

### Connection Refused

```
URLError: Connection refused
```

**Solution:** Verify backend is running and route is accessible:
```bash
oc get pods -n llm-bench
curl -k https://$ROUTE/health
```

### Timeout Errors

```
TimeoutError: Request timed out
```

**Solution:** Increase timeout or reduce concurrency:
```bash
# In inference_client.py, modify timeout parameter
result = single_inference(backend, prompt, max_tokens, timeout=120)
```

### High Error Rate

If error rate > 10%:
1. Reduce concurrency
2. Check backend logs for errors
3. Verify GPU memory isn't exhausted

---

## Next Steps

- Compare results with [results/SUMMARY.md](../results/SUMMARY.md)
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for issues
- Share results by updating the benchmark data
