# Docker Images for LLM Inference Servers

Build Docker images with pre-downloaded Mistral-7B model for offline deployment on OpenShift.

**Author:** Deepak Soni
**Contact:** deepak.satna@gmail.com

## Overview

These Dockerfiles create container images with:
- Pre-downloaded Mistral-7B-Instruct-v0.2 model (~14GB)
- Optimized settings for A100 MIG (20GB) partitions
- Offline mode enabled (no runtime downloads)
- NCCL configured for OpenShift environments

## Prerequisites

1. **Docker** installed and running
2. **HuggingFace Token** with access to Mistral model
3. **Sufficient disk space** (~50GB for all builds)
4. **Network access** during build (for model download)

## Quick Start

```bash
# Set HuggingFace token
export HF_TOKEN=hf_your_token_here

# Build all images
chmod +x build.sh
./build.sh all your-registry.io/llm

# Or build individually
./build.sh vllm your-registry.io/llm/vllm-mistral:v1
./build.sh tgi your-registry.io/llm/tgi-mistral:v1
./build.sh triton your-registry.io/llm/triton-mistral:v1
```

## Image Specifications

### vLLM (`docker/vllm/`)

| Property | Value |
|----------|-------|
| Base Image | `vllm/vllm-openai:v0.6.4.post1` |
| Model | Mistral-7B-Instruct-v0.2 |
| Memory Config | max_model_len=2048, gpu_memory_util=0.85 |
| Precision | FP16 (--dtype half) |
| Port | 8000 (OpenAI-compatible API) |

**Build:**
```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t vllm-mistral:latest -f vllm/Dockerfile vllm/
```

### TGI (`docker/tgi/`)

| Property | Value |
|----------|-------|
| Base Image | `ghcr.io/huggingface/text-generation-inference:2.4.0` |
| Model | Mistral-7B-Instruct-v0.2 |
| Memory Config | max_total_tokens=2048 |
| Quantization | bitsandbytes-nf4 (required for 20GB) |
| Port | 8080 |

**Build:**
```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t tgi-mistral:latest -f tgi/Dockerfile tgi/
```

**Note:** TGI requires 4-bit quantization on 20GB MIG partitions.

### Triton (`docker/triton/`)

| Property | Value |
|----------|-------|
| Base Image | `nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3` |
| Backend | vLLM |
| Model | Mistral-7B-Instruct-v0.2 |
| Memory Config | max_model_len=2048, gpu_memory_util=0.85 |
| Ports | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) |

**Build:**
```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t triton-mistral:latest -f triton/Dockerfile triton/
```

## Pushing to Registry

### Docker Hub
```bash
docker login
docker push your-username/vllm-mistral:latest
docker push your-username/tgi-mistral:latest
docker push your-username/triton-mistral:latest
```

### OpenShift Internal Registry
```bash
# Login to OpenShift registry
oc registry login

# Tag and push
docker tag vllm-mistral:latest default-route-openshift-image-registry.apps.cluster/llm-bench/vllm-mistral:latest
docker push default-route-openshift-image-registry.apps.cluster/llm-bench/vllm-mistral:latest
```

### Private Registry
```bash
# Login to private registry
docker login your-registry.io

# Push images
docker push your-registry.io/llm/vllm-mistral:latest
docker push your-registry.io/llm/tgi-mistral:latest
docker push your-registry.io/llm/triton-mistral:latest
```

## Using Custom Images in Manifests

Update the deployment manifests to use your images:

```yaml
# manifests/vllm/vllm-mistral.yaml
spec:
  containers:
    - name: vllm
      image: your-registry.io/llm/vllm-mistral:latest  # Update this
```

## Build Times

| Image | Approximate Build Time |
|-------|----------------------|
| vLLM | 15-30 min (14GB download) |
| TGI | 15-30 min (14GB download) |
| Triton | 20-40 min (14GB download + config) |

## Troubleshooting

### HF Token Issues
```
Error: 401 Unauthorized
```
Ensure your HuggingFace token has access to `mistralai/Mistral-7B-Instruct-v0.2`.

### Disk Space
```
Error: no space left on device
```
Clean up Docker: `docker system prune -a`

### Network Timeout
```
Error: Connection timed out
```
Model download requires stable internet. Retry with `--network=host` if needed.

## Directory Structure

```
docker/
├── README.md           # This file
├── build.sh           # Build script for all images
├── vllm/
│   ├── Dockerfile     # vLLM image definition
│   ├── download_model.py
│   └── verify_model.py
├── tgi/
│   └── Dockerfile     # TGI image definition
└── triton/
    ├── Dockerfile     # Triton image definition
    ├── download_model.py
    ├── config.pbtxt   # Triton model config
    └── model.json     # vLLM backend config
```

## Security Notes

- HF_TOKEN is only used during build and cleared from final image
- Images run in offline mode (no outbound connections required)
- Model weights are embedded in images (air-gapped deployment ready)

## Contact

For questions or issues, please contact: deepak.satna@gmail.com
