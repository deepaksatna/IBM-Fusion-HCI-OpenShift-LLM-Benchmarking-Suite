# vLLM with pre-downloaded Mistral-7B-Instruct model
# Optimized for IBM Fusion HCI OpenShift with A100 MIG (20GB)
# Fully offline - no runtime downloads required
#
# Author: Deepak Soni
# Contact: deepak.satna@gmail.com

FROM vllm/vllm-openai:v0.6.4.post1

# NCCL settings for OpenShift
ENV NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL

# HF token for build-time download only
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Copy and run download script (BEFORE setting offline mode)
COPY download_model.py /tmp/download_model.py
RUN python3 /tmp/download_model.py && rm /tmp/download_model.py

# Clear HF token from final image (security)
ENV HF_TOKEN=""

# NOW set offline mode (AFTER download complete)
ENV HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    DISABLE_TELEMETRY=1

# Copy and run verification script
COPY verify_model.py /tmp/verify_model.py
RUN python3 /tmp/verify_model.py && rm /tmp/verify_model.py

# Default command optimized for 20GB MIG partition
# Note: --enforce-eager required for MIG compatibility
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "mistralai/Mistral-7B-Instruct-v0.2", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "2048", \
     "--gpu-memory-utilization", "0.85", \
     "--dtype", "half", \
     "--enforce-eager", \
     "--trust-remote-code"]
