# HuggingFace Text Generation Inference with pre-downloaded Mistral model
# Optimized for IBM Fusion HCI OpenShift with A100 MIG (20GB)
# Uses 4-bit quantization to fit in 20GB memory
# Fully offline - no runtime downloads required
#
# Author: Deepak Soni
# Contact: deepak.satna@gmail.com

FROM ghcr.io/huggingface/text-generation-inference:2.4.0

# NCCL settings for OpenShift
ENV NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_ASYNC_ERROR_HANDLING=1

# HF token for build-time download
ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# TGI uses /data for model cache
ENV HF_HOME=/data \
    HUGGINGFACE_HUB_CACHE=/data

# Pre-download model using TGI's download utility (BEFORE setting offline mode)
RUN text-generation-server download-weights mistralai/Mistral-7B-Instruct-v0.2

# Clear token from final image
ENV HUGGING_FACE_HUB_TOKEN=""

# NOW set offline mode (AFTER download complete)
ENV HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Verify model download
RUN echo "Model files:" && ls -la /data/ && \
    echo "" && \
    find /data -name "*.safetensors" -o -name "*.json" | head -20

# Note: TGI uses its own entrypoint
# Default args for 20GB MIG (requires quantization):
#   --model-id mistralai/Mistral-7B-Instruct-v0.2
#   --max-input-length 1024
#   --max-total-tokens 2048
#   --quantize bitsandbytes-nf4
