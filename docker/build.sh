#!/usr/bin/env bash
# =============================================================================
# Docker Image Build Script for LLM Inference Servers
# =============================================================================
# Builds Docker images for vLLM, TGI, and Triton with pre-downloaded models
#
# Author: Deepak Soni
# Contact: deepak.satna@gmail.com
#
# Usage:
#   ./build.sh vllm [image-tag]
#   ./build.sh tgi [image-tag]
#   ./build.sh triton [image-tag]
#   ./build.sh all [registry-prefix]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default registry (update for your environment)
DEFAULT_REGISTRY="docker.io/yourusername"

# Check for HuggingFace token
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN environment variable is required"
    echo "  export HF_TOKEN=hf_xxxxx"
    exit 1
fi

build_vllm() {
    local tag="${1:-${DEFAULT_REGISTRY}/vllm-mistral:latest}"
    echo "Building vLLM image: ${tag}"

    docker build \
        --build-arg HF_TOKEN="${HF_TOKEN}" \
        -t "${tag}" \
        -f "${SCRIPT_DIR}/vllm/Dockerfile" \
        "${SCRIPT_DIR}/vllm"

    echo "vLLM build complete: ${tag}"
}

build_tgi() {
    local tag="${1:-${DEFAULT_REGISTRY}/tgi-mistral:latest}"
    echo "Building TGI image: ${tag}"

    docker build \
        --build-arg HF_TOKEN="${HF_TOKEN}" \
        -t "${tag}" \
        -f "${SCRIPT_DIR}/tgi/Dockerfile" \
        "${SCRIPT_DIR}/tgi"

    echo "TGI build complete: ${tag}"
}

build_triton() {
    local tag="${1:-${DEFAULT_REGISTRY}/triton-mistral:latest}"
    echo "Building Triton image: ${tag}"

    docker build \
        --build-arg HF_TOKEN="${HF_TOKEN}" \
        -t "${tag}" \
        -f "${SCRIPT_DIR}/triton/Dockerfile" \
        "${SCRIPT_DIR}/triton"

    echo "Triton build complete: ${tag}"
}

build_all() {
    local registry="${1:-${DEFAULT_REGISTRY}}"
    echo "Building all images with registry: ${registry}"
    echo ""

    build_vllm "${registry}/vllm-mistral:latest"
    echo ""
    build_tgi "${registry}/tgi-mistral:latest"
    echo ""
    build_triton "${registry}/triton-mistral:latest"

    echo ""
    echo "=============================================="
    echo "  All builds complete!"
    echo "=============================================="
    echo ""
    echo "Images built:"
    echo "  - ${registry}/vllm-mistral:latest"
    echo "  - ${registry}/tgi-mistral:latest"
    echo "  - ${registry}/triton-mistral:latest"
    echo ""
    echo "Push images:"
    echo "  docker push ${registry}/vllm-mistral:latest"
    echo "  docker push ${registry}/tgi-mistral:latest"
    echo "  docker push ${registry}/triton-mistral:latest"
}

# Main
case "${1:-help}" in
    vllm)
        build_vllm "${2:-}"
        ;;
    tgi)
        build_tgi "${2:-}"
        ;;
    triton)
        build_triton "${2:-}"
        ;;
    all)
        build_all "${2:-}"
        ;;
    *)
        echo "Usage: $0 {vllm|tgi|triton|all} [image-tag|registry]"
        echo ""
        echo "Examples:"
        echo "  export HF_TOKEN=hf_xxxxx"
        echo "  $0 vllm myregistry/vllm-mistral:v1"
        echo "  $0 all myregistry.io/llm"
        exit 1
        ;;
esac
