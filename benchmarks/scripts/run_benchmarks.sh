#!/bin/bash
# =============================================================================
# LLM Inference Benchmark Suite
# =============================================================================
# Runs comprehensive benchmarks across all backends at various concurrency levels
# Usage: ./run_benchmarks.sh
#
# Author: Deepak Soni
# Contact: deepak.satna@gmail.com
# =============================================================================

set -e

# Configuration
BACKENDS="vllm triton tgi"
CONCURRENCIES="1 4 8 16"
ITERATIONS=30
WARMUP=10
MAX_TOKENS=100
OUTPUT_DIR="../../results/data"
CLIENT_DIR="../client"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "  LLM Inference Benchmark Suite"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Backends: $BACKENDS"
echo "  Concurrency levels: $CONCURRENCIES"
echo "  Iterations per test: $ITERATIONS"
echo "  Warmup requests: $WARMUP"
echo "  Max tokens: $MAX_TOKENS"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to client directory
cd "$CLIENT_DIR"

# Function to run single benchmark
run_benchmark() {
    local backend=$1
    local concurrency=$2
    local output_file="$OUTPUT_DIR/${backend}_c${concurrency}_benchmark.json"

    echo -e "${YELLOW}>>> Testing $backend at concurrency=$concurrency <<<${NC}"

    if python3 inference_client.py \
        --backend "$backend" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$concurrency" \
        --output-file "$output_file"; then

        echo -e "${GREEN}SUCCESS: Results saved to $output_file${NC}"
    else
        echo -e "${RED}FAILED: $backend at concurrency=$concurrency${NC}"
    fi

    echo ""

    # Cool down between tests
    sleep 5
}

# Run all benchmarks
total_tests=$(($(echo $BACKENDS | wc -w) * $(echo $CONCURRENCIES | wc -w)))
current_test=0

for backend in $BACKENDS; do
    echo "============================================================"
    echo "  Backend: $backend"
    echo "============================================================"

    for c in $CONCURRENCIES; do
        current_test=$((current_test + 1))
        echo "Test $current_test of $total_tests"
        run_benchmark "$backend" "$c"
    done
done

echo "============================================================"
echo "  Benchmark Suite Complete!"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To generate visualizations:"
echo "  cd ../visualization"
echo "  python3 generate_plots.py"
echo ""
