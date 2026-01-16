#!/usr/bin/env python3
"""
IBM Fusion HCI OpenShift LLM Benchmark Visualization
Generates performance comparison graphs for vLLM, Triton, and TGI

Author: Deepak Soni
Contact: deepak.satna@gmail.com
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color scheme
COLORS = {
    'vllm': '#2ecc71',      # Green
    'triton': '#3498db',    # Blue
    'tgi': '#e74c3c'        # Red
}

BACKEND_NAMES = {
    'vllm': 'vLLM',
    'triton': 'NVIDIA Triton',
    'tgi': 'HuggingFace TGI'
}

def load_benchmark_data(benchmark_dir):
    """Load all benchmark JSON files and normalize data structure"""
    results = []
    benchmark_path = Path(benchmark_dir)

    for f in benchmark_path.glob("*.json"):
        if f.name.startswith(('vllm_', 'triton_', 'tgi_')):
            with open(f) as fp:
                data = json.load(fp)

                # Normalize data structure
                result = {
                    'backend': data.get('backend', ''),
                    'concurrency': data.get('config', {}).get('concurrency', 1),
                    'max_tokens': data.get('config', {}).get('max_tokens', 100),
                }

                stats = data.get('statistics', {})
                result['throughput_rps'] = stats.get('throughput', {}).get('requests_per_sec', 0)
                result['tokens_per_sec'] = stats.get('throughput', {}).get('tokens_per_sec', 0)
                result['success_rate'] = 100 - stats.get('error_rate_percent', 0)

                latency = stats.get('latency_ms', {})
                result['avg_latency_ms'] = latency.get('mean', 0)
                result['p50_latency_ms'] = latency.get('p50', latency.get('median', 0))
                result['p95_latency_ms'] = latency.get('p95', 0)
                result['p99_latency_ms'] = latency.get('p99', 0)
                result['min_latency_ms'] = latency.get('min', 0)
                result['max_latency_ms'] = latency.get('max', 0)

                results.append(result)

    return results

def plot_throughput_comparison(data, output_dir):
    """Generate throughput comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group data by backend
    backends = ['vllm', 'triton', 'tgi']

    for backend in backends:
        backend_data = sorted(
            [d for d in data if d['backend'] == backend],
            key=lambda x: x['concurrency']
        )
        if backend_data:
            concurrencies = [d['concurrency'] for d in backend_data]
            throughput = [d['throughput_rps'] for d in backend_data]
            ax.plot(concurrencies, throughput, marker='o', linewidth=2.5, markersize=10,
                   color=COLORS[backend], label=BACKEND_NAMES[backend])

            # Add value labels
            for c, t in zip(concurrencies, throughput):
                ax.annotate(f'{t:.2f}', xy=(c, t), xytext=(5, 5),
                           textcoords='offset points', fontsize=9)

    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Throughput (requests/sec)')
    ax.set_title('Throughput vs Concurrency\nIBM Fusion HCI OpenShift - Mistral-7B')
    ax.legend(loc='upper left')
    ax.set_xticks([1, 4, 8, 16])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: throughput_comparison.png")

def plot_tokens_per_second(data, output_dir):
    """Generate tokens/second bar chart"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique concurrency levels
    concurrencies = sorted(set(d['concurrency'] for d in data))
    x = np.arange(len(concurrencies))
    width = 0.25

    backends = ['vllm', 'triton', 'tgi']

    for i, backend in enumerate(backends):
        backend_data = {d['concurrency']: d['tokens_per_sec']
                       for d in data if d['backend'] == backend}
        values = [backend_data.get(c, 0) for c in concurrencies]
        bars = ax.bar(x + i*width, values, width, label=BACKEND_NAMES[backend],
                     color=COLORS[backend], edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val),
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Token Generation Rate Comparison\nIBM Fusion HCI OpenShift - Mistral-7B (100 tokens output)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(concurrencies)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'tokens_per_second.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tokens_per_second.png")

def plot_latency_comparison(data, output_dir):
    """Generate latency comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    backends = ['vllm', 'triton', 'tgi']

    # P50 Latency vs Concurrency
    ax1 = axes[0, 0]
    for backend in backends:
        backend_data = sorted(
            [d for d in data if d['backend'] == backend],
            key=lambda x: x['concurrency']
        )
        if backend_data:
            concurrencies = [d['concurrency'] for d in backend_data]
            latencies = [d['p50_latency_ms'] for d in backend_data]
            ax1.plot(concurrencies, latencies, marker='o', linewidth=2,
                    color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('P50 Latency vs Concurrency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # P95 Latency vs Concurrency
    ax2 = axes[0, 1]
    for backend in backends:
        backend_data = sorted(
            [d for d in data if d['backend'] == backend],
            key=lambda x: x['concurrency']
        )
        if backend_data:
            concurrencies = [d['concurrency'] for d in backend_data]
            latencies = [d['p95_latency_ms'] for d in backend_data]
            ax2.plot(concurrencies, latencies, marker='s', linewidth=2,
                    color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('P95 Latency vs Concurrency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Latency distribution at peak concurrency
    ax3 = axes[1, 0]
    max_concurrency = max(d['concurrency'] for d in data)
    peak_data = [d for d in data if d['concurrency'] == max_concurrency]

    x = np.arange(4)
    width = 0.25
    metrics = ['avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms']
    metric_names = ['Avg', 'P50', 'P95', 'P99']

    for i, backend in enumerate(backends):
        backend_row = next((d for d in peak_data if d['backend'] == backend), None)
        if backend_row:
            values = [backend_row[m] for m in metrics]
            ax3.bar(x + i*width, values, width, label=BACKEND_NAMES[backend], color=COLORS[backend])

    ax3.set_xlabel('Latency Percentile')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title(f'Latency Distribution at Concurrency={max_concurrency}')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metric_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Min/Max latency comparison
    ax4 = axes[1, 1]
    min_lats = []
    max_lats = []
    backend_labels = []

    for backend in backends:
        backend_row = next((d for d in peak_data if d['backend'] == backend), None)
        if backend_row:
            min_lats.append(backend_row['min_latency_ms'])
            max_lats.append(backend_row['max_latency_ms'])
            backend_labels.append(BACKEND_NAMES[backend])

    x = np.arange(len(backend_labels))
    width = 0.35
    ax4.bar(x - width/2, min_lats, width, label='Min Latency', color='green', alpha=0.7)
    ax4.bar(x + width/2, max_lats, width, label='Max Latency', color='red', alpha=0.7)
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title(f'Min/Max Latency at Concurrency={max_concurrency}')
    ax4.set_xticks(x)
    ax4.set_xticklabels(backend_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Latency Analysis - IBM Fusion HCI OpenShift', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: latency_comparison.png")

def plot_scaling_efficiency(data, output_dir):
    """Generate scaling efficiency chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    backends = ['vllm', 'triton', 'tgi']

    for backend in backends:
        backend_data = sorted(
            [d for d in data if d['backend'] == backend],
            key=lambda x: x['concurrency']
        )

        if len(backend_data) >= 2:
            # Get base throughput at c=1 (or lowest available)
            base_throughput = backend_data[0]['throughput_rps']
            base_concurrency = backend_data[0]['concurrency']

            if base_throughput > 0:
                concurrencies = [d['concurrency'] for d in backend_data]
                actual_throughput = [d['throughput_rps'] for d in backend_data]
                ideal_throughput = [base_throughput * (c / base_concurrency) for c in concurrencies]
                efficiency = [(a / i) * 100 if i > 0 else 0 for a, i in zip(actual_throughput, ideal_throughput)]

                ax.plot(concurrencies, efficiency, marker='o', linewidth=2, markersize=8,
                       color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect Scaling')
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Scaling Efficiency (%)')
    ax.set_title('Scaling Efficiency vs Concurrency\n(100% = Linear Scaling)')
    ax.legend()
    ax.set_xticks([1, 4, 8, 16])
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scaling_efficiency.png")

def plot_summary_dashboard(data, output_dir):
    """Generate comprehensive summary dashboard"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    backends = ['vllm', 'triton', 'tgi']

    # 1. Peak Throughput (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    peak_throughput = {}
    for backend in backends:
        backend_data = [d for d in data if d['backend'] == backend]
        if backend_data:
            peak_throughput[backend] = max(d['throughput_rps'] for d in backend_data)

    bars = ax1.bar([BACKEND_NAMES[b] for b in peak_throughput.keys()],
                   peak_throughput.values(),
                   color=[COLORS[b] for b in peak_throughput.keys()],
                   edgecolor='black', linewidth=1)
    ax1.set_ylabel('Requests/sec')
    ax1.set_title('Peak Throughput', fontweight='bold')
    for bar, val in zip(bars, peak_throughput.values()):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Peak Tokens/sec (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    peak_tokens = {}
    for backend in backends:
        backend_data = [d for d in data if d['backend'] == backend]
        if backend_data:
            peak_tokens[backend] = max(d['tokens_per_sec'] for d in backend_data)

    bars = ax2.bar([BACKEND_NAMES[b] for b in peak_tokens.keys()],
                   peak_tokens.values(),
                   color=[COLORS[b] for b in peak_tokens.keys()],
                   edgecolor='black', linewidth=1)
    ax2.set_ylabel('Tokens/sec')
    ax2.set_title('Peak Token Rate', fontweight='bold')
    for bar, val in zip(bars, peak_tokens.values()):
        ax2.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Best P50 Latency (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    best_latency = {}
    max_c = max(d['concurrency'] for d in data)
    for backend in backends:
        backend_data = [d for d in data if d['backend'] == backend and d['concurrency'] == max_c]
        if backend_data:
            best_latency[backend] = min(d['p50_latency_ms'] for d in backend_data)

    bars = ax3.bar([BACKEND_NAMES[b] for b in best_latency.keys()],
                   best_latency.values(),
                   color=[COLORS[b] for b in best_latency.keys()],
                   edgecolor='black', linewidth=1)
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title(f'P50 Latency (c={max_c})', fontweight='bold')
    for bar, val in zip(bars, best_latency.values()):
        ax3.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Throughput scaling (middle, spans 2 cols)
    ax4 = fig.add_subplot(gs[1, :2])
    for backend in backends:
        backend_data = sorted(
            [d for d in data if d['backend'] == backend],
            key=lambda x: x['concurrency']
        )
        if backend_data:
            concurrencies = [d['concurrency'] for d in backend_data]
            throughput = [d['throughput_rps'] for d in backend_data]
            ax4.plot(concurrencies, throughput, marker='o', linewidth=2.5, markersize=10,
                    color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax4.set_xlabel('Concurrency')
    ax4.set_ylabel('Throughput (req/s)')
    ax4.set_title('Throughput Scaling', fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.set_xticks([1, 4, 8, 16])
    ax4.grid(True, alpha=0.3)

    # 5. Success Rate (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    avg_success = {}
    for backend in backends:
        backend_data = [d for d in data if d['backend'] == backend]
        if backend_data:
            avg_success[backend] = sum(d['success_rate'] for d in backend_data) / len(backend_data)

    bars = ax5.bar([BACKEND_NAMES[b] for b in avg_success.keys()],
                   avg_success.values(),
                   color=[COLORS[b] for b in avg_success.keys()],
                   edgecolor='black', linewidth=1)
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Average Success Rate', fontweight='bold')
    ax5.set_ylim(0, 110)
    for bar, val in zip(bars, avg_success.values()):
        ax5.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Latency scaling (bottom, spans 2 cols)
    ax6 = fig.add_subplot(gs[2, :2])
    for backend in backends:
        backend_data = sorted(
            [d for d in data if d['backend'] == backend],
            key=lambda x: x['concurrency']
        )
        if backend_data:
            concurrencies = [d['concurrency'] for d in backend_data]
            latencies = [d['p95_latency_ms'] for d in backend_data]
            ax6.plot(concurrencies, latencies, marker='s', linewidth=2.5, markersize=10,
                    color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax6.set_xlabel('Concurrency')
    ax6.set_ylabel('P95 Latency (ms)')
    ax6.set_title('P95 Latency Scaling', fontweight='bold')
    ax6.legend(loc='upper left')
    ax6.set_xticks([1, 4, 8, 16])
    ax6.grid(True, alpha=0.3)

    # 7. Winner summary (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Determine winners
    winners = {}
    if peak_throughput:
        winners['Peak Throughput'] = max(peak_throughput, key=peak_throughput.get)
    if peak_tokens:
        winners['Peak Tokens/s'] = max(peak_tokens, key=peak_tokens.get)
    if best_latency:
        winners['Lowest Latency'] = min(best_latency, key=best_latency.get)
    if avg_success:
        winners['Best Reliability'] = max(avg_success, key=avg_success.get)

    summary_text = "WINNERS\n" + "="*25 + "\n\n"
    for category, winner in winners.items():
        summary_text += f"{category}:\n  {BACKEND_NAMES[winner]}\n\n"

    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('LLM Inference Benchmark Summary\nIBM Fusion HCI OpenShift - Mistral-7B on A100 MIG (20GB)',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'benchmark_summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: benchmark_summary_dashboard.png")

def plot_performance_heatmap(data, output_dir):
    """Generate performance heatmap"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    backends = ['vllm', 'triton', 'tgi']
    concurrencies = sorted(set(d['concurrency'] for d in data))

    metrics = [
        ('tokens_per_sec', 'Tokens/sec', 'RdYlGn'),
        ('p50_latency_ms', 'P50 Latency (ms)', 'RdYlGn_r'),
        ('success_rate', 'Success Rate (%)', 'RdYlGn')
    ]

    for idx, (metric_key, metric_label, cmap) in enumerate(metrics):
        ax = axes[idx]

        # Create matrix
        matrix = []
        for backend in backends:
            row = []
            for c in concurrencies:
                val = next((d[metric_key] for d in data
                           if d['backend'] == backend and d['concurrency'] == c), 0)
                row.append(val)
            matrix.append(row)

        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Labels
        ax.set_xticks(range(len(concurrencies)))
        ax.set_xticklabels([f'c={c}' for c in concurrencies])
        ax.set_yticks(range(len(backends)))
        ax.set_yticklabels([BACKEND_NAMES[b] for b in backends])
        ax.set_title(metric_label, fontweight='bold')

        # Add value annotations
        for i in range(len(backends)):
            for j in range(len(concurrencies)):
                val = matrix[i, j]
                text = f'{val:.1f}' if val < 100 else f'{val:.0f}'
                color = 'white' if (val > matrix.max()*0.7 or val < matrix.min()*1.3) else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

    plt.suptitle('Performance Heatmap - IBM Fusion HCI OpenShift', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: performance_heatmap.png")

def main():
    # Paths
    benchmark_dir = Path(__file__).parent
    output_dir = benchmark_dir / "plots"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  IBM Fusion HCI OpenShift LLM Benchmark Visualization")
    print("="*60)
    print(f"\nBenchmark data: {benchmark_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading benchmark data...")
    data = load_benchmark_data(benchmark_dir)
    print(f"  Loaded {len(data)} benchmark results")

    # Show loaded data summary
    for d in data:
        print(f"    - {d['backend']} c={d['concurrency']}: {d['tokens_per_sec']:.1f} tok/s")

    print("\nGenerating visualizations...")

    # Generate plots
    plot_throughput_comparison(data, output_dir)
    plot_tokens_per_second(data, output_dir)
    plot_latency_comparison(data, output_dir)
    plot_scaling_efficiency(data, output_dir)
    plot_performance_heatmap(data, output_dir)
    plot_summary_dashboard(data, output_dir)

    print("\n" + "="*60)
    print("  Visualization Complete!")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
