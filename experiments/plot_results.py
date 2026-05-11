# experiments/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Load benchmark CSV
# -------------------------------------------------------
memory_df = pd.read_csv(
    "experiments/memory_scaling_results.csv"
)

latency_df = pd.read_csv(
    "experiments/latency_scaling_results.csv"
)

# =======================================================
# MEMORY SCALING PLOT
# =======================================================

plt.figure(figsize=(10, 6))

plt.plot(
    memory_df["seq_len"],
    memory_df["naive_memory_mb"],
    marker='o',
    linewidth=2,
    label="Naive Attention"
)

plt.plot(
    memory_df["seq_len"],
    memory_df["tiled_memory_mb"],
    marker='o',
    linewidth=2,
    label="Tiled FlashAttention"
)

plt.plot(
    memory_df["seq_len"],
    memory_df["chunked_memory_mb"],
    marker='o',
    linewidth=2,
    label="Chunked/Streaming Attention"
)

plt.xlabel("Sequence Length")
plt.ylabel("Peak GPU Memory (MB)")

plt.title(
    "Memory Scaling: Naive vs FlashAttention"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "experiments/memory_scaling_plot.png",
    dpi=300
)

print(
    "Saved: memory_scaling_plot.png"
)

# =======================================================
# LATENCY SCALING PLOT
# =======================================================

plt.figure(figsize=(10, 6))

plt.plot(
    latency_df["seq_len"],
    latency_df["naive_avg_latency_sec"],
    marker='o',
    linewidth=2,
    label="Naive Attention"
)

plt.plot(
    latency_df["seq_len"],
    latency_df["tiled_avg_latency_sec"],
    marker='o',
    linewidth=2,
    label="Tiled FlashAttention"
)

plt.plot(
    latency_df["seq_len"],
    latency_df["chunked_avg_latency_sec"],
    marker='o',
    linewidth=2,
    label="Chunked/Streaming Attention"
)

plt.xlabel("Sequence Length")

plt.ylabel("Latency (sec)")

plt.title(
    "Latency Scaling: Naive vs FlashAttention"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "experiments/latency_scaling_plot.png",
    dpi=300
)

print(
    "Saved: latency_scaling_plot.png"
)

# =======================================================
# MEMORY REDUCTION FACTOR PLOT
# =======================================================

memory_reduction = (
    memory_df["naive_memory_mb"] /
    memory_df["chunked_memory_mb"]
)

plt.figure(figsize=(10, 6))

plt.plot(
    memory_df["seq_len"],
    memory_reduction,
    marker='o',
    linewidth=2
)

plt.xlabel("Sequence Length")

plt.ylabel("Memory Reduction Factor")

plt.title(
    "Memory Reduction vs Naive Attention"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "experiments/memory_reduction_plot.png",
    dpi=300
)

print(
    "Saved: memory_reduction_plot.png"
)

# =======================================================
# SHOW ALL PLOTS
# =======================================================

plt.show()