# experiments/latency_scaling.py

import os
import sys
import time
import torch
import pandas as pd

# -------------------------------------------------------
# Project paths
# -------------------------------------------------------
CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

FLASH_ATTENTION_DIR = os.path.join(
    PROJECT_ROOT,
    "flash_attention"
)

CHUNKED_ATTENTION_DIR = os.path.join(
    PROJECT_ROOT,
    "chunked_attention"
)

sys.path.append(FLASH_ATTENTION_DIR)
sys.path.append(CHUNKED_ATTENTION_DIR)

# -------------------------------------------------------
# Imports
# -------------------------------------------------------
from naive_attention import NaiveAttention
from tiled_attention import FlashAttention
from chunk_executor import ChunkedAttentionExecutor


def benchmark_latency(
    model_name,
    model,
    Q,
    K,
    V,
    device,
    warmup_runs=3,
    benchmark_runs=10
):

    # ---------------------------------------------------
    # Warmup runs
    # ---------------------------------------------------
    for _ in range(warmup_runs):

        if model_name == "chunked":

            _ = model.execute(
                request_id="warmup_request",
                Q=Q,
                K=K,
                V=V
            )

        else:

            _ = model.forward(
                Q,
                K,
                V
            )

        if device == "cuda":
            torch.cuda.synchronize()

    # ---------------------------------------------------
    # Benchmark runs
    # ---------------------------------------------------
    latencies = []

    for run_idx in range(benchmark_runs):

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        if model_name == "chunked":

            _ = model.execute(
                request_id=f"benchmark_{run_idx}",
                Q=Q,
                K=K,
                V=V
            )

        else:

            _ = model.forward(
                Q,
                K,
                V
            )

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

        latency = end_time - start_time

        latencies.append(latency)

    avg_latency = (
        sum(latencies) / len(latencies)
    )

    min_latency = min(latencies)

    max_latency = max(latencies)

    return {
        "avg_latency_sec": avg_latency,
        "min_latency_sec": min_latency,
        "max_latency_sec": max_latency
    }


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nRunning Latency Scaling Benchmark")
    print(f"Device : {device}\n")

    # ---------------------------------------------------
    # Configurations
    # ---------------------------------------------------
    d_model = 128

    tile_size = 256

    chunk_size = 2048

    sequence_lengths = [
        1024,
        2048,
        4096,
        8192
    ]

    results = []

    # ---------------------------------------------------
    # Benchmark loop
    # ---------------------------------------------------
    for seq_len in sequence_lengths:

        print(
            f"\n=============================="
        )

        print(
            f"Sequence Length : {seq_len}"
        )

        print(
            f"=============================="
        )

        # ---------------------------------------------------
        # Random QKV tensors
        # ---------------------------------------------------
        Q = torch.randn(
            seq_len,
            d_model,
            device=device
        )

        K = torch.randn(
            seq_len,
            d_model,
            device=device
        )

        V = torch.randn(
            seq_len,
            d_model,
            device=device
        )

        # ===================================================
        # Naive Attention
        # ===================================================
        try:

            naive_attention = NaiveAttention(
                d_model=d_model
            )

            naive_result = benchmark_latency(
                "naive",
                naive_attention,
                Q,
                K,
                V,
                device
            )

            print("\nNaive Attention")

            print(
                f"Average Latency : "
                f"{naive_result['avg_latency_sec']:.6f} sec"
            )

        except RuntimeError as e:

            naive_result = {
                "avg_latency_sec": None
            }

            print(
                "\nNaive Attention OOM"
            )

            print(e)

        # ===================================================
        # Tiled FlashAttention
        # ===================================================
        try:

            tiled_attention = FlashAttention(
                d_model=d_model,
                tile_size=tile_size
            )

            tiled_result = benchmark_latency(
                "tiled",
                tiled_attention,
                Q,
                K,
                V,
                device
            )

            print("\nTiled FlashAttention")

            print(
                f"Average Latency : "
                f"{tiled_result['avg_latency_sec']:.6f} sec"
            )

        except RuntimeError as e:

            tiled_result = {
                "avg_latency_sec": None
            }

            print(
                "\nTiled Attention OOM"
            )

            print(e)

        # ===================================================
        # Chunked Long-Context Attention
        # ===================================================
        try:

            chunked_attention = (
                ChunkedAttentionExecutor(
                    d_model=d_model,
                    tile_size=tile_size,
                    chunk_size=chunk_size,
                    tokens_per_page=256,
                    num_pages=128,
                    device=device
                )
            )

            chunked_result = benchmark_latency(
                "chunked",
                chunked_attention,
                Q,
                K,
                V,
                device
            )

            print(
                "\nChunked Long-Context Attention"
            )

            print(
                f"Average Latency : "
                f"{chunked_result['avg_latency_sec']:.6f} sec"
            )

        except RuntimeError as e:

            chunked_result = {
                "avg_latency_sec": None
            }

            print(
                "\nChunked Attention OOM"
            )

            print(e)

        # ---------------------------------------------------
        # Store benchmark results
        # ---------------------------------------------------
        results.append({

            "seq_len": seq_len,

            "naive_avg_latency_sec":
                naive_result["avg_latency_sec"],

            "tiled_avg_latency_sec":
                tiled_result["avg_latency_sec"],

            "chunked_avg_latency_sec":
                chunked_result["avg_latency_sec"]
        })

    # ---------------------------------------------------
    # Results dataframe
    # ---------------------------------------------------
    results_df = pd.DataFrame(results)

    print("\n==============================")
    print("FINAL LATENCY RESULTS")
    print("==============================\n")

    print(results_df)

    # ---------------------------------------------------
    # Save results
    # ---------------------------------------------------
    output_path = os.path.join(
        CURRENT_DIR,
        "latency_scaling_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )