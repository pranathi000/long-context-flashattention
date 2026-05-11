# experiments/memory_scaling.py

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


def benchmark_model(
    model_name,
    model,
    Q,
    K,
    V,
    device
):

    if device == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()

    # ---------------------------------------------------
    # Timing start
    # ---------------------------------------------------
    start_time = time.time()

    # ---------------------------------------------------
    # Forward execution
    # ---------------------------------------------------
    if model_name == "chunked":

        output = model.execute(
            request_id="benchmark_request",
            Q=Q,
            K=K,
            V=V
        )

    else:

        output = model.forward(
            Q,
            K,
            V
        )

    # ---------------------------------------------------
    # GPU synchronization
    # ---------------------------------------------------
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # ---------------------------------------------------
    # Peak memory usage
    # ---------------------------------------------------
    peak_memory = 0

    if device == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    latency = end_time - start_time

    return {
        "output_shape": tuple(output.shape),
        "peak_memory_mb": peak_memory,
        "latency_sec": latency
    }


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nRunning Memory Scaling Benchmark")
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
        # 1. Naive Attention
        # ===================================================
        try:

            naive_attention = NaiveAttention(
                d_model=d_model
            )

            naive_result = benchmark_model(
                "naive",
                naive_attention,
                Q,
                K,
                V,
                device
            )

            print(
                f"\nNaive Attention"
            )

            print(
                f"Peak Memory : "
                f"{naive_result['peak_memory_mb']:.2f} MB"
            )

            print(
                f"Latency : "
                f"{naive_result['latency_sec']:.4f} sec"
            )

        except RuntimeError as e:

            naive_result = {
                "peak_memory_mb": None,
                "latency_sec": None
            }

            print(
                f"\nNaive Attention OOM"
            )

            print(e)

        # ===================================================
        # 2. Tiled FlashAttention
        # ===================================================
        try:

            tiled_attention = FlashAttention(
                d_model=d_model,
                tile_size=tile_size
            )

            tiled_result = benchmark_model(
                "tiled",
                tiled_attention,
                Q,
                K,
                V,
                device
            )

            print(
                f"\nTiled FlashAttention"
            )

            print(
                f"Peak Memory : "
                f"{tiled_result['peak_memory_mb']:.2f} MB"
            )

            print(
                f"Latency : "
                f"{tiled_result['latency_sec']:.4f} sec"
            )

        except RuntimeError as e:

            tiled_result = {
                "peak_memory_mb": None,
                "latency_sec": None
            }

            print(
                f"\nTiled Attention OOM"
            )

            print(e)

        # ===================================================
        # 3. Chunked Long-Context Attention
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

            chunked_result = benchmark_model(
                "chunked",
                chunked_attention,
                Q,
                K,
                V,
                device
            )

            print(
                f"\nChunked Long-Context Attention"
            )

            print(
                f"Peak Memory : "
                f"{chunked_result['peak_memory_mb']:.2f} MB"
            )

            print(
                f"Latency : "
                f"{chunked_result['latency_sec']:.4f} sec"
            )

        except RuntimeError as e:

            chunked_result = {
                "peak_memory_mb": None,
                "latency_sec": None
            }

            print(
                f"\nChunked Attention OOM"
            )

            print(e)

        # ---------------------------------------------------
        # Store benchmark results
        # ---------------------------------------------------
        results.append({

            "seq_len": seq_len,

            "naive_memory_mb":
                naive_result["peak_memory_mb"],

            "naive_latency_sec":
                naive_result["latency_sec"],

            "tiled_memory_mb":
                tiled_result["peak_memory_mb"],

            "tiled_latency_sec":
                tiled_result["latency_sec"],

            "chunked_memory_mb":
                chunked_result["peak_memory_mb"],

            "chunked_latency_sec":
                chunked_result["latency_sec"]
        })

    # ---------------------------------------------------
    # Results dataframe
    # ---------------------------------------------------
    results_df = pd.DataFrame(results)

    print("\n==============================")
    print("FINAL BENCHMARK RESULTS")
    print("==============================\n")

    print(results_df)

    # ---------------------------------------------------
    # Save results
    # ---------------------------------------------------
    output_path = os.path.join(
        CURRENT_DIR,
        "memory_scaling_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )