# experiments/correctness_validation.py

import sys
import torch

# -------------------------------------------------------
# Import paths
# -------------------------------------------------------
sys.path.append("./flash_attention")
sys.path.append("./chunked_attention")

from naive_attention import NaiveAttention
from tiled_attention import FlashAttention
from streaming_attention import StreamingFlashAttention


def compare_outputs(
    reference_output,
    test_output,
    model_name
):

    # ---------------------------------------------------
    # Absolute error
    # ---------------------------------------------------
    abs_diff = torch.abs(
        reference_output - test_output
    )

    mean_error = abs_diff.mean().item()

    max_error = abs_diff.max().item()

    # ---------------------------------------------------
    # Relative error
    # ---------------------------------------------------
    relative_error = (
        abs_diff /
        (torch.abs(reference_output) + 1e-8)
    ).mean().item()

    print(
        f"\n===== {model_name} Validation ====="
    )

    print(
        f"Mean Absolute Error : "
        f"{mean_error:.10f}"
    )

    print(
        f"Max Absolute Error : "
        f"{max_error:.10f}"
    )

    print(
        f"Mean Relative Error : "
        f"{relative_error:.10f}"
    )

    # ---------------------------------------------------
    # Validation heuristic
    # ---------------------------------------------------
    if mean_error < 1e-4:

        print(
            "\nValidation Status : PASSED"
        )

    else:

        print(
            "\nValidation Status : FAILED"
        )


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nRunning Streaming Attention "
        "Correctness Validation"
    )

    print(f"Device : {device}\n")

    # ---------------------------------------------------
    # Validation configuration
    # ---------------------------------------------------
    seq_len = 1024

    d_model = 64

    tile_size = 128

    # ---------------------------------------------------
    # Random tensors
    # ---------------------------------------------------
    torch.manual_seed(42)

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
    print(
        "Running Naive Attention..."
    )

    naive_attention = NaiveAttention(
        d_model=d_model
    )

    naive_output = naive_attention.forward(
        Q,
        K,
        V
    )

    # ===================================================
    # Tiled FlashAttention
    # ===================================================
    print(
        "\nRunning Tiled FlashAttention..."
    )

    tiled_attention = FlashAttention(
        d_model=d_model,
        tile_size=tile_size
    )

    tiled_output = tiled_attention.forward(
        Q,
        K,
        V
    )

    compare_outputs(
        naive_output,
        tiled_output,
        "Tiled FlashAttention"
    )

    # ===================================================
    # Streaming FlashAttention
    # ===================================================
    print(
        "\nRunning Streaming FlashAttention..."
    )

    streaming_attention = (
        StreamingFlashAttention(
            d_model=d_model,
            tile_size=tile_size,
            tokens_per_page=256,
            num_pages=32,
            device=device
        )
    )

    streaming_output = (
        streaming_attention.forward(
            request_id="validation_request",
            Q=Q,
            K=K,
            V=V
        )
    )

    compare_outputs(
        naive_output,
        streaming_output,
        "Streaming FlashAttention"
    )