# flashattention/tiled_attention.py

import math
import torch

from flashattention_utils import (
    update_running_max,
    update_running_sum
)


class FlashAttention:
    """
    Simplified FlashAttention-style tiled attention.

    Implements:
    - tile-wise QK computation
    - streaming softmax
    - running-max stabilization
    - running-sum normalization
    """

    def __init__(
        self,
        d_model: int,
        tile_size: int
    ):

        self.d_model = d_model
        self.tile_size = tile_size

    def compute_scores(
        self,
        Q_tile: torch.Tensor,
        K_tile: torch.Tensor
    ):

        scores = torch.matmul(
            Q_tile,
            K_tile.T
        )

        scores = scores / math.sqrt(
            self.d_model
        )

        return scores

    def update_output(
        self,
        previous_output: torch.Tensor,
        previous_max: torch.Tensor,
        new_max: torch.Tensor,
        current_exp: torch.Tensor,
        V_tile: torch.Tensor
    ):

        previous_scale = torch.exp(
            previous_max - new_max
        )

        updated_output = (
            previous_output * previous_scale
            + torch.matmul(current_exp, V_tile)
        )

        return updated_output

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ):

        seq_len = Q.shape[0]

        device = Q.device

        output = torch.zeros_like(Q)

        # ---------------------------------------------------
        # Process query tiles
        # ---------------------------------------------------
        for q_start in range(
            0,
            seq_len,
            self.tile_size
        ):

            q_end = min(
                q_start + self.tile_size,
                seq_len
            )

            Q_tile = Q[q_start:q_end]

            # ---------------------------------------------------
            # Streaming accumulators
            # ---------------------------------------------------
            running_max = torch.full(
                (q_end - q_start, 1),
                -float("inf"),
                device=device
            )

            running_sum = torch.zeros(
                (q_end - q_start, 1),
                device=device
            )

            tile_output = torch.zeros(
                (q_end - q_start, self.d_model),
                device=device
            )

            # ---------------------------------------------------
            # Process KV tiles
            # ---------------------------------------------------
            for k_start in range(
                0,
                seq_len,
                self.tile_size
            ):

                k_end = min(
                    k_start + self.tile_size,
                    seq_len
                )

                K_tile = K[k_start:k_end]
                V_tile = V[k_start:k_end]

                # ---------------------------------------------------
                # QK^T tile computation
                # ---------------------------------------------------
                scores = self.compute_scores(
                    Q_tile,
                    K_tile
                )

                # ---------------------------------------------------
                # Running max update
                # ---------------------------------------------------
                _, new_running_max = (
                    update_running_max(
                        running_max,
                        scores
                    )
                )

                # ---------------------------------------------------
                # Running sum update
                # ---------------------------------------------------
                current_exp, running_sum = (
                    update_running_sum(
                        running_sum,
                        running_max,
                        new_running_max,
                        scores
                    )
                )

                # ---------------------------------------------------
                # Output accumulation
                # ---------------------------------------------------
                tile_output = self.update_output(
                    tile_output,
                    running_max,
                    new_running_max,
                    current_exp,
                    V_tile
                )

                running_max = new_running_max

            # ---------------------------------------------------
            # Final normalization
            # ---------------------------------------------------
            output[q_start:q_end] = (
                tile_output / running_sum
            )

        return output


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    seq_len = 4096
    d_model = 128
    tile_size = 256

    print(
        "\nRunning Refactored "
        "FlashAttention-Style Attention"
    )

    print(f"Sequence Length : {seq_len}")
    print(f"Embedding Dim   : {d_model}")
    print(f"Tile Size       : {tile_size}")
    print(f"Device          : {device}\n")

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

    attention = FlashAttention(
        d_model=d_model,
        tile_size=tile_size
    )

    # ---------------------------------------------------
    # GPU memory tracking
    # ---------------------------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------
    # Forward pass
    # ---------------------------------------------------
    output = attention.forward(
        Q,
        K,
        V
    )

    torch.cuda.synchronize()

    print(
        f"Output Shape : {output.shape}"
    )

    # ---------------------------------------------------
    # Peak GPU memory usage
    # ---------------------------------------------------
    if device == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(
            f"Peak GPU Memory Usage : "
            f"{peak_memory:.2f} MB"
        )