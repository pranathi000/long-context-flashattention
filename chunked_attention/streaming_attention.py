# chunked_attention/streaming_attention.py

import sys
import math
import torch

# -------------------------------------------------------
# Import paths
# -------------------------------------------------------
sys.path.append("./paged_kv_cache")

from reuse_policy import KVCacheReuseManager


class StreamingFlashAttention:
    """
    Exact streaming FlashAttention-style attention.

    Implements:
    - tiled KV processing
    - online softmax accumulation
    - running max stabilization
    - running sum normalization

    This preserves:
    - exact global softmax correctness
    while keeping memory bounded.
    """

    def __init__(
        self,
        d_model: int,
        tile_size: int,
        tokens_per_page: int,
        num_pages: int,
        device: str = "cuda"
    ):

        self.d_model = d_model

        self.tile_size = tile_size

        self.device = device

        # ---------------------------------------------------
        # KV-cache manager
        # ---------------------------------------------------
        self.kv_manager = KVCacheReuseManager(
            num_pages=num_pages,
            tokens_per_page=tokens_per_page,
            d_model=d_model,
            device=device
        )

    def forward(
        self,
        request_id: str,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ):

        seq_len = Q.shape[0]

        scale = 1.0 / math.sqrt(
            self.d_model
        )

        # ---------------------------------------------------
        # Allocate paged KV-cache
        # ---------------------------------------------------
        self.kv_manager.create_request(
            request_id=request_id,
            num_tokens=seq_len
        )

        print(
            f"\nStreaming Attention Execution "
            f"for sequence length {seq_len}"
        )

        # ---------------------------------------------------
        # Final output tensor
        # ---------------------------------------------------
        output = torch.zeros_like(Q)

        # ===================================================
        # Process query chunks
        # ===================================================
        for q_start in range(
            0,
            seq_len,
            self.tile_size
        ):

            q_end = min(
                q_start + self.tile_size,
                seq_len
            )

            print(
                f"\nProcessing Query Tile : "
                f"{q_start} -> {q_end}"
            )

            Q_tile = Q[q_start:q_end]

            q_tile_size = Q_tile.shape[0]

            # ---------------------------------------------------
            # Running statistics
            # ---------------------------------------------------
            running_max = torch.full(
                (q_tile_size, 1),
                float("-inf"),
                device=self.device
            )

            running_sum = torch.zeros(
                (q_tile_size, 1),
                device=self.device
            )

            running_output = torch.zeros(
                (q_tile_size, self.d_model),
                device=self.device
            )

            # ===================================================
            # Stream over KV tiles
            # ===================================================
            for kv_start in range(
                0,
                seq_len,
                self.tile_size
            ):

                kv_end = min(
                    kv_start + self.tile_size,
                    seq_len
                )

                K_tile = K[
                    kv_start:kv_end
                ]

                V_tile = V[
                    kv_start:kv_end
                ]

                # ---------------------------------------------------
                # Attention scores
                # ---------------------------------------------------
                scores = torch.matmul(
                    Q_tile,
                    K_tile.T
                ) * scale

                # ---------------------------------------------------
                # Tile max
                # ---------------------------------------------------
                tile_max = torch.max(
                    scores,
                    dim=1,
                    keepdim=True
                )[0]

                # ---------------------------------------------------
                # Updated running max
                # ---------------------------------------------------
                new_running_max = torch.maximum(
                    running_max,
                    tile_max
                )

                # ---------------------------------------------------
                # Rescale old sums
                # ---------------------------------------------------
                old_scale = torch.exp(
                    running_max - new_running_max
                )

                # ---------------------------------------------------
                # New exponentials
                # ---------------------------------------------------
                exp_scores = torch.exp(
                    scores - new_running_max
                )

                # ---------------------------------------------------
                # Updated running sum
                # ---------------------------------------------------
                running_sum = (
                    old_scale * running_sum
                    + torch.sum(
                        exp_scores,
                        dim=1,
                        keepdim=True
                    )
                )

                # ---------------------------------------------------
                # Rescale old outputs
                # ---------------------------------------------------
                running_output = (
                    old_scale * running_output
                )

                # ---------------------------------------------------
                # Add current tile contribution
                # ---------------------------------------------------
                running_output += torch.matmul(
                    exp_scores,
                    V_tile
                )

                # ---------------------------------------------------
                # Update running max
                # ---------------------------------------------------
                running_max = new_running_max

            # ===================================================
            # Final normalization
            # ===================================================
            output[q_start:q_end] = (
                running_output / running_sum
            )

        # ---------------------------------------------------
        # Free KV-cache pages
        # ---------------------------------------------------
        self.kv_manager.complete_request(
            request_id
        )

        return output


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nRunning Exact Streaming FlashAttention"
    )

    print(f"Device : {device}\n")

    # ---------------------------------------------------
    # Configuration
    # ---------------------------------------------------
    seq_len = 4096

    d_model = 64

    tile_size = 128

    num_pages = 64

    tokens_per_page = 256

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

    # ---------------------------------------------------
    # Attention engine
    # ---------------------------------------------------
    attention = StreamingFlashAttention(
        d_model=d_model,
        tile_size=tile_size,
        tokens_per_page=tokens_per_page,
        num_pages=num_pages,
        device=device
    )

    # ---------------------------------------------------
    # Memory tracking
    # ---------------------------------------------------
    if device == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------
    # Execute streaming attention
    # ---------------------------------------------------
    output = attention.forward(
        request_id="streaming_request",
        Q=Q,
        K=K,
        V=V
    )

    if device == "cuda":
        torch.cuda.synchronize()

    print(
        f"\nFinal Output Shape : "
        f"{output.shape}"
    )

    # ---------------------------------------------------
    # Peak GPU memory
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