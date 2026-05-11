# chunked_attention/chunk_executor.py

import os
import sys
import torch

# -------------------------------------------------------
# Import FlashAttention implementation
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

PAGED_KV_DIR = os.path.join(
    PROJECT_ROOT,
    "paged_kv_cache"
)

sys.path.append(FLASH_ATTENTION_DIR)
sys.path.append(PAGED_KV_DIR)

from tiled_attention import FlashAttention
from reuse_policy import KVCacheReuseManager


class ChunkedAttentionExecutor:
    """
    Simulates long-context transformer inference
    using:

    - chunked execution
    - FlashAttention-style tiled attention
    - paged KV-cache allocation

    Goal:
    keep memory usage bounded while processing
    extremely long sequences.
    """

    def __init__(
        self,
        d_model: int,
        tile_size: int,
        chunk_size: int,
        tokens_per_page: int,
        num_pages: int,
        device: str = "cuda"
    ):

        self.device = device

        self.chunk_size = chunk_size

        # ---------------------------------------------------
        # FlashAttention engine
        # ---------------------------------------------------
        self.attention = FlashAttention(
            d_model=d_model,
            tile_size=tile_size
        )

        # ---------------------------------------------------
        # KV-cache manager
        # ---------------------------------------------------
        self.kv_manager = KVCacheReuseManager(
            num_pages=num_pages,
            tokens_per_page=tokens_per_page,
            d_model=d_model,
            device=device
        )

    def execute(
        self,
        request_id: str,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ):

        seq_len = Q.shape[0]

        outputs = []

        # ---------------------------------------------------
        # Allocate KV-cache pages
        # ---------------------------------------------------
        self.kv_manager.create_request(
            request_id=request_id,
            num_tokens=seq_len
        )

        print(
            f"\nExecuting chunked attention "
            f"for sequence length {seq_len}"
        )

        # ---------------------------------------------------
        # Process sequence chunk-by-chunk
        # ---------------------------------------------------
        for chunk_start in range(
            0,
            seq_len,
            self.chunk_size
        ):

            chunk_end = min(
                chunk_start + self.chunk_size,
                seq_len
            )

            print(
                f"\nProcessing Chunk : "
                f"{chunk_start} -> {chunk_end}"
            )

            Q_chunk = Q[
                chunk_start:chunk_end
            ]

            # ---------------------------------------------------
            # Full KV visibility
            # ---------------------------------------------------
            chunk_output = self.attention.forward(
                Q_chunk,
                K,
                V
            )

            outputs.append(chunk_output)

        # ---------------------------------------------------
        # Concatenate chunk outputs
        # ---------------------------------------------------
        final_output = torch.cat(
            outputs,
            dim=0
        )

        # ---------------------------------------------------
        # Release KV-cache pages
        # ---------------------------------------------------
        self.kv_manager.complete_request(
            request_id
        )

        return final_output


if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nRunning Chunked Long-Context "
        "Attention Executor"
    )

    print(f"Device : {device}\n")

    # ---------------------------------------------------
    # Configuration
    # ---------------------------------------------------
    seq_len = 8192

    d_model = 128

    tile_size = 256

    chunk_size = 2048

    tokens_per_page = 256

    num_pages = 64

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

    executor = ChunkedAttentionExecutor(
        d_model=d_model,
        tile_size=tile_size,
        chunk_size=chunk_size,
        tokens_per_page=tokens_per_page,
        num_pages=num_pages,
        device=device
    )

    # ---------------------------------------------------
    # GPU memory tracking
    # ---------------------------------------------------
    if device == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------
    # Execute chunked inference
    # ---------------------------------------------------
    output = executor.execute(
        request_id="request_long_context",
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
    # Peak memory usage
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