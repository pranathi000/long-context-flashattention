import math
import torch


class NaiveAttention:
    def __init__(self, d_model: int):
        self.d_model = d_model

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Q : [seq_len, d_model]
        K : [seq_len, d_model]
        V : [seq_len, d_model]
        """

        # ---------------------------------------------------
        # Step 1: Compute attention scores
        # QK^T
        # Shape -> [seq_len, seq_len]
        # ---------------------------------------------------
        scores = torch.matmul(Q, K.T)

        # ---------------------------------------------------
        # Step 2: Scale scores
        # ---------------------------------------------------
        scores = scores / math.sqrt(self.d_model)

        # ---------------------------------------------------
        # Step 3: Softmax normalization
        # ---------------------------------------------------
        attention_weights = torch.softmax(scores, dim=-1)

        # ---------------------------------------------------
        # Step 4: Weighted value aggregation
        # ---------------------------------------------------
        output = torch.matmul(attention_weights, V)

        return output


if __name__ == "__main__":

    # ---------------------------------------------------
    # Device setup
    # ---------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------
    # Configuration
    # ---------------------------------------------------
    seq_len = 4096
    d_model = 128

    print(f"\nRunning Naive Attention")
    print(f"Sequence Length : {seq_len}")
    print(f"Embedding Dim   : {d_model}")
    print(f"Device          : {device}\n")

    # ---------------------------------------------------
    # Random QKV tensors
    # ---------------------------------------------------
    Q = torch.randn(seq_len, d_model, device=device)
    K = torch.randn(seq_len, d_model, device=device)
    V = torch.randn(seq_len, d_model, device=device)

    # ---------------------------------------------------
    # Initialize attention
    # ---------------------------------------------------
    attention = NaiveAttention(d_model)

    # ---------------------------------------------------
    # GPU memory tracking
    # ---------------------------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------
    # Forward pass
    # ---------------------------------------------------
    output = attention.forward(Q, K, V)

    # ---------------------------------------------------
    # Output info
    # ---------------------------------------------------
    print(f"Output Shape : {output.shape}")

    # ---------------------------------------------------
    # Peak memory usage
    # ---------------------------------------------------
    if device == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

        print(f"Peak GPU Memory Usage : {peak_memory:.2f} MB")