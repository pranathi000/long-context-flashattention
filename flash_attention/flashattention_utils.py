# flashattention/flashattention_utils.py

import torch


def update_running_max(
    previous_max: torch.Tensor,
    current_scores: torch.Tensor
):
    """
    FlashAttention-style streaming max update.

    Maintains numerically stable softmax execution
    across attention tiles.
    """

    # ---------------------------------------------------
    # Maximum value inside current tile
    # ---------------------------------------------------
    current_tile_max = torch.max(
        current_scores,
        dim=-1,
        keepdim=True
    ).values

    # ---------------------------------------------------
    # Updated running max
    # ---------------------------------------------------
    new_running_max = torch.maximum(
        previous_max,
        current_tile_max
    )

    return current_tile_max, new_running_max


def update_running_sum(
    previous_sum: torch.Tensor,
    previous_max: torch.Tensor,
    new_max: torch.Tensor,
    current_scores: torch.Tensor
):
    """
    FlashAttention-style streaming softmax
    denominator accumulation.
    """

    # ---------------------------------------------------
    # Rescale previous accumulated sum
    # ---------------------------------------------------
    previous_scale = torch.exp(
        previous_max - new_max
    )

    # ---------------------------------------------------
    # Current tile exponentials
    # ---------------------------------------------------
    current_exp = torch.exp(
        current_scores - new_max
    )

    # ---------------------------------------------------
    # Current tile contribution
    # ---------------------------------------------------
    current_sum = torch.sum(
        current_exp,
        dim=-1,
        keepdim=True
    )

    # ---------------------------------------------------
    # Updated running sum
    # ---------------------------------------------------
    updated_sum = (
        previous_sum * previous_scale
        + current_sum
    )

    return current_exp, updated_sum


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nRunning FlashAttention Utilities Demo")
    print(f"Device : {device}\n")

    batch_size = 4
    tile_size = 8

    # ---------------------------------------------------
    # Simulated tile scores
    # ---------------------------------------------------
    scores = torch.randn(
        batch_size,
        tile_size,
        device=device
    )

    # ---------------------------------------------------
    # Initial running statistics
    # ---------------------------------------------------
    previous_max = torch.full(
        (batch_size, 1),
        -float("inf"),
        device=device
    )

    previous_sum = torch.zeros(
        (batch_size, 1),
        device=device
    )

    # ---------------------------------------------------
    # Running max update
    # ---------------------------------------------------
    current_tile_max, new_max = (
        update_running_max(
            previous_max,
            scores
        )
    )

    # ---------------------------------------------------
    # Running sum update
    # ---------------------------------------------------
    current_exp, updated_sum = (
        update_running_sum(
            previous_sum,
            previous_max,
            new_max,
            scores
        )
    )

    print("Current Tile Max:")
    print(current_tile_max)

    print("\nUpdated Running Max:")
    print(new_max)

    print("\nExponentiated Scores:")
    print(current_exp)

    print("\nUpdated Running Sum:")
    print(updated_sum)