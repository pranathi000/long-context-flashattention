# paged_kv_cache/allocator.py

import torch


class KVCacheAllocator:
    """
    Simple page-based KV-cache allocator.

    Simulates vLLM-style paged KV-cache allocation
    for long-context transformer inference.
    """

    def __init__(
        self,
        num_pages: int,
        tokens_per_page: int,
        d_model: int,
        device: str = "cuda"
    ):

        self.num_pages = num_pages
        self.tokens_per_page = tokens_per_page
        self.d_model = d_model
        self.device = device

        # ---------------------------------------------------
        # Physical KV-cache pages
        # ---------------------------------------------------
        self.pages = []

        for _ in range(num_pages):

            page = torch.zeros(
                (tokens_per_page, d_model),
                device=device
            )

            self.pages.append(page)

        # ---------------------------------------------------
        # Free page tracking
        # ---------------------------------------------------
        self.free_pages = list(range(num_pages))

        # ---------------------------------------------------
        # Allocated page mapping
        # request_id -> allocated pages
        # ---------------------------------------------------
        self.allocations = {}

    def allocate(self, request_id: str, num_tokens: int):

        # ---------------------------------------------------
        # Calculate required pages
        # ---------------------------------------------------
        required_pages = (
            num_tokens + self.tokens_per_page - 1
        ) // self.tokens_per_page

        # ---------------------------------------------------
        # Check available memory
        # ---------------------------------------------------
        if required_pages > len(self.free_pages):

            raise RuntimeError(
                "Insufficient KV-cache pages available."
            )

        # ---------------------------------------------------
        # Allocate pages
        # ---------------------------------------------------
        allocated = []

        for _ in range(required_pages):

            page_id = self.free_pages.pop(0)

            allocated.append(page_id)

        # ---------------------------------------------------
        # Store allocation
        # ---------------------------------------------------
        self.allocations[request_id] = allocated

        return allocated

    def free(self, request_id: str):

        if request_id not in self.allocations:
            return

        released_pages = self.allocations.pop(request_id)

        # ---------------------------------------------------
        # Return pages to free pool
        # ---------------------------------------------------
        self.free_pages.extend(released_pages)

    def get_free_page_count(self):

        return len(self.free_pages)

    def get_allocated_page_count(self):

        return (
            self.num_pages
            - len(self.free_pages)
        )


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nRunning KV-Cache Allocator Demo")
    print(f"Device : {device}\n")

    allocator = KVCacheAllocator(
        num_pages=16,
        tokens_per_page=256,
        d_model=128,
        device=device
    )

    # ---------------------------------------------------
    # Simulated requests
    # ---------------------------------------------------
    request_a = allocator.allocate(
        request_id="request_A",
        num_tokens=600
    )

    request_b = allocator.allocate(
        request_id="request_B",
        num_tokens=900
    )

    print(f"Request A Pages : {request_a}")
    print(f"Request B Pages : {request_b}")

    print(
        f"\nAllocated Pages : "
        f"{allocator.get_allocated_page_count()}"
    )

    print(
        f"Free Pages : "
        f"{allocator.get_free_page_count()}"
    )

    # ---------------------------------------------------
    # Free request A
    # ---------------------------------------------------
    allocator.free("request_A")

    print("\nAfter Freeing Request A")

    print(
        f"Allocated Pages : "
        f"{allocator.get_allocated_page_count()}"
    )

    print(
        f"Free Pages : "
        f"{allocator.get_free_page_count()}"
    )