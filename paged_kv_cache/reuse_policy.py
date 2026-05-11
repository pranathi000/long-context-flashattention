# paged_kv_cache/reuse_policy.py

from allocator import KVCacheAllocator
from page_table import PageTable


class KVCacheReuseManager:
    """
    Simulates KV-cache page reuse policy
    for long-context inference serving.

    Handles:
    - request registration
    - page allocation
    - page-table updates
    - request completion
    - memory recycling
    """

    def __init__(
        self,
        num_pages: int,
        tokens_per_page: int,
        d_model: int,
        device: str = "cuda"
    ):

        self.allocator = KVCacheAllocator(
            num_pages=num_pages,
            tokens_per_page=tokens_per_page,
            d_model=d_model,
            device=device
        )

        self.page_table = PageTable(
            tokens_per_page=tokens_per_page
        )

        self.active_requests = set()

    def create_request(
        self,
        request_id: str,
        num_tokens: int
    ):

        # ---------------------------------------------------
        # Allocate physical pages
        # ---------------------------------------------------
        allocated_pages = self.allocator.allocate(
            request_id=request_id,
            num_tokens=num_tokens
        )

        # ---------------------------------------------------
        # Register logical mapping
        # ---------------------------------------------------
        self.page_table.register_request(
            request_id=request_id,
            allocated_pages=allocated_pages
        )

        self.active_requests.add(request_id)

        print(
            f"\nRequest {request_id} created"
        )

        print(
            f"Allocated Pages : "
            f"{allocated_pages}"
        )

    def complete_request(
        self,
        request_id: str
    ):

        if request_id not in self.active_requests:

            print(
                f"Request {request_id} "
                f"not active."
            )

            return

        # ---------------------------------------------------
        # Free allocator pages
        # ---------------------------------------------------
        self.allocator.free(request_id)

        # ---------------------------------------------------
        # Remove page-table mappings
        # ---------------------------------------------------
        self.page_table.free_request(
            request_id
        )

        self.active_requests.remove(
            request_id
        )

        print(
            f"\nRequest {request_id} completed"
        )

    def display_memory_state(self):

        print("\n========== MEMORY STATE ==========")

        print(
            f"Active Requests : "
            f"{list(self.active_requests)}"
        )

        print(
            f"Allocated Pages : "
            f"{self.allocator.get_allocated_page_count()}"
        )

        print(
            f"Free Pages : "
            f"{self.allocator.get_free_page_count()}"
        )

        print(
            f"Free Page Pool : "
            f"{self.allocator.free_pages}"
        )

        print("==================================")

    def translate_token(
        self,
        request_id: str,
        token_position: int
    ):

        translation = (
            self.page_table.translate_token(
                request_id=request_id,
                token_position=token_position
            )
        )

        print(
            f"\nToken {token_position}"
            f" -> "
            f"Physical Page "
            f"{translation['physical_page_id']}, "
            f"Offset {translation['offset']}"
        )


if __name__ == "__main__":

    print("\nRunning KV-Cache Reuse Policy Demo")

    reuse_manager = KVCacheReuseManager(
        num_pages=16,
        tokens_per_page=256,
        d_model=128,
        device="cuda"
    )

    # ---------------------------------------------------
    # Create requests
    # ---------------------------------------------------
    reuse_manager.create_request(
        request_id="request_A",
        num_tokens=600
    )

    reuse_manager.create_request(
        request_id="request_B",
        num_tokens=900
    )

    reuse_manager.display_memory_state()

    # ---------------------------------------------------
    # Token translation
    # ---------------------------------------------------
    reuse_manager.translate_token(
        request_id="request_B",
        token_position=700
    )

    # ---------------------------------------------------
    # Complete request A
    # ---------------------------------------------------
    reuse_manager.complete_request(
        "request_A"
    )

    reuse_manager.display_memory_state()

    # ---------------------------------------------------
    # New request reuses freed pages
    # ---------------------------------------------------
    reuse_manager.create_request(
        request_id="request_C",
        num_tokens=500
    )

    reuse_manager.display_memory_state()