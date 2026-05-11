# paged_kv_cache/page_table.py

class PageTable:
    """
    Logical-to-physical KV-cache page mapping.

    Simulates how inference engines like vLLM map
    logical token positions to physical KV pages.
    """

    def __init__(self, tokens_per_page: int):

        self.tokens_per_page = tokens_per_page

        # ---------------------------------------------------
        # request_id -> logical page mappings
        # ---------------------------------------------------
        self.page_tables = {}

    def register_request(
        self,
        request_id: str,
        allocated_pages: list
    ):

        """
        Creates logical-to-physical page mapping.

        Example:

        logical_page_0 -> physical_page_7
        logical_page_1 -> physical_page_2
        """

        logical_mapping = {}

        for logical_page_id, physical_page_id in enumerate(
            allocated_pages
        ):

            logical_mapping[
                logical_page_id
            ] = physical_page_id

        self.page_tables[
            request_id
        ] = logical_mapping

    def translate_token(
        self,
        request_id: str,
        token_position: int
    ):

        """
        Translate logical token position
        into physical page location.
        """

        if request_id not in self.page_tables:

            raise ValueError(
                f"Request {request_id} not found."
            )

        # ---------------------------------------------------
        # Logical page index
        # ---------------------------------------------------
        logical_page_id = (
            token_position // self.tokens_per_page
        )

        # ---------------------------------------------------
        # Offset within page
        # ---------------------------------------------------
        offset = (
            token_position % self.tokens_per_page
        )

        logical_mapping = self.page_tables[
            request_id
        ]

        if logical_page_id not in logical_mapping:

            raise ValueError(
                "Logical page does not exist."
            )

        physical_page_id = logical_mapping[
            logical_page_id
        ]

        return {
            "logical_page_id": logical_page_id,
            "physical_page_id": physical_page_id,
            "offset": offset
        }

    def free_request(
        self,
        request_id: str
    ):

        if request_id in self.page_tables:

            del self.page_tables[request_id]

    def display_page_table(
        self,
        request_id: str
    ):

        if request_id not in self.page_tables:

            print("Request not found.")
            return

        print(
            f"\nPage Table for {request_id}\n"
        )

        mapping = self.page_tables[
            request_id
        ]

        for logical_page, physical_page in mapping.items():

            print(
                f"Logical Page {logical_page}"
                f" -> "
                f"Physical Page {physical_page}"
            )


if __name__ == "__main__":

    print("\nRunning Page Table Demo")

    tokens_per_page = 256

    page_table = PageTable(
        tokens_per_page=tokens_per_page
    )

    # ---------------------------------------------------
    # Simulated allocation
    # ---------------------------------------------------
    allocated_pages = [4, 7, 12, 15]

    request_id = "request_A"

    page_table.register_request(
        request_id=request_id,
        allocated_pages=allocated_pages
    )

    # ---------------------------------------------------
    # Display mapping
    # ---------------------------------------------------
    page_table.display_page_table(
        request_id
    )

    # ---------------------------------------------------
    # Translate token positions
    # ---------------------------------------------------
    token_positions = [
        0,
        128,
        300,
        700,
        900
    ]

    print("\nToken Translation Results\n")

    for token_position in token_positions:

        translation = page_table.translate_token(
            request_id=request_id,
            token_position=token_position
        )

        print(
            f"Token {token_position}"
            f" -> "
            f"Physical Page "
            f"{translation['physical_page_id']}, "
            f"Offset {translation['offset']}"
        )