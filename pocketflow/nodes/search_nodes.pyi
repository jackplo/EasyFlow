from typing import Any, Dict, List, Optional, Union
from pocketflow import Node

SearchResult = Dict[str, str]  # {"title": str, "snippet": str, "url": str}

class SearchNode(Node[str, Union[List[SearchResult], str], None]):
    """
    Node that performs web search using a registered search provider.

    Example:
        node = SearchNode(
            input_key="user_query",
            output_key="results",
            provider="duckduckgo",
            num_results=10
        )
    """

    input_key: str
    output_key: str
    provider: Optional[str]
    num_results: int
    format_results: bool
    search_kwargs: Dict[str, Any]

    def __init__(
        self,
        input_key: str = "query",
        output_key: str = "search_results",
        provider: Optional[str] = None,
        num_results: int = 5,
        format_results: bool = False,
        max_retries: int = 3,
        wait: Union[int, float] = 1,
        **search_kwargs: Any
    ) -> None:
        """
        Initialize a SearchNode.

        Args:
            input_key: Key to read search query from shared store (default: "query").
            output_key: Key to write search results to shared store (default: "search_results").
            provider: Search provider name (e.g., "duckduckgo"). None uses default.
            num_results: Number of search results to return (default: 5).
            format_results: If True, formats results as readable string. If False, returns raw list.
                           (default: False)
            max_retries: Number of retry attempts on failure (default: 3).
            wait: Seconds to wait between retries (default: 1).
            **search_kwargs: Additional arguments passed to web_search().
        """
        ...

    def prep(self, shared: Dict[str, Any]) -> str:
        """Read search query from shared store."""
        ...

    def exec(self, query: str) -> Union[List[SearchResult], str]:
        """Execute search and optionally format results."""
        ...

    def _format_results(self, results: List[SearchResult]) -> str:
        """Format search results as a readable string."""
        ...

    def post(
        self,
        shared: Dict[str, Any],
        prep_res: str,
        exec_res: Union[List[SearchResult], str]
    ) -> None:
        """Store search results in shared store."""
        ...
