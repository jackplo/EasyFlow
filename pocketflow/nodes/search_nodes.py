from pocketflow import Node
from pocketflow.utils.search import web_search


class SearchNode(Node):
    """Node that performs web search using a registered search provider.

    Args:
        input_key: Key to read search query from shared store (default: "query")
        output_key: Key to write search results to shared store (default: "search_results")
        provider: Search provider name (e.g., "duckduckgo"). None uses default.
        num_results: Number of search results to return (default: 5)
        format_results: If True, formats results as readable string. If False, returns raw list.
                       (default: False)
        max_retries: Number of retry attempts on failure (default: 3)
        wait: Seconds to wait between retries (default: 1)
        **search_kwargs: Additional arguments passed to search()

    Example:
        node = SearchNode(
            input_key="user_query",
            output_key="results",
            provider="duckduckgo",
            num_results=10
        )
    """

    def __init__(
        self,
        input_key="query",
        output_key="search_results",
        provider=None,
        num_results=5,
        format_results=False,
        max_retries=3,
        wait=1,
        **search_kwargs
    ):
        super().__init__(max_retries=max_retries, wait=wait)
        self.input_key = input_key
        self.output_key = output_key
        self.provider = provider
        self.num_results = num_results
        self.format_results = format_results
        self.search_kwargs = search_kwargs

    def prep(self, shared):
        """Read search query from shared store."""
        return shared.get(self.input_key, "")

    def exec(self, query):
        """Execute search and optionally format results."""
        if not query:
            return [] if not self.format_results else ""

        results = web_search(
            query,
            provider=self.provider,
            num_results=self.num_results,
            **self.search_kwargs
        )

        if self.format_results:
            return self._format_results(results)
        return results

    def _format_results(self, results):
        """Format search results as a readable string."""
        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            snippet = r.get("snippet", "No description")
            url = r.get("url", "")
            formatted.append(f"{i}. {title}\n   {snippet}\n   URL: {url}")

        return "\n\n".join(formatted)

    def post(self, shared, prep_res, exec_res):
        """Store search results in shared store."""
        shared[self.output_key] = exec_res
