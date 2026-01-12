from typing import Callable, Dict, List, Optional, Any
import threading

SearchResult = Dict[str, str]  # {"title": str, "snippet": str, "url": str}
SearchFn = Callable[[str, int, Any], List[SearchResult]]

_providers: Dict[str, SearchFn]
_default: Optional[str]
_lock: threading.Lock

def register(name: str, search_fn: SearchFn) -> None:
    """
    Register a search provider's function.

    The function should accept: (query: str, num_results: int, **kwargs) -> List[Dict]
    Expected return format: [{"title": str, "snippet": str, "url": str}, ...]

    Must be called during initialization before any flows execute.
    Not thread-safe - do not call concurrently.

    Args:
        name: Provider name (e.g., "duckduckgo", "brave"). Cannot be empty or contain '/'.
        search_fn: Callable function that accepts (query, num_results, **kwargs) and returns List[Dict].

    Raises:
        ValueError: If name is empty, contains '/', or has whitespace.
        TypeError: If search_fn is not callable.
    """
    ...

def web_search(
    query: str,
    provider: Optional[str] = None,
    num_results: int = 5,
    **kwargs: Any
) -> List[SearchResult]:
    """
    Execute a web search.

    Thread-safe for concurrent calls.

    Examples:
        web_search("python tutorial")
        web_search("python tutorial", provider="duckduckgo")
        web_search("python tutorial", num_results=10)

    Args:
        query: The search query string.
        provider: Provider name (e.g., "duckduckgo"). None uses default provider.
        num_results: Number of results to return (default: 5).
        **kwargs: Additional arguments passed to the provider function.

    Returns:
        List of dicts with keys: title, snippet, url.

    Raises:
        ValueError: If provider not registered or no default provider set.
        TypeError: If query is not a string.
    """
    ...
