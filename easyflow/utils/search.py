import threading

_providers = {}
_default = None
_lock = threading.Lock()


def register(name, search_fn):
    """Register a search provider. Must be called during initialization before any flows execute.

    Provider function signature: fn(query: str, num_results: int, **kwargs) -> List[Dict]
    Expected return format: [{"title": str, "snippet": str, "url": str}, ...]
    """
    global _default
    if not name or not isinstance(name, str):
        raise ValueError("Provider name must be a non-empty string")
    if name.strip() != name:
        raise ValueError(f"Provider name '{name}' contains leading/trailing whitespace")
    if "/" in name:
        raise ValueError(f"Provider name '{name}' cannot contain '/' character")
    if not callable(search_fn):
        raise TypeError("Provider function must be callable")
    _providers[name] = search_fn
    if _default is None:
        _default = name


def web_search(query, provider=None, num_results=5, **kwargs):
    """Execute a web search. Thread-safe for concurrent calls.

    Args:
        query: The search query string
        provider: Provider name (e.g., "duckduckgo") or None for default
        num_results: Number of results to return (default: 5)
        **kwargs: Additional arguments passed to the provider function

    Returns:
        List of dicts with keys: title, snippet, url
    """
    global _default
    if not isinstance(query, str):
        raise TypeError(f"Query must be a string, got {type(query).__name__}")

    with _lock:
        if provider is None:
            provider = _default

        if provider is None:
            if not _providers:
                raise ValueError("No providers registered. Register a provider first using register()")
            raise ValueError("No default provider set and no provider specified")

        if provider not in _providers:
            available = list(_providers.keys())
            raise ValueError(
                f"Provider '{provider}' not registered. "
                f"Available providers: {available}. "
                f"Use register('{provider}', fn) to register it."
            )

        fn = _providers[provider]

    return fn(query, num_results, **kwargs)
