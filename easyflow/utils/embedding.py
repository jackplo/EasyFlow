import threading

_providers = {}
_default = None
_lock = threading.Lock()

def register(name, embed_fn):
    """Register an embedding provider. Must be called during initialization before any flows execute."""
    global _default
    if not name or not isinstance(name, str):
        raise ValueError("Provider name must be a non-empty string")
    if name.strip() != name:
        raise ValueError(f"Provider name '{name}' contains leading/trailing whitespace")
    if "/" in name:
        raise ValueError(f"Provider name '{name}' cannot contain '/' character")
    if not callable(embed_fn):
        raise TypeError("Provider function must be callable")
    _providers[name] = embed_fn
    if _default is None:
        _default = name

def embed(text, model=None, **kwargs):
    """Embed text. Thread-safe for concurrent calls."""
    global _default
    if not isinstance(text, str):
        raise TypeError(f"Text must be a string, got {type(text).__name__}")

    if model and "/" in model:
        parts = model.split("/", 1)
        provider, model_name = parts[0], parts[1]
        if not provider:
            raise ValueError(f"Invalid model format '{model}': provider part is empty")
    else:
        provider = None
        model_name = model

    with _lock:
        if provider is None:
            provider = _default

        if provider is None:
            if not _providers:
                raise ValueError("No providers registered. Register a provider first using register()")
            raise ValueError("No default provider set and no provider specified in model")

        if provider not in _providers:
            available = list(_providers.keys())
            raise ValueError(
                f"Provider '{provider}' not registered. "
                f"Available providers: {available}. "
                f"Use register('{provider}', fn) to register it."
            )

        fn = _providers[provider]

    return fn(text, model_name, **kwargs)