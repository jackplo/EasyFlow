_providers = {}
_default = None

def register(name, call_fn):
    global _default
    _providers[name] = call_fn
    if _default is None:
        _default = name

def call_llm(prompt, model=None, **kwargs):
    global _default
    if model and "/" in model:
        provider, model_name = model.split("/", 1)
    else:
        provider = _default
        model_name = model
    if provider is None or provider not in _providers:
        raise ValueError(f"Provider '{provider}' not registered. Available: {list(_providers.keys())}")
    return _providers[provider](prompt, model_name, **kwargs)