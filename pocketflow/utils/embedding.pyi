from typing import Callable, List, Sequence, Optional, Any

EmbedFn = Callable[[str, Optional[str], Any], Sequence[float]]

_providers: dict[str, EmbedFn]
_default: Optional[str]

def register(name: str, embed_fn: EmbedFn) -> None:
    """
    Register an embedding provider's function.
    
    The function should accept: (text: str, model: str | None, **kwargs) -> Sequence[float]
    """
    ...

def embed(text: str, model: Optional[str] = None, **kwargs: Any) -> Sequence[float]:
    """
    Embed text. Model format: "provider/model-name"
    
    Examples:
        embed("Hello world", "openai/text-embedding-3-small")
        embed("Hello world", "cohere/embed-english-v3.0")
    """
    ...