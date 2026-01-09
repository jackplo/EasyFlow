from typing import Callable, List, Sequence, Optional, Any
import threading

EmbedFn = Callable[[str, Optional[str], Any], Sequence[float]]

_providers: dict[str, EmbedFn]
_default: Optional[str]
_lock: threading.Lock

def register(name: str, embed_fn: EmbedFn) -> None:
    """
    Register an embedding provider's function.

    The function should accept: (text: str, model: str | None, **kwargs) -> Sequence[float]

    Must be called during initialization before any flows execute.
    Not thread-safe - do not call concurrently.

    Args:
        name: Provider name (e.g., "openai", "cohere"). Cannot be empty or contain '/'.
        embed_fn: Callable function that accepts (text, model, **kwargs) and returns Sequence[float].

    Raises:
        ValueError: If name is empty, contains '/', or has whitespace.
        TypeError: If embed_fn is not callable.
    """
    ...

def embed(text: str, model: Optional[str] = None, **kwargs: Any) -> Sequence[float]:
    """
    Embed text. Model format: "provider/model-name"

    Thread-safe for concurrent calls.

    Examples:
        embed("Hello world", "openai/text-embedding-3-small")
        embed("Hello world", "cohere/embed-english-v3.0")
        embed("Hello world", "text-embedding-3-small")  # Uses default provider

    Args:
        text: The text string to embed.
        model: Model specifier in "provider/model" format, or just "model" to use default provider.
        **kwargs: Additional arguments passed to the provider function.

    Returns:
        Sequence of floats representing the embedding vector.

    Raises:
        ValueError: If provider not registered or no default provider set.
        TypeError: If text is not a string.
    """
    ...