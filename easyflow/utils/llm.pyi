from typing import Callable, Dict, Optional, Any
import threading

ProviderFn = Callable[[str, Optional[str], Any], str]

_providers: Dict[str, ProviderFn]
_default: Optional[str]
_lock: threading.Lock

def register(name: str, call_fn: ProviderFn) -> None:
    """
    Register a provider's call function.

    The function should accept: (prompt: str, model: str | None, **kwargs) -> str

    Must be called during initialization before any flows execute.
    Not thread-safe - do not call concurrently.

    Args:
        name: Provider name (e.g., "openai", "anthropic"). Cannot be empty or contain '/'.
        call_fn: Callable function that accepts (prompt, model, **kwargs) and returns str.

    Raises:
        ValueError: If name is empty, contains '/', or has whitespace.
        TypeError: If call_fn is not callable.
    """
    ...

def call_llm(prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
    """
    Call an LLM. Model format: "provider/model-name"

    Thread-safe for concurrent calls.

    Examples:
        call_llm("Hello", "openai/gpt-4o")
        call_llm("Hello", "anthropic/claude-sonnet-4-0")
        call_llm("Hello", "gpt-4o")  # Uses default provider

    Args:
        prompt: The prompt string to send to the LLM.
        model: Model specifier in "provider/model" format, or just "model" to use default provider.
        **kwargs: Additional arguments passed to the provider function.

    Returns:
        String response from the LLM.

    Raises:
        ValueError: If provider not registered or no default provider set.
        TypeError: If prompt is not a string.
    """
    ...