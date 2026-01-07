from typing import Callable, Dict, Optional, Any

ProviderFn = Callable[[str, Optional[str], Any], str]

_providers: Dict[str, ProviderFn]
_default: Optional[str]

def register(name: str, call_fn: ProviderFn) -> None:
    """
    Register a provider's call function.
    
    The function should accept: (prompt: str, model: str | None, **kwargs) -> str
    """
    ...

def call_llm(prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
    """
    Call an LLM. Model format: "provider/model-name"
    
    Examples:
        call_llm("Hello", "openai/gpt-4o")
        call_llm("Hello", "anthropic/claude-sonnet-4-0")
    """
    ...