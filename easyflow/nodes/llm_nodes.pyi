from typing import Any, Dict, Optional, Set, Union
from easyflow import Node

class LLMNode(Node[Dict[str, str], str, None]):
    """
    Node that calls an LLM with a configurable prompt template.

    The prompt template can contain {placeholders} that will be filled from the shared store.
    For example: "Summarize this: {document}" will look up shared["document"].

    Example:
        node = LLMNode(
            input_key="document",
            output_key="summary",
            prompt_template="Summarize this in 3 sentences:\\n\\n{document}",
            model="openai/gpt-4o"
        )
    """

    input_key: str
    output_key: str
    prompt_template: str
    model: Optional[str]
    llm_kwargs: Dict[str, Any]
    _template_keys: Set[str]

    def __init__(
        self,
        input_key: str = "input",
        output_key: str = "output",
        prompt_template: str = "{input}",
        model: Optional[str] = None,
        max_retries: int = 3,
        wait: Union[int, float] = 1,
        **llm_kwargs: Any
    ) -> None:
        """
        Initialize an LLMNode.

        Args:
            input_key: Key to read primary input from shared store (default: "input").
                       This is also available as {input} in the template.
            output_key: Key to write LLM response to shared store (default: "output").
            prompt_template: Template string with {placeholders} for values from shared.
                            Default is "{input}" which just passes the input directly.
            model: Model identifier (e.g., "openai/gpt-4o"). None uses default provider.
            max_retries: Number of retry attempts on failure (default: 3).
            wait: Seconds to wait between retries (default: 1).
            **llm_kwargs: Additional arguments passed to call_llm().
        """
        ...

    def _extract_template_keys(self) -> Set[str]:
        """Extract all {placeholder} keys from the prompt template."""
        ...

    def prep(self, shared: Dict[str, Any]) -> Dict[str, str]:
        """Gather all values needed for the prompt template from shared store."""
        ...

    def exec(self, context: Dict[str, str]) -> str:
        """Build prompt from template and call LLM."""
        ...

    def post(
        self,
        shared: Dict[str, Any],
        prep_res: Dict[str, str],
        exec_res: str
    ) -> None:
        """Store LLM response in shared store."""
        ...
