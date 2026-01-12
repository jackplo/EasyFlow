import re
from pocketflow import Node
from pocketflow.utils import call_llm


class LLMNode(Node):
    """Node that calls an LLM with a configurable prompt template.

    The prompt template can contain {placeholders} that will be filled from the shared store.
    For example: "Summarize this: {document}" will look up shared["document"].

    Args:
        input_key: Key to read primary input from shared store (default: "input").
                   This is also available as {input} in the template.
        output_key: Key to write LLM response to shared store (default: "output")
        prompt_template: Template string with {placeholders} for values from shared.
                        Default is "{input}" which just passes the input directly.
        model: Model identifier (e.g., "openai/gpt-4o"). None uses default provider.
        max_retries: Number of retry attempts on failure (default: 3)
        wait: Seconds to wait between retries (default: 1)
        **llm_kwargs: Additional arguments passed to call_llm()

    Example:
        node = LLMNode(
            input_key="document",
            output_key="summary",
            prompt_template="Summarize this in 3 sentences:\\n\\n{document}",
            model="openai/gpt-4o"
        )
    """

    def __init__(
        self,
        input_key="input",
        output_key="output",
        prompt_template="{input}",
        model=None,
        max_retries=3,
        wait=1,
        **llm_kwargs
    ):
        super().__init__(max_retries=max_retries, wait=wait)
        self.input_key = input_key
        self.output_key = output_key
        self.prompt_template = prompt_template
        self.model = model
        self.llm_kwargs = llm_kwargs
        self._template_keys = self._extract_template_keys()

    def _extract_template_keys(self):
        """Extract all {placeholder} keys from the prompt template."""
        return set(re.findall(r'\{(\w+)\}', self.prompt_template))

    def prep(self, shared):
        """Gather all values needed for the prompt template from shared store."""
        context = {}
        for key in self._template_keys:
            context[key] = shared.get(key, "")
        # Also ensure input_key is available even if not in template
        if self.input_key not in context:
            context[self.input_key] = shared.get(self.input_key, "")
        # Alias input_key value to "input" for default template compatibility
        # This allows LLMNode(input_key="question") to work with default template "{input}"
        if "input" in self._template_keys and self.input_key != "input":
            context["input"] = shared.get(self.input_key, "")
        return context

    def exec(self, context):
        """Build prompt from template and call LLM."""
        prompt = self.prompt_template.format(**context)
        return call_llm(prompt, model=self.model, **self.llm_kwargs)

    def post(self, shared, prep_res, exec_res):
        """Store LLM response in shared store."""
        shared[self.output_key] = exec_res
