# tests/test_nodes_llm.py
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from easyflow.utils import llm as llm_module
from easyflow.nodes import LLMNode


def create_mock_llm(name):
    """Creates a mock LLM provider that echoes back the prompt"""
    return lambda prompt, model, **kwargs: f"[{name}:{model}] {prompt}"


class TestLLMNodeBasic(unittest.TestCase):
    """Test basic LLMNode functionality"""

    def setUp(self):
        """Clear LLM module state and register mock provider"""
        llm_module._providers.clear()
        llm_module._default = None
        llm_module.register("mock", create_mock_llm("mock"))

    def test_simple_passthrough(self):
        """Test basic input -> output passthrough"""
        node = LLMNode(input_key="input", output_key="output")
        shared = {"input": "Hello, world!"}
        node.run(shared)
        self.assertIn("[mock:None] Hello, world!", shared["output"])

    def test_custom_keys(self):
        """Test custom input/output keys"""
        node = LLMNode(input_key="question", output_key="answer")
        shared = {"question": "What is 2+2?"}
        node.run(shared)
        self.assertIn("What is 2+2?", shared["answer"])
        self.assertNotIn("output", shared)

    def test_prompt_template(self):
        """Test prompt template with placeholders"""
        node = LLMNode(
            input_key="document",
            output_key="summary",
            prompt_template="Summarize: {document}"
        )
        shared = {"document": "Long text here..."}
        node.run(shared)
        self.assertIn("Summarize: Long text here...", shared["summary"])

    def test_template_multiple_placeholders(self):
        """Test template with multiple placeholders"""
        node = LLMNode(
            prompt_template="Question: {question}\nContext: {context}",
            output_key="answer"
        )
        shared = {"question": "What color?", "context": "The sky is blue."}
        node.run(shared)
        self.assertIn("Question: What color?", shared["answer"])
        self.assertIn("Context: The sky is blue.", shared["answer"])

    def test_template_missing_placeholder_uses_empty_string(self):
        """Test that missing placeholders default to empty string"""
        node = LLMNode(
            prompt_template="Name: {name}, Age: {age}",
            output_key="result"
        )
        shared = {"name": "Alice"}  # age is missing
        node.run(shared)
        self.assertIn("Name: Alice, Age:", shared["result"])

    def test_model_specification(self):
        """Test that model is passed to call_llm"""
        node = LLMNode(
            input_key="input",
            output_key="output",
            model="mock/gpt-4"
        )
        shared = {"input": "test"}
        node.run(shared)
        self.assertIn("[mock:gpt-4]", shared["output"])


class TestLLMNodeConfiguration(unittest.TestCase):
    """Test LLMNode configuration options"""

    def setUp(self):
        """Clear LLM module state and register mock provider"""
        llm_module._providers.clear()
        llm_module._default = None
        llm_module.register("mock", create_mock_llm("mock"))

    def test_default_values(self):
        """Test default configuration values"""
        node = LLMNode()
        self.assertEqual(node.input_key, "input")
        self.assertEqual(node.output_key, "output")
        self.assertEqual(node.prompt_template, "{input}")
        self.assertIsNone(node.model)
        self.assertEqual(node.max_retries, 3)
        self.assertEqual(node.wait, 1)

    def test_retry_configuration(self):
        """Test max_retries and wait configuration"""
        node = LLMNode(max_retries=5, wait=2)
        self.assertEqual(node.max_retries, 5)
        self.assertEqual(node.wait, 2)

    def test_llm_kwargs_passed_through(self):
        """Test that extra kwargs are passed to call_llm"""
        received_kwargs = {}

        def capturing_llm(prompt, model, **kwargs):
            received_kwargs.update(kwargs)
            return "response"

        llm_module._providers["mock"] = capturing_llm

        node = LLMNode(input_key="input", temperature=0.7, top_p=0.9)
        shared = {"input": "test"}
        node.run(shared)

        self.assertEqual(received_kwargs.get("temperature"), 0.7)
        self.assertEqual(received_kwargs.get("top_p"), 0.9)


class TestLLMNodeTemplateExtraction(unittest.TestCase):
    """Test template key extraction"""

    def test_extract_simple_keys(self):
        """Test extraction of simple template keys"""
        node = LLMNode(prompt_template="{input}")
        self.assertEqual(node._template_keys, {"input"})

    def test_extract_multiple_keys(self):
        """Test extraction of multiple template keys"""
        node = LLMNode(prompt_template="{a} and {b} and {c}")
        self.assertEqual(node._template_keys, {"a", "b", "c"})

    def test_extract_duplicate_keys(self):
        """Test that duplicate keys are deduplicated"""
        node = LLMNode(prompt_template="{x} {x} {y}")
        self.assertEqual(node._template_keys, {"x", "y"})

    def test_no_keys(self):
        """Test template with no placeholders"""
        node = LLMNode(prompt_template="Static prompt")
        self.assertEqual(node._template_keys, set())


if __name__ == '__main__':
    unittest.main()
