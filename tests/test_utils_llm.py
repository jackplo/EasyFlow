# tests/test_utils_llm.py
import unittest
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from easyflow.utils import llm


def create_mock_provider(name):
    """Creates a simple mock provider for testing"""
    return lambda prompt, model, **kwargs: f"{name}:{model}:{prompt}"


class TestLLMBasicFunctionality(unittest.TestCase):
    """Test basic registration and calling functionality"""

    def setUp(self):
        """Clear module state before each test"""
        llm._providers.clear()
        llm._default = None

    def test_register_single_provider(self):
        """Test registering a single provider"""
        llm.register("openai", create_mock_provider("openai"))
        self.assertEqual(len(llm._providers), 1)
        self.assertIn("openai", llm._providers)

    def test_register_sets_default_first_time(self):
        """Test that first registered provider becomes default"""
        llm.register("openai", create_mock_provider("openai"))
        self.assertEqual(llm._default, "openai")

    def test_register_multiple_providers(self):
        """Test registering multiple providers"""
        llm.register("openai", create_mock_provider("openai"))
        llm.register("anthropic", create_mock_provider("anthropic"))
        self.assertEqual(len(llm._providers), 2)
        self.assertEqual(llm._default, "openai")  # First one remains default

    def test_call_with_provider_model_format(self):
        """Test calling with 'provider/model' format"""
        llm.register("openai", create_mock_provider("openai"))
        result = llm.call_llm("test prompt", "openai/gpt-4")
        self.assertEqual(result, "openai:gpt-4:test prompt")

    def test_call_with_default_provider(self):
        """Test calling with default provider (no provider specified)"""
        llm.register("openai", create_mock_provider("openai"))
        result = llm.call_llm("test prompt", "gpt-4")
        self.assertEqual(result, "openai:gpt-4:test prompt")

    def test_call_with_model_only(self):
        """Test calling with model name only, using default provider"""
        llm.register("anthropic", create_mock_provider("anthropic"))
        result = llm.call_llm("hello", "claude")
        self.assertEqual(result, "anthropic:claude:hello")


class TestLLMErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        """Clear module state before each test"""
        llm._providers.clear()
        llm._default = None

    def test_call_unregistered_provider_raises_error(self):
        """Test that calling unregistered provider raises ValueError"""
        llm.register("openai", create_mock_provider("openai"))
        with self.assertRaises(ValueError) as cm:
            llm.call_llm("test", "anthropic/claude")
        self.assertIn("anthropic", str(cm.exception))
        self.assertIn("not registered", str(cm.exception))
        self.assertIn("openai", str(cm.exception))  # Should list available

    def test_call_no_providers_raises_error(self):
        """Test calling when no providers are registered"""
        with self.assertRaises(ValueError) as cm:
            llm.call_llm("test", "model")
        self.assertIn("No providers registered", str(cm.exception))

    def test_call_no_default_provider_raises_error(self):
        """Test calling without provider when no default is set"""
        # This case shouldn't happen in practice, but let's test it
        # by manually clearing default after registration
        llm.register("openai", create_mock_provider("openai"))
        llm._default = None
        with self.assertRaises(ValueError) as cm:
            llm.call_llm("test", "model")
        self.assertIn("No default provider", str(cm.exception))

    def test_register_empty_name_raises_error(self):
        """Test that empty provider name raises ValueError"""
        with self.assertRaises(ValueError) as cm:
            llm.register("", create_mock_provider("test"))
        self.assertIn("non-empty string", str(cm.exception))

    def test_register_name_with_slash_raises_error(self):
        """Test that provider name with '/' raises ValueError"""
        with self.assertRaises(ValueError) as cm:
            llm.register("my/provider", create_mock_provider("test"))
        self.assertIn("cannot contain '/'", str(cm.exception))

    def test_register_non_callable_raises_error(self):
        """Test that non-callable provider raises TypeError"""
        with self.assertRaises(TypeError) as cm:
            llm.register("openai", "not_a_function")
        self.assertIn("must be callable", str(cm.exception))

    def test_call_empty_provider_in_model_raises_error(self):
        """Test that '/model' format raises ValueError"""
        llm.register("openai", create_mock_provider("openai"))
        with self.assertRaises(ValueError) as cm:
            llm.call_llm("test", "/gpt-4")
        self.assertIn("provider part is empty", str(cm.exception))

    def test_call_non_string_prompt_raises_error(self):
        """Test that non-string prompt raises TypeError"""
        llm.register("openai", create_mock_provider("openai"))
        with self.assertRaises(TypeError) as cm:
            llm.call_llm(123, "model")
        self.assertIn("must be a string", str(cm.exception))


class TestLLMThreadSafety(unittest.TestCase):
    """Test thread safety of concurrent calls"""

    def setUp(self):
        """Clear module state and register test providers"""
        llm._providers.clear()
        llm._default = None
        # Register providers BEFORE threads start (mimics real usage)
        llm.register("test", create_mock_provider("test"))
        llm.register("openai", create_mock_provider("openai"))
        llm.register("anthropic", create_mock_provider("anthropic"))

    def test_concurrent_calls_same_provider(self):
        """Test multiple threads calling the same provider simultaneously"""
        results = []
        errors = []

        def call_provider(prompt):
            try:
                result = llm.call_llm(prompt, "test/model")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_provider, args=(f"prompt_{i}",))
                   for i in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All calls should succeed
        self.assertEqual(len(results), 50)
        self.assertEqual(len(errors), 0)

    def test_concurrent_calls_different_providers(self):
        """Test multiple threads calling different providers simultaneously"""
        results = []

        def call_provider(provider_model, prompt):
            result = llm.call_llm(prompt, provider_model)
            results.append(result)

        threads = []
        for i in range(25):
            threads.append(threading.Thread(target=call_provider, args=("openai/gpt-4", f"p{i}")))
            threads.append(threading.Thread(target=call_provider, args=("anthropic/claude", f"p{i}")))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 50)
        # Verify we got results from both providers
        openai_results = [r for r in results if r.startswith("openai:")]
        anthropic_results = [r for r in results if r.startswith("anthropic:")]
        self.assertEqual(len(openai_results), 25)
        self.assertEqual(len(anthropic_results), 25)

    def test_concurrent_calls_with_default_provider(self):
        """Test multiple threads using default provider simultaneously"""
        results = []

        def call_provider(prompt):
            # No provider specified, should use default (test)
            result = llm.call_llm(prompt, "model")
            results.append(result)

        threads = [threading.Thread(target=call_provider, args=(f"prompt_{i}",))
                   for i in range(30)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 30)
        # All should use the default provider (test)
        for result in results:
            self.assertTrue(result.startswith("test:"))

    def test_provider_function_isolation(self):
        """Test that provider functions execute independently without lock contention"""
        import time
        call_times = []

        def slow_provider(prompt, model, **kwargs):
            """Simulates a slow provider (like a real LLM API call)"""
            time.sleep(0.01)  # 10ms delay
            return f"slow:{model}:{prompt}"

        llm.register("slow", slow_provider)

        def timed_call(prompt):
            start = time.time()
            result = llm.call_llm(prompt, "slow/model")
            duration = time.time() - start
            call_times.append(duration)

        threads = [threading.Thread(target=timed_call, args=(f"p{i}",))
                   for i in range(10)]

        overall_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        overall_duration = time.time() - overall_start

        # All calls completed
        self.assertEqual(len(call_times), 10)

        # Overall time should be ~10ms (parallel), not ~100ms (serial)
        # Allow some overhead, but should be well under 50ms
        self.assertLess(overall_duration, 0.05)


if __name__ == '__main__':
    unittest.main()
