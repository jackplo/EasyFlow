# tests/test_utils_embedding.py
import unittest
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pocketflow.utils import embedding


def create_mock_embedding_provider(name):
    """Creates a simple mock embedding provider for testing"""
    return lambda text, model, **kwargs: [float(ord(c)) for c in f"{name}:{model}:{text}"[:10]]


class TestEmbeddingBasicFunctionality(unittest.TestCase):
    """Test basic registration and embedding functionality"""

    def setUp(self):
        """Clear module state before each test"""
        embedding._providers.clear()
        embedding._default = None

    def test_register_single_provider(self):
        """Test registering a single provider"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        self.assertEqual(len(embedding._providers), 1)
        self.assertIn("openai", embedding._providers)

    def test_register_sets_default_first_time(self):
        """Test that first registered provider becomes default"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        self.assertEqual(embedding._default, "openai")

    def test_register_multiple_providers(self):
        """Test registering multiple providers"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        embedding.register("cohere", create_mock_embedding_provider("cohere"))
        self.assertEqual(len(embedding._providers), 2)
        self.assertEqual(embedding._default, "openai")  # First one remains default

    def test_embed_with_provider_model_format(self):
        """Test embedding with 'provider/model' format"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        result = embedding.embed("test text", "openai/text-embedding-3-small")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 10)
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_embed_with_default_provider(self):
        """Test embedding with default provider (no provider specified)"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        result = embedding.embed("test text", "text-embedding-3-small")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 10)

    def test_embed_with_model_only(self):
        """Test embedding with model name only, using default provider"""
        embedding.register("cohere", create_mock_embedding_provider("cohere"))
        result = embedding.embed("hello", "embed-english")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 10)


class TestEmbeddingErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        """Clear module state before each test"""
        embedding._providers.clear()
        embedding._default = None

    def test_embed_unregistered_provider_raises_error(self):
        """Test that embedding with unregistered provider raises ValueError"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        with self.assertRaises(ValueError) as cm:
            embedding.embed("test", "cohere/embed-model")
        self.assertIn("cohere", str(cm.exception))
        self.assertIn("not registered", str(cm.exception))
        self.assertIn("openai", str(cm.exception))  # Should list available

    def test_embed_no_providers_raises_error(self):
        """Test embedding when no providers are registered"""
        with self.assertRaises(ValueError) as cm:
            embedding.embed("test", "model")
        self.assertIn("No providers registered", str(cm.exception))

    def test_embed_no_default_provider_raises_error(self):
        """Test embedding without provider when no default is set"""
        # This case shouldn't happen in practice, but let's test it
        # by manually clearing default after registration
        embedding.register("openai", create_mock_embedding_provider("openai"))
        embedding._default = None
        with self.assertRaises(ValueError) as cm:
            embedding.embed("test", "model")
        self.assertIn("No default provider", str(cm.exception))

    def test_register_empty_name_raises_error(self):
        """Test that empty provider name raises ValueError"""
        with self.assertRaises(ValueError) as cm:
            embedding.register("", create_mock_embedding_provider("test"))
        self.assertIn("non-empty string", str(cm.exception))

    def test_register_name_with_slash_raises_error(self):
        """Test that provider name with '/' raises ValueError"""
        with self.assertRaises(ValueError) as cm:
            embedding.register("my/provider", create_mock_embedding_provider("test"))
        self.assertIn("cannot contain '/'", str(cm.exception))

    def test_register_non_callable_raises_error(self):
        """Test that non-callable provider raises TypeError"""
        with self.assertRaises(TypeError) as cm:
            embedding.register("openai", "not_a_function")
        self.assertIn("must be callable", str(cm.exception))

    def test_embed_empty_provider_in_model_raises_error(self):
        """Test that '/model' format raises ValueError"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        with self.assertRaises(ValueError) as cm:
            embedding.embed("test", "/text-embedding-3-small")
        self.assertIn("provider part is empty", str(cm.exception))

    def test_embed_non_string_text_raises_error(self):
        """Test that non-string text raises TypeError"""
        embedding.register("openai", create_mock_embedding_provider("openai"))
        with self.assertRaises(TypeError) as cm:
            embedding.embed(123, "model")
        self.assertIn("must be a string", str(cm.exception))


class TestEmbeddingThreadSafety(unittest.TestCase):
    """Test thread safety of concurrent embedding calls"""

    def setUp(self):
        """Clear module state and register test providers"""
        embedding._providers.clear()
        embedding._default = None
        # Register providers BEFORE threads start (mimics real usage)
        embedding.register("test", create_mock_embedding_provider("test"))
        embedding.register("openai", create_mock_embedding_provider("openai"))
        embedding.register("cohere", create_mock_embedding_provider("cohere"))

    def test_concurrent_embeds_same_provider(self):
        """Test multiple threads embedding with the same provider simultaneously"""
        results = []
        errors = []

        def embed_text(text):
            try:
                result = embedding.embed(text, "test/model")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=embed_text, args=(f"text_{i}",))
                   for i in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All embeds should succeed
        self.assertEqual(len(results), 50)
        self.assertEqual(len(errors), 0)

    def test_concurrent_embeds_different_providers(self):
        """Test multiple threads embedding with different providers simultaneously"""
        results = []

        def embed_text(provider_model, text):
            result = embedding.embed(text, provider_model)
            results.append(result)

        threads = []
        for i in range(25):
            threads.append(threading.Thread(target=embed_text, args=("openai/text-embedding-3-small", f"t{i}")))
            threads.append(threading.Thread(target=embed_text, args=("cohere/embed-english", f"t{i}")))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 50)
        # All results should be valid embeddings
        for result in results:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 10)

    def test_concurrent_embeds_with_default_provider(self):
        """Test multiple threads using default provider simultaneously"""
        results = []

        def embed_text(text):
            # No provider specified, should use default (test)
            result = embedding.embed(text, "model")
            results.append(result)

        threads = [threading.Thread(target=embed_text, args=(f"text_{i}",))
                   for i in range(30)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 30)
        # All should return valid embeddings
        for result in results:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 10)

    def test_provider_function_isolation(self):
        """Test that provider functions execute independently without lock contention"""
        import time
        embed_times = []

        def slow_embedding_provider(text, model, **kwargs):
            """Simulates a slow embedding provider (like a real API call)"""
            time.sleep(0.01)  # 10ms delay
            return [1.0, 2.0, 3.0]

        embedding.register("slow", slow_embedding_provider)

        def timed_embed(text):
            start = time.time()
            result = embedding.embed(text, "slow/model")
            duration = time.time() - start
            embed_times.append(duration)

        threads = [threading.Thread(target=timed_embed, args=(f"t{i}",))
                   for i in range(10)]

        overall_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        overall_duration = time.time() - overall_start

        # All embeds completed
        self.assertEqual(len(embed_times), 10)

        # Overall time should be ~10ms (parallel), not ~100ms (serial)
        # Allow some overhead, but should be well under 50ms
        self.assertLess(overall_duration, 0.05)


if __name__ == '__main__':
    unittest.main()
