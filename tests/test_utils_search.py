# tests/test_utils_search.py
import unittest
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pocketflow.utils import search as search_module


def create_mock_provider(name):
    """Creates a simple mock search provider for testing"""
    return lambda query, num_results, **kwargs: [
        {"title": f"{name} Result {i}", "snippet": f"Snippet for {query}", "url": f"https://{name}.com/{i}"}
        for i in range(num_results)
    ]


class TestSearchBasicFunctionality(unittest.TestCase):
    """Test basic registration and search functionality"""

    def setUp(self):
        """Clear module state before each test"""
        search_module._providers.clear()
        search_module._default = None

    def test_register_single_provider(self):
        """Test registering a single provider"""
        search_module.register("duckduckgo", create_mock_provider("duckduckgo"))
        self.assertEqual(len(search_module._providers), 1)
        self.assertIn("duckduckgo", search_module._providers)

    def test_register_sets_default_first_time(self):
        """Test that first registered provider becomes default"""
        search_module.register("duckduckgo", create_mock_provider("duckduckgo"))
        self.assertEqual(search_module._default, "duckduckgo")

    def test_register_multiple_providers(self):
        """Test registering multiple providers"""
        search_module.register("duckduckgo", create_mock_provider("duckduckgo"))
        search_module.register("brave", create_mock_provider("brave"))
        self.assertEqual(len(search_module._providers), 2)
        self.assertEqual(search_module._default, "duckduckgo")  # First one remains default

    def test_search_with_explicit_provider(self):
        """Test searching with explicit provider"""
        search_module.register("duckduckgo", create_mock_provider("duckduckgo"))
        results = search_module.web_search("test query", provider="duckduckgo", num_results=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["title"], "duckduckgo Result 0")

    def test_search_with_default_provider(self):
        """Test searching with default provider (no provider specified)"""
        search_module.register("duckduckgo", create_mock_provider("duckduckgo"))
        results = search_module.web_search("test query", num_results=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "duckduckgo Result 0")

    def test_search_num_results_default(self):
        """Test that num_results defaults to 5"""
        search_module.register("test", create_mock_provider("test"))
        results = search_module.web_search("query")
        self.assertEqual(len(results), 5)

    def test_search_passes_kwargs(self):
        """Test that extra kwargs are passed to provider"""
        received_kwargs = {}

        def capturing_provider(query, num_results, **kwargs):
            received_kwargs.update(kwargs)
            return []

        search_module.register("test", capturing_provider)
        search_module.web_search("query", custom_param="value")
        self.assertEqual(received_kwargs.get("custom_param"), "value")


class TestSearchErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        """Clear module state before each test"""
        search_module._providers.clear()
        search_module._default = None

    def test_search_unregistered_provider_raises_error(self):
        """Test that searching with unregistered provider raises ValueError"""
        search_module.register("duckduckgo", create_mock_provider("duckduckgo"))
        with self.assertRaises(ValueError) as cm:
            search_module.web_search("test", provider="brave")
        self.assertIn("brave", str(cm.exception))
        self.assertIn("not registered", str(cm.exception))
        self.assertIn("duckduckgo", str(cm.exception))  # Should list available

    def test_search_no_providers_raises_error(self):
        """Test searching when no providers are registered"""
        with self.assertRaises(ValueError) as cm:
            search_module.web_search("test")
        self.assertIn("No providers registered", str(cm.exception))

    def test_search_no_default_provider_raises_error(self):
        """Test searching without provider when no default is set"""
        search_module.register("test", create_mock_provider("test"))
        search_module._default = None
        with self.assertRaises(ValueError) as cm:
            search_module.web_search("test")
        self.assertIn("No default provider", str(cm.exception))

    def test_register_empty_name_raises_error(self):
        """Test that empty provider name raises ValueError"""
        with self.assertRaises(ValueError) as cm:
            search_module.register("", create_mock_provider("test"))
        self.assertIn("non-empty string", str(cm.exception))

    def test_register_name_with_slash_raises_error(self):
        """Test that provider name with '/' raises ValueError"""
        with self.assertRaises(ValueError) as cm:
            search_module.register("my/provider", create_mock_provider("test"))
        self.assertIn("cannot contain '/'", str(cm.exception))

    def test_register_non_callable_raises_error(self):
        """Test that non-callable provider raises TypeError"""
        with self.assertRaises(TypeError) as cm:
            search_module.register("test", "not_a_function")
        self.assertIn("must be callable", str(cm.exception))

    def test_search_non_string_query_raises_error(self):
        """Test that non-string query raises TypeError"""
        search_module.register("test", create_mock_provider("test"))
        with self.assertRaises(TypeError) as cm:
            search_module.web_search(123)
        self.assertIn("must be a string", str(cm.exception))


class TestSearchThreadSafety(unittest.TestCase):
    """Test thread safety of concurrent searches"""

    def setUp(self):
        """Clear module state and register test providers"""
        search_module._providers.clear()
        search_module._default = None
        search_module.register("test", create_mock_provider("test"))
        search_module.register("duckduckgo", create_mock_provider("duckduckgo"))
        search_module.register("brave", create_mock_provider("brave"))

    def test_concurrent_searches_same_provider(self):
        """Test multiple threads searching the same provider simultaneously"""
        results = []
        errors = []

        def do_search(query):
            try:
                result = search_module.web_search(query, provider="test", num_results=3)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_search, args=(f"query_{i}",))
                   for i in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 50)
        self.assertEqual(len(errors), 0)

    def test_concurrent_searches_different_providers(self):
        """Test multiple threads searching different providers simultaneously"""
        results = []

        def do_search(provider, query):
            result = search_module.web_search(query, provider=provider, num_results=2)
            results.append((provider, result))

        threads = []
        for i in range(25):
            threads.append(threading.Thread(target=do_search, args=("duckduckgo", f"q{i}")))
            threads.append(threading.Thread(target=do_search, args=("brave", f"q{i}")))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 50)
        duckduckgo_count = sum(1 for p, _ in results if p == "duckduckgo")
        brave_count = sum(1 for p, _ in results if p == "brave")
        self.assertEqual(duckduckgo_count, 25)
        self.assertEqual(brave_count, 25)


if __name__ == '__main__':
    unittest.main()
