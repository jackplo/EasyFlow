# tests/test_nodes_search.py
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from easyflow.utils import search as search_module
from easyflow.nodes import SearchNode


def create_mock_search(name):
    """Creates a mock search provider"""
    return lambda query, num_results, **kwargs: [
        {"title": f"{name} Result {i}", "snippet": f"About {query}", "url": f"https://{name}.com/{i}"}
        for i in range(num_results)
    ]


class TestSearchNodeBasic(unittest.TestCase):
    """Test basic SearchNode functionality"""

    def setUp(self):
        """Clear search module state and register mock provider"""
        search_module._providers.clear()
        search_module._default = None
        search_module.register("mock", create_mock_search("mock"))

    def test_simple_search(self):
        """Test basic search functionality"""
        node = SearchNode(input_key="query", output_key="results")
        shared = {"query": "python tutorial"}
        node.run(shared)
        self.assertIn("results", shared)
        self.assertEqual(len(shared["results"]), 5)  # default num_results

    def test_custom_keys(self):
        """Test custom input/output keys"""
        node = SearchNode(input_key="user_query", output_key="search_output")
        shared = {"user_query": "test search"}
        node.run(shared)
        self.assertIn("search_output", shared)
        self.assertNotIn("search_results", shared)

    def test_num_results(self):
        """Test custom num_results"""
        node = SearchNode(num_results=3)
        shared = {"query": "test"}
        node.run(shared)
        self.assertEqual(len(shared["search_results"]), 3)

    def test_empty_query(self):
        """Test that empty query returns empty list"""
        node = SearchNode()
        shared = {"query": ""}
        node.run(shared)
        self.assertEqual(shared["search_results"], [])

    def test_missing_query(self):
        """Test that missing query returns empty list"""
        node = SearchNode()
        shared = {}
        node.run(shared)
        self.assertEqual(shared["search_results"], [])

    def test_explicit_provider(self):
        """Test explicit provider specification"""
        search_module.register("other", create_mock_search("other"))
        node = SearchNode(provider="other")
        shared = {"query": "test"}
        node.run(shared)
        self.assertEqual(shared["search_results"][0]["title"], "other Result 0")


class TestSearchNodeFormatting(unittest.TestCase):
    """Test SearchNode result formatting"""

    def setUp(self):
        """Clear search module state and register mock provider"""
        search_module._providers.clear()
        search_module._default = None
        search_module.register("mock", create_mock_search("mock"))

    def test_format_results_false(self):
        """Test that format_results=False returns raw list"""
        node = SearchNode(format_results=False)
        shared = {"query": "test"}
        node.run(shared)
        self.assertIsInstance(shared["search_results"], list)
        self.assertIsInstance(shared["search_results"][0], dict)

    def test_format_results_true(self):
        """Test that format_results=True returns formatted string"""
        node = SearchNode(format_results=True, num_results=2)
        shared = {"query": "test"}
        node.run(shared)
        result = shared["search_results"]
        self.assertIsInstance(result, str)
        self.assertIn("1. mock Result 0", result)
        self.assertIn("2. mock Result 1", result)
        self.assertIn("URL:", result)

    def test_format_empty_results(self):
        """Test formatting with no results"""
        def empty_provider(query, num_results, **kwargs):
            return []

        search_module.register("empty", empty_provider)
        node = SearchNode(provider="empty", format_results=True)
        shared = {"query": "test"}
        node.run(shared)
        self.assertEqual(shared["search_results"], "No results found.")

    def test_format_empty_query(self):
        """Test that empty query with format_results returns empty string"""
        node = SearchNode(format_results=True)
        shared = {"query": ""}
        node.run(shared)
        self.assertEqual(shared["search_results"], "")


class TestSearchNodeConfiguration(unittest.TestCase):
    """Test SearchNode configuration options"""

    def setUp(self):
        """Clear search module state and register mock provider"""
        search_module._providers.clear()
        search_module._default = None
        search_module.register("mock", create_mock_search("mock"))

    def test_default_values(self):
        """Test default configuration values"""
        node = SearchNode()
        self.assertEqual(node.input_key, "query")
        self.assertEqual(node.output_key, "search_results")
        self.assertIsNone(node.provider)
        self.assertEqual(node.num_results, 5)
        self.assertFalse(node.format_results)
        self.assertEqual(node.max_retries, 3)
        self.assertEqual(node.wait, 1)

    def test_retry_configuration(self):
        """Test max_retries and wait configuration"""
        node = SearchNode(max_retries=5, wait=2)
        self.assertEqual(node.max_retries, 5)
        self.assertEqual(node.wait, 2)

    def test_search_kwargs_passed_through(self):
        """Test that extra kwargs are passed to search"""
        received_kwargs = {}

        def capturing_search(query, num_results, **kwargs):
            received_kwargs.update(kwargs)
            return []

        search_module._providers["mock"] = capturing_search

        node = SearchNode(region="us", safe_search=True)
        shared = {"query": "test"}
        node.run(shared)

        self.assertEqual(received_kwargs.get("region"), "us")
        self.assertEqual(received_kwargs.get("safe_search"), True)


if __name__ == '__main__':
    unittest.main()
