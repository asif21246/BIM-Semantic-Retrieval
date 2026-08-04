import unittest

from retrieval.semantic_search import BIMSemanticSearchEngine


class TestQueryExpansion(unittest.TestCase):
    def test_expand_query_keeps_product_family_and_room_context(self):
        engine = BIMSemanticSearchEngine.__new__(BIMSemanticSearchEngine)
        expanded = engine._expand_query("Find sprinklers in Room 101 on Level 1")
        self.assertIn("sprinkler", expanded.lower())
        self.assertIn("room 101", expanded.lower())
        self.assertIn("level 1", expanded.lower())


if __name__ == "__main__":
    unittest.main()
