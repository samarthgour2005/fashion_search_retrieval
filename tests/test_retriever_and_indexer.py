import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from retriever.query_parser import QueryParser
from indexer.embed_index import build_faiss_index


class TestQueryParserJSONExtraction(unittest.TestCase):
    def test_standard_json(self):
        text = '{"shirt": "white", "tie": "red"}'
        result = QueryParser._extract_json(text)
        self.assertEqual(result, {"shirt": "white", "tie": "red"})

    def test_nested_json(self):
        text = '{"shirt": "blue", "details": {"fit": "slim", "color": "blue"}}'
        result = QueryParser._extract_json(text)
        self.assertEqual(result, {"shirt": "blue", "details": {"fit": "slim", "color": "blue"}})

    def test_markdown_code_block_json(self):
        text = "```json\n{\n  \"pants\": \"black\"\n}\n```"
        result = QueryParser._extract_json(text)
        self.assertEqual(result, {"pants": "black"})

    def test_markdown_code_block_with_surrounding_text(self):
        text = "Here is the extracted attributes:\n```json\n{\"coat\": \"yellow\"}\n```\nHope this helps!"
        result = QueryParser._extract_json(text)
        self.assertEqual(result, {"coat": "yellow"})

    def test_invalid_json(self):
        text = "This is not valid json {shirt: blue}"
        result = QueryParser._extract_json(text)
        self.assertEqual(result, {})

    def test_non_dict_json(self):
        text = '[1, 2, 3]'
        result = QueryParser._extract_json(text)
        self.assertEqual(result, {})


class TestFashionSearchEngineLazyLoading(unittest.TestCase):
    @patch("retriever.rerank_search.load_faiss_index")
    @patch("retriever.rerank_search.load_clip")
    @patch("pickle.load")
    @patch("builtins.open")
    def test_lazy_loading(self, mock_open, mock_pickle, mock_load_clip, mock_load_faiss):
        mock_load_clip.return_value = (MagicMock(), MagicMock())
        mock_load_faiss.return_value = MagicMock()
        mock_pickle.return_value = {
            "image_ids": [101, 102],
            "descriptions": ["desc1", "desc2"],
            "garment_colors_per_image": [{"shirt": "blue"}, {"shirt": "red"}]
        }

        from retriever.rerank_search import FashionSearchEngine

        engine = FashionSearchEngine()

        # Verify parser is NOT instantiated at init time
        self.assertIsNone(engine._parser)

        # Baseline search should NOT trigger parser instantiation
        engine.encode_query = MagicMock(return_value=np.zeros((1, 512), dtype=np.float32))
        engine.index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))

        results = engine.search_baseline("blue shirt")
        self.assertEqual(len(results), 2)
        self.assertIsNone(engine._parser)

        # Explicit query_attrs passed to search() should NOT trigger parser instantiation
        results, query_attrs = engine.search("blue shirt", query_attrs={"shirt": "blue"})
        self.assertEqual(query_attrs, {"shirt": "blue"})
        self.assertIsNone(engine._parser)

        # Accessing engine.parser property should lazily load parser
        with patch("retriever.rerank_search.QueryParser") as mock_qp_cls:
            mock_parser_inst = MagicMock()
            mock_qp_cls.return_value = mock_parser_inst
            p = engine.parser
            self.assertEqual(p, mock_parser_inst)
            self.assertEqual(engine._parser, mock_parser_inst)


class TestFAISSIndexBuilding(unittest.TestCase):
    def test_build_faiss_index_cpu(self):
        fused = np.random.randn(50, 128).astype("float32")
        # normalize vectors
        fused = fused / np.linalg.norm(fused, axis=1, keepdims=True)

        index, use_gpu = build_faiss_index(fused, force_cpu=True)
        self.assertFalse(use_gpu)
        self.assertEqual(index.ntotal, 50)


if __name__ == "__main__":
    unittest.main()
