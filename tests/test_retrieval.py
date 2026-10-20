import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from learning_agent import knowledge


class FakeEmbedding:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for text in texts:
            yield [1., 0.] if 'rag' in text.lower() else [0., 1.]


class RetrievalTests(unittest.TestCase):
    def test_fusion(self):
        self.assertEqual(knowledge.fuse_rankings(['b', 'a'], ['a', 'c']), ['a', 'b', 'c'])

    def test_quoted_keywords(self):
        self.assertIn('"rag"', knowledge.keyword_query('什么是RAG？" *'))
        self.assertNotIn('*', knowledge.keyword_query('RAG *'))
        self.assertEqual(knowledge.keyword_query('?!'), '')

    def test_cache_and_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / 'docs'
            docs.mkdir()
            path = docs / 'rag.txt'
            path.write_text('RAG 是检索增强生成。', encoding='utf-8')
            (docs / 'python.txt').write_text('Python 是编程语言。', encoding='utf-8')
            model = FakeEmbedding()
            with patch.object(knowledge, 'embedding_model', return_value=model):
                first = knowledge.retrieve_candidates('什么是 RAG？', root / 'index.db', docs)
                self.assertEqual(first[0].source, 'rag.txt')
                self.assertEqual(model.calls, 2)
                self.assertEqual(first, knowledge.retrieve_candidates('什么是 RAG？', root / 'index.db', docs))
                self.assertEqual(model.calls, 3)
                path.write_text('RAG 资料更新了。', encoding='utf-8')
                updated = knowledge.retrieve_candidates('RAG', root / 'index.db', docs)
                self.assertEqual(model.calls, 5)
                self.assertNotEqual(first[0].chunk_id, updated[0].chunk_id)

    def test_missing_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(knowledge, 'embedding_model') as model:
                self.assertEqual(knowledge.retrieve_candidates('RAG', Path(directory)/'index.db', Path(directory)), [])
                model.assert_not_called()

    def test_dimensions(self):
        with self.assertRaises(ValueError):
            knowledge.cosine([1], [1, 2])
