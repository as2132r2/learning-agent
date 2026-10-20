"""Day11 第五步：使用Mock测试RAG检索到回答的完整链路。"""

import unittest
from dataclasses import dataclass
from typing import Callable
from unittest.mock import Mock


@dataclass(frozen=True)
class RetrievedDocument:
    source: str
    content: str


class RAGPipeline:
    """把检索器和回答模型作为依赖注入，方便做集成测试。"""

    def __init__(
        self,
        retriever: Callable[[str], list[RetrievedDocument]],
        answer_model: Callable[[str], str]
    ) -> None:
        self.retriever = retriever
        self.answer_model = answer_model

    def ask(self, question: str) -> dict:
        documents = self.retriever(question)

        if not documents:
            return {
                "answer": "抱歉，资料中没有相关内容。",
                "sources": [],
                "used_model": False
            }

        context = "\n\n".join(
            f"[来源：{document.source}]\n{document.content}"
            for document in documents
        )

        prompt = (
            f"参考资料：\n{context}\n\n"
            f"用户问题：{question}\n"
            "请只根据参考资料回答，并保留来源。"
        )

        answer = self.answer_model(prompt)

        return {
            "answer": answer,
            "sources": [document.source for document in documents],
            "used_model": True
        }


class TestRAGIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.rag_document = RetrievedDocument(
            source="day6_rag_test.pdf",
            content="RAG 会先检索资料，再把资料交给模型生成回答。"
        )

        self.retriever = Mock()
        self.answer_model = Mock(
            return_value="RAG 会先检索资料，再生成回答。[来源：day6_rag_test.pdf]"
        )
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            answer_model=self.answer_model
        )

    def test_related_question_passes_context_and_source(self) -> None:
        self.retriever.return_value = [self.rag_document]

        result = self.pipeline.ask("什么是 RAG？")

        self.assertTrue(result["used_model"])
        self.assertEqual(result["sources"], ["day6_rag_test.pdf"])
        self.assertIn("[来源：day6_rag_test.pdf]", result["answer"])

        self.answer_model.assert_called_once()
        prompt = self.answer_model.call_args.args[0]
        self.assertIn("RAG 会先检索资料", prompt)
        self.assertIn("什么是 RAG？", prompt)

    def test_unrelated_question_is_rejected_before_model(self) -> None:
        self.retriever.return_value = []

        result = self.pipeline.ask("法国的首都是哪里？")

        self.assertFalse(result["used_model"])
        self.assertEqual(result["sources"], [])
        self.assertIn("没有相关内容", result["answer"])
        self.answer_model.assert_not_called()

    def test_retriever_receives_original_question(self) -> None:
        self.retriever.return_value = []
        question = "为什么要保留来源信息？"

        self.pipeline.ask(question)

        self.retriever.assert_called_once_with(question)

    def test_multiple_documents_are_all_passed_to_model(self) -> None:
        documents = [
            self.rag_document,
            RetrievedDocument(
                source="day6_langgraph.md",
                content="LangGraph 使用节点和边组织流程。"
            )
        ]
        self.retriever.return_value = documents

        result = self.pipeline.ask("请介绍资料中的内容。")

        prompt = self.answer_model.call_args.args[0]
        self.assertIn("day6_rag_test.pdf", prompt)
        self.assertIn("day6_langgraph.md", prompt)
        self.assertEqual(
            result["sources"],
            ["day6_rag_test.pdf", "day6_langgraph.md"]
        )


if __name__ == "__main__":
    print("开始运行 Day11 RAG 集成测试……")
    unittest.main(verbosity=2)


# ============================================================
# 数据流
# ============================================================
#
# 用户问题
#    ↓
# Mock retriever
#    ├── 有Document → 组装context → Mock answer_model
#    │                                  ↓
#    │                              带来源回答
#    └── 空列表 → 拒答 → 不调用answer_model
#
# 集成测试重点：验证组件之间的数据是否正确传递，
# 而不是验证某一个函数内部的单独判断。
