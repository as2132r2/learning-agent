"""Day31 第3步：使用Fake依赖执行端到端RAG集成测试。

运行：
    python day31/python/day31_fake_integration_tests.py

与第2步的区别：
    第2步把execute_rag_pipeline整体替换成AsyncMock；
    第3步真实执行execute_rag_pipeline，只替换检索数据、Reranker结果
    和最外部模型网络调用。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from langchain_core.messages import AIMessage, BaseMessage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY30_PYTHON = PROJECT_ROOT / "day30" / "python"
DAY22_PYTHON = PROJECT_ROOT / "day22" / "python"
for path in (DAY30_PYTHON, DAY22_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import day30_production_rag_service as service  # noqa: E402
from day30_config_management import load_config  # noqa: E402
from day30_database_migrations import migrate_database, open_database  # noqa: E402
from day22_llm_reranker import RerankedResult  # noqa: E402
from day22_rrf_fusion import RRFResult  # noqa: E402


QUESTION = "什么是 RAG？"
EVIDENCE_CONTENT = (
    "RAG 的中文名称是检索增强生成。"
    "它先检索外部资料，再把资料和问题交给大语言模型。"
)


class DeterministicFakeLLM:
    """模拟ChatOpenAI的ainvoke接口，并记录收到的Prompt。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.error: Exception | None = None
        self.call_count = 0
        self.last_messages: list[BaseMessage] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.call_count += 1
        self.last_messages = messages
        if self.error is not None:
            raise self.error
        return AIMessage(
            content=self.answer,
            usage_metadata={
                "input_tokens": 88,
                "output_tokens": 18,
                "total_tokens": 106,
            },
        )

def fake_rrf_candidates(question: str) -> tuple[list[RRFResult], dict[str, Any]]:
    """模拟已经完成向量检索、关键词检索和RRF融合。"""

    candidate = RRFResult(
        chunk_id="chunk-fake-rag-001",
        location="fixtures/rag.txt",
        content=EVIDENCE_CONTENT,
        vector_rank=1,
        vector_distance=0.12,
        keyword_rank=1,
        bm25_score=-2.5,
        vector_rrf=1 / 61,
        keyword_rrf=1 / 61,
        rrf_score=2 / 61,
    )
    return [candidate], {
        "terms": ["rag"],
        "match_query": '"rag"',
        "vector_count": 1,
        "keyword_count": 1,
    }


def fake_rerank_with_score(score: int):
    """创建一个可传给patch的确定性Reranker函数。"""

    def rerank(
        question: str,
        candidates: list[RRFResult],
    ) -> tuple[list[RerankedResult], None]:
        candidate = candidates[0]
        result = RerankedResult(
            chunk_id=candidate.chunk_id,
            location=candidate.location,
            content=candidate.content,
            rrf_rank=1,
            rrf_score=candidate.rrf_score,
            relevance_score=score,
            reason="Fake Reranker确定性评分。",
        )
        return [result], None

    return rerank


class FakeEndToEndIntegrationTests(unittest.IsolatedAsyncioTestCase):
    client: TestClient

    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="day31-integration-"
        )
        database_path = Path(self.temporary_directory.name) / "integration.db"

        loaded = load_config(service.empty_config_arguments())
        self.config = replace(
            loaded.config,
            database_path=database_path,
            request_timeout_seconds=2.0,
        )
        connection = open_database(database_path)
        try:
            migrate_database(connection)
        finally:
            connection.close()

        self.app = service.create_app(self.config)
        # 必须在Application启动前注入依赖；启动后修改app状态会产生弃用警告。
        self.fake_llm = DeterministicFakeLLM("默认Fake回答。[资料1]")
        self.app["llm"] = self.fake_llm
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.temporary_directory.cleanup()

    async def test_01_relevant_evidence_runs_answer_model_and_persists(self) -> None:
        self.fake_llm.answer = (
            "RAG 是检索增强生成，先检索资料再生成回答。[资料1]"
        )

        with (
            patch.object(
                service,
                "retrieve_rrf_candidates",
                side_effect=fake_rrf_candidates,
            ) as retrieve_mock,
            patch.object(
                service,
                "rerank_candidates",
                side_effect=fake_rerank_with_score(99),
            ) as rerank_mock,
        ):
            response = await self.client.post(
                "/ask",
                json={"question": QUESTION},
            )

        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertIs(data["ok"], True)
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["relevance_score"], 99)
        self.assertEqual(data["usage"]["total_tokens"], 106)
        self.assertEqual(self.fake_llm.call_count, 1)
        retrieve_mock.assert_called_once_with(QUESTION)
        rerank_mock.assert_called_once()

        # 检查真实Prompt确实包含问题、证据和稳定引用编号。
        prompt_text = "\n".join(
            str(message.content) for message in self.fake_llm.last_messages
        )
        self.assertIn(QUESTION, prompt_text)
        self.assertIn(EVIDENCE_CONTENT, prompt_text)
        self.assertIn("[资料1]", prompt_text)

        detail_response = await self.client.get(
            f"/requests/{data['request_id']}"
        )
        detail = await detail_response.json()
        self.assertEqual(detail["status"], "success")
        self.assertEqual(detail["sources"][0]["chunk_id"], "chunk-fake-rag-001")

    async def test_02_low_score_is_rejected_before_answer_model(self) -> None:
        self.fake_llm.answer = "这段回答不应该被调用。"

        with (
            patch.object(
                service,
                "retrieve_rrf_candidates",
                side_effect=fake_rrf_candidates,
            ),
            patch.object(
                service,
                "rerank_candidates",
                side_effect=fake_rerank_with_score(40),
            ),
        ):
            response = await self.client.post(
                "/ask",
                json={"question": "法国的首都是哪里？"},
            )

        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(
            data["answer"],
            "知识库中没有足够可靠的资料回答这个问题。",
        )
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["usage"]["total_tokens"], 0)
        self.assertEqual(self.fake_llm.call_count, 0)

        detail = await (
            await self.client.get(f"/requests/{data['request_id']}")
        ).json()
        self.assertEqual(detail["status"], "success")
        self.assertEqual(detail["sources"], [])

    async def test_03_model_failure_returns_500_and_persists_failed(self) -> None:
        self.fake_llm.error = RuntimeError("模拟模型服务不可用")

        with (
            patch.object(
                service,
                "retrieve_rrf_candidates",
                side_effect=fake_rrf_candidates,
            ),
            patch.object(
                service,
                "rerank_candidates",
                side_effect=fake_rerank_with_score(99),
            ),
            self.assertLogs(service.LOGGER, level="ERROR") as captured_logs,
        ):
            response = await self.client.post(
                "/ask",
                json={"question": QUESTION},
            )

        self.assertEqual(response.status, 500)
        self.assertTrue(
            any("unhandled_request_error" in line for line in captured_logs.output)
        )
        error_data = await response.json()
        self.assertIs(error_data["ok"], False)
        self.assertEqual(error_data["error"], "服务器内部错误。")
        self.assertNotIn("模拟模型服务不可用", error_data["error"])
        self.assertEqual(self.fake_llm.call_count, 1)

        # 中间件在安全错误响应中返回request_id，可继续审计失败记录。
        detail = await (
            await self.client.get(
                f"/requests/{error_data['request_id']}"
            )
        ).json()
        self.assertEqual(detail["status"], "failed")
        self.assertEqual(detail["answer"], "RAG流水线执行失败。")
        self.assertEqual(detail["sources"], [])


def main() -> None:
    print("Day31 Fake端到端RAG集成测试")
    print("真实execute_rag_pipeline：是")
    print("真实HTTP、中间件、Semaphore、Timeout和SQLite：是")
    print("真实外部检索库和模型网络：否")
    print("=" * 92)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        FakeEndToEndIntegrationTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 92)
    print("集成测试总数：", result.testsRun)
    print("失败数量：", len(result.failures))
    print("错误数量：", len(result.errors))
    print("最终结论：", "通过" if result.wasSuccessful() else "失败")
    print("\n本步覆盖的三条业务路径：")
    print("1. 高相关证据 → 构造Prompt → Fake LLM → success + sources")
    print("2. 低相关证据 → 阈值守门 → 不调用回答模型 → 安全拒绝")
    print("3. 模型异常 → 中间件安全500 → failed记录可按request_id审计")
    print("\n本步数据流：")
    print("Fake候选 → 真实execute_rag_pipeline → Fake Reranker分数")
    print("→ 真实阈值过滤与Prompt构造 → Fake LLM.ainvoke")
    print("→ 真实HTTP响应与SQLite事务 → 查询持久化结果 → 自动清理")

    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
