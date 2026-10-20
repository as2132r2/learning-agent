"""Day31 第2步：最终RAG服务的HTTP API契约测试。

运行：
    python day31/python/day31_api_contract_tests.py

特点：
    - 使用aiohttp本机临时测试服务器；
    - 使用临时SQLite数据库；
    - 使用Fake RAG结果；
    - 不监听8050端口，不调用真实大模型，不修改正式数据库。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiohttp.test_utils import TestClient, TestServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY30_PYTHON = PROJECT_ROOT / "day30" / "python"
if str(DAY30_PYTHON) not in sys.path:
    sys.path.insert(0, str(DAY30_PYTHON))

import day30_production_rag_service as service  # noqa: E402
from day30_config_management import load_config  # noqa: E402
from day30_database_migrations import migrate_database, open_database  # noqa: E402


FAKE_ANSWER = "RAG 是检索增强生成。[资料1]"
FAKE_SOURCES = [
    {
        "chunk_id": "chunk-contract-001",
        "location": "tests/contract-fixture.txt",
        "relevance_score": 99,
        "content": "RAG 的中文名称是检索增强生成。",
    }
]
FAKE_USAGE = {
    "input_tokens": 120,
    "output_tokens": 20,
    "total_tokens": 140,
}


def assert_exact_keys(
    testcase: unittest.TestCase,
    data: dict[str, object],
    expected: set[str],
) -> None:
    """契约要求字段不能缺少，也不能悄悄增加。"""

    testcase.assertEqual(set(data), expected)


class ProductionRAGContractTests(unittest.IsolatedAsyncioTestCase):
    """每个测试方法共享一台隔离的aiohttp测试服务。"""

    client: TestClient
    temporary_directory: tempfile.TemporaryDirectory[str]
    rag_patch: object

    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="day31-contract-"
        )
        database_path = Path(self.temporary_directory.name) / "contract.db"

        loaded = load_config(service.empty_config_arguments())
        self.config = replace(loaded.config, database_path=database_path)

        connection = open_database(database_path)
        try:
            migrate_database(connection)
        finally:
            connection.close()

        self.fake_pipeline = AsyncMock(
            return_value=(FAKE_ANSWER, FAKE_SOURCES, FAKE_USAGE)
        )
        self.rag_patch = patch.object(
            service,
            "execute_rag_pipeline",
            self.fake_pipeline,
        )
        self.rag_patch.start()

        app = service.create_app(self.config)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.rag_patch.stop()
        self.temporary_directory.cleanup()

    async def test_01_index_contract(self) -> None:
        response = await self.client.get("/")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "application/json")

        data = await response.json()
        assert_exact_keys(self, data, {"service", "version", "endpoints"})
        self.assertIsInstance(data["service"], str)
        self.assertEqual(data["version"], "day30-v1")
        self.assertIsInstance(data["endpoints"], list)
        self.assertIn("POST /ask", data["endpoints"])

    async def test_02_liveness_contract(self) -> None:
        response = await self.client.get("/livez")
        self.assertEqual(response.status, 200)
        data = await response.json()

        assert_exact_keys(
            self,
            data,
            {"status", "service", "timestamp", "uptime_seconds"},
        )
        self.assertEqual(data["status"], "alive")
        self.assertIsInstance(data["uptime_seconds"], float)

    async def test_03_readiness_contract(self) -> None:
        response = await self.client.get("/readyz")
        self.assertEqual(response.status, 200)
        data = await response.json()

        assert_exact_keys(
            self,
            data,
            {"status", "service", "timestamp", "checks"},
        )
        self.assertEqual(data["status"], "ready")
        self.assertIsInstance(data["checks"], list)
        self.assertGreaterEqual(len(data["checks"]), 4)
        for check in data["checks"]:
            assert_exact_keys(
                self,
                check,
                {"name", "status", "critical", "latency_ms", "detail"},
            )
            self.assertEqual(check["status"], "pass")

    async def test_04_successful_ask_contract(self) -> None:
        response = await self.client.post(
            "/ask",
            json={"question": "什么是 RAG？"},
        )
        self.assertEqual(response.status, 200)
        data = await response.json()

        assert_exact_keys(
            self,
            data,
            {
                "ok",
                "request_id",
                "trace_id",
                "answer",
                "sources",
                "usage",
            },
        )
        self.assertIs(data["ok"], True)
        self.assertRegex(data["request_id"], r"^req-[0-9a-f]{16}$")
        self.assertRegex(data["trace_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(data["answer"], FAKE_ANSWER)
        self.assertEqual(data["usage"], FAKE_USAGE)
        self.assertEqual(len(data["sources"]), 1)

        source = data["sources"][0]
        assert_exact_keys(
            self,
            source,
            {"chunk_id", "location", "relevance_score"},
        )
        # HTTP响应不应该返回用于生成答案的完整正文。
        self.assertNotIn("content", source)
        self.fake_pipeline.assert_awaited_once()

    async def test_05_request_detail_contract(self) -> None:
        ask_response = await self.client.post(
            "/ask",
            json={"question": "什么是 RAG？"},
        )
        request_id = (await ask_response.json())["request_id"]

        response = await self.client.get(f"/requests/{request_id}")
        self.assertEqual(response.status, 200)
        data = await response.json()

        assert_exact_keys(
            self,
            data,
            {
                "request_id",
                "question",
                "answer",
                "status",
                "created_at",
                "trace_id",
                "sources",
            },
        )
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["answer"], FAKE_ANSWER)
        self.assertEqual(len(data["sources"]), 1)
        assert_exact_keys(
            self,
            data["sources"][0],
            {"chunk_id", "source_path", "relevance_score"},
        )

    async def test_06_ops_status_contract(self) -> None:
        await self.client.post("/ask", json={"question": "什么是 RAG？"})
        response = await self.client.get("/ops/status")
        self.assertEqual(response.status, 200)
        data = await response.json()

        assert_exact_keys(
            self,
            data,
            {
                "request_counts",
                "recent_requests",
                "service",
                "environment",
                "model",
                "max_concurrency",
            },
        )
        self.assertEqual(data["request_counts"]["success"], 1)
        self.assertEqual(len(data["recent_requests"]), 1)
        self.assertIsInstance(data["max_concurrency"], int)

    async def test_07_invalid_json_is_400(self) -> None:
        response = await self.client.post(
            "/ask",
            data="这不是JSON",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status, 400)
        self.fake_pipeline.assert_not_awaited()

    async def test_08_extra_field_is_400(self) -> None:
        response = await self.client.post(
            "/ask",
            json={"question": "什么是 RAG？", "admin": True},
        )
        self.assertEqual(response.status, 400)
        self.fake_pipeline.assert_not_awaited()

    async def test_09_blank_question_is_400(self) -> None:
        response = await self.client.post("/ask", json={"question": "   "})
        self.assertEqual(response.status, 400)
        self.fake_pipeline.assert_not_awaited()

    async def test_10_missing_request_is_404(self) -> None:
        response = await self.client.get("/requests/req-does-not-exist")
        self.assertEqual(response.status, 404)


def main() -> None:
    print("Day31 HTTP API契约测试")
    print("测试目标：Day30最终生产化RAG服务")
    print("真实模型调用：否")
    print("正式数据库修改：否")
    print("=" * 88)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        ProductionRAGContractTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 88)
    print("契约测试总数：", result.testsRun)
    print("失败数量：", len(result.failures))
    print("错误数量：", len(result.errors))
    print("跳过数量：", len(result.skipped))
    print("最终结论：", "通过" if result.wasSuccessful() else "失败")
    print("\n本步数据流：")
    print("unittest → 临时SQLite迁移 → aiohttp TestServer/TestClient")
    print("→ patch真实RAG为AsyncMock → 发送真实HTTP请求")
    print("→ 检查状态码、字段集合、字段类型和数据库结果")
    print("→ tearDown关闭测试服务并删除临时数据库")

    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
