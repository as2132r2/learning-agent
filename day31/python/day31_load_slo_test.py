"""Day31 第4步：并发压测与SLO自动判定。

运行：
    python day31/python/day31_load_slo_test.py

可选参数：
    python day31/python/day31_load_slo_test.py \
        --requests 30 --client-concurrency 10 --pipeline-concurrency 3

本程序使用Fake RAG流水线，不调用真实模型，但HTTP、Semaphore、SQLite和
请求持久化全部真实执行。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY30_PYTHON = PROJECT_ROOT / "day30" / "python"
if str(DAY30_PYTHON) not in sys.path:
    sys.path.insert(0, str(DAY30_PYTHON))

import day30_production_rag_service as service  # noqa: E402
from day30_config_management import load_config  # noqa: E402
from day30_database_migrations import migrate_database, open_database  # noqa: E402


DAY31_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = DAY31_DIR / "logs" / "day31_load_test_report.json"
FAKE_PIPELINE_DELAY_SECONDS = 0.08

# 本地Fake压测的服务目标，不代表真实模型服务的生产SLO。
MIN_SUCCESS_RATE = 0.99
MAX_P95_SECONDS = 2.0
MIN_THROUGHPUT_RPS = 10.0


@dataclass(frozen=True)
class RequestMeasurement:
    request_number: int
    status_code: int | None
    success: bool
    latency_seconds: float
    request_id: str | None
    error: str | None


@dataclass(frozen=True)
class SLOCheck:
    name: str
    actual: float | int
    target: str
    passed: bool


class FakePipelineProbe:
    """模拟固定耗时流水线，并观测服务端实际并发数。"""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.active = 0
        self.max_active = 0
        self.total_calls = 0
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        app: Any,
        question: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        async with self._lock:
            self.active += 1
            self.total_calls += 1
            self.max_active = max(self.max_active, self.active)

        try:
            await asyncio.sleep(self.delay_seconds)
            return (
                f"Fake回答：{question} [资料1]",
                [
                    {
                        "chunk_id": "chunk-load-test-001",
                        "location": "fixtures/load-test.txt",
                        "relevance_score": 100,
                        "content": "用于并发压测的固定资料。",
                    }
                ],
                {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                },
            )
        finally:
            async with self._lock:
                self.active -= 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def percentile(values: list[float], percent: float) -> float:
    """使用nearest-rank方法计算百分位数。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


async def send_one_request(
    client: TestClient,
    client_semaphore: asyncio.Semaphore,
    request_number: int,
) -> RequestMeasurement:
    """客户端Semaphore控制同时发出的HTTP请求数量。"""

    async with client_semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/ask",
                json={"question": f"压测问题-{request_number}"},
            )
            data = await response.json()
            latency = time.perf_counter() - started
            success = response.status == 200 and data.get("ok") is True
            return RequestMeasurement(
                request_number=request_number,
                status_code=response.status,
                success=success,
                latency_seconds=round(latency, 6),
                request_id=data.get("request_id"),
                error=None if success else str(data.get("error", "未知错误")),
            )
        except Exception as error:
            return RequestMeasurement(
                request_number=request_number,
                status_code=None,
                success=False,
                latency_seconds=round(time.perf_counter() - started, 6),
                request_id=None,
                error=f"{type(error).__name__}: {error}",
            )


def count_database_statuses(config: Any) -> dict[str, int]:
    with service.open_business_database(config) as connection:
        return {
            row["status"]: row["count"]
            for row in connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM rag_requests GROUP BY status
                """
            )
        }


def write_report_atomically(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".load-report-",
        suffix=".json",
        dir=REPORT_PATH.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, REPORT_PATH)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


async def run_load_test(arguments: argparse.Namespace) -> int:
    temporary_directory = tempfile.TemporaryDirectory(prefix="day31-load-")
    database_path = Path(temporary_directory.name) / "load-test.db"
    loaded = load_config(service.empty_config_arguments())
    config = replace(
        loaded.config,
        database_path=database_path,
        max_concurrency=arguments.pipeline_concurrency,
        request_timeout_seconds=5.0,
    )

    connection = open_database(database_path)
    try:
        migrate_database(connection)
    finally:
        connection.close()

    probe = FakePipelineProbe(FAKE_PIPELINE_DELAY_SECONDS)
    app = service.create_app(config)
    client = TestClient(TestServer(app))
    client_semaphore = asyncio.Semaphore(arguments.client_concurrency)

    try:
        with patch.object(service, "execute_rag_pipeline", probe):
            await client.start_server()

            # 两个预热请求用于完成连接池、路由和SQLite首次访问，不计入指标。
            for warmup_number in (-2, -1):
                warmup = await send_one_request(
                    client,
                    asyncio.Semaphore(1),
                    warmup_number,
                )
                if not warmup.success:
                    raise RuntimeError(f"预热请求失败：{warmup.error}")

            started = time.perf_counter()
            tasks = [
                asyncio.create_task(
                    send_one_request(
                        client,
                        client_semaphore,
                        request_number,
                    ),
                    name=f"load-request-{request_number}",
                )
                for request_number in range(1, arguments.requests + 1)
            ]
            measurements = await asyncio.gather(*tasks)
            wall_time = time.perf_counter() - started

        database_counts = await asyncio.to_thread(count_database_statuses, config)
    finally:
        await client.close()
        temporary_directory.cleanup()

    latencies = [item.latency_seconds for item in measurements]
    success_count = sum(item.success for item in measurements)
    success_rate = success_count / len(measurements)
    throughput = len(measurements) / wall_time
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    # total_calls包含2个预热请求。
    expected_persisted = arguments.requests + 2
    checks = [
        SLOCheck(
            "success_rate",
            round(success_rate, 4),
            f">= {MIN_SUCCESS_RATE:.2%}",
            success_rate >= MIN_SUCCESS_RATE,
        ),
        SLOCheck(
            "p95_latency_seconds",
            round(p95, 4),
            f"<= {MAX_P95_SECONDS:.3f}",
            p95 <= MAX_P95_SECONDS,
        ),
        SLOCheck(
            "throughput_rps",
            round(throughput, 3),
            f">= {MIN_THROUGHPUT_RPS:.3f}",
            throughput >= MIN_THROUGHPUT_RPS,
        ),
        SLOCheck(
            "server_max_active_pipeline",
            probe.max_active,
            f"<= {arguments.pipeline_concurrency}",
            probe.max_active <= arguments.pipeline_concurrency,
        ),
        SLOCheck(
            "persisted_success_records",
            database_counts.get("success", 0),
            f"== {expected_persisted}",
            database_counts.get("success", 0) == expected_persisted,
        ),
    ]
    passed = all(check.passed for check in checks)

    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "configuration": {
            "request_count": arguments.requests,
            "client_concurrency": arguments.client_concurrency,
            "pipeline_concurrency": arguments.pipeline_concurrency,
            "fake_pipeline_delay_seconds": FAKE_PIPELINE_DELAY_SECONDS,
            "warmup_requests": 2,
        },
        "metrics": {
            "success_count": success_count,
            "failure_count": len(measurements) - success_count,
            "success_rate": round(success_rate, 4),
            "wall_time_seconds": round(wall_time, 4),
            "throughput_rps": round(throughput, 3),
            "latency_min_seconds": round(min(latencies), 4),
            "latency_average_seconds": round(
                sum(latencies) / len(latencies), 4
            ),
            "latency_p50_seconds": round(p50, 4),
            "latency_p95_seconds": round(p95, 4),
            "latency_p99_seconds": round(p99, 4),
            "latency_max_seconds": round(max(latencies), 4),
            "server_max_active_pipeline": probe.max_active,
            "database_status_counts": database_counts,
        },
        "slo_checks": [asdict(check) for check in checks],
        "slo_result": "passed" if passed else "failed",
        "requests": [asdict(item) for item in measurements],
    }
    write_report_atomically(report)

    print("\n" + "=" * 100)
    print("压测结果：")
    print(f"成功请求：{success_count}/{len(measurements)}")
    print(f"整体耗时：{wall_time:.3f}秒")
    print(f"吞吐量：{throughput:.2f} requests/second")
    print(f"延迟 min/avg：{min(latencies):.3f}/{sum(latencies)/len(latencies):.3f}秒")
    print(f"延迟 P50/P95/P99/max：{p50:.3f}/{p95:.3f}/{p99:.3f}/{max(latencies):.3f}秒")
    print("服务端观测到的最大流水线并发：", probe.max_active)
    print("数据库状态统计（包含2个预热请求）：", database_counts)

    print("\nSLO判定：")
    for check in checks:
        symbol = "通过" if check.passed else "失败"
        print(f"- {symbol} | {check.name}: actual={check.actual}, target={check.target}")
    print("最终SLO：", "通过" if passed else "失败")
    print("报告：", REPORT_PATH)

    print("\n本步数据流：")
    print("N个客户端Task → 客户端Semaphore → N条POST /ask")
    print("→ 服务端Semaphore限制Fake流水线并发 → SQLite并发持久化")
    print("→ RequestMeasurement列表 → P50/P95/P99/吞吐量/成功率")
    print("→ SLOCheck列表 → passed/failed → 原子JSON报告 + 退出码")
    return 0 if passed else 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day31并发压测与SLO")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--client-concurrency", type=int, default=12)
    parser.add_argument("--pipeline-concurrency", type=int, default=3)
    arguments = parser.parse_args()
    for name in ("requests", "client_concurrency", "pipeline_concurrency"):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name.replace('_', '-')}必须大于0")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    print("Day31 并发压测与SLO自动判定")
    print("请求数量：", arguments.requests)
    print("客户端最大并发：", arguments.client_concurrency)
    print("服务端流水线最大并发：", arguments.pipeline_concurrency)
    print("Fake流水线单次延迟：", FAKE_PIPELINE_DELAY_SECONDS, "秒")
    print("真实模型调用：否")
    print("正式数据库修改：否")
    raise SystemExit(asyncio.run(run_load_test(arguments)))


if __name__ == "__main__":
    main()
