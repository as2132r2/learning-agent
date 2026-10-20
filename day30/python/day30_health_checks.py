"""Day30 第 3 步：存活、就绪与依赖健康检查。

一次性检查（默认）：
    python day30/python/day30_health_checks.py

启动HTTP健康检查服务：
    python day30/python/day30_health_checks.py serve

接口：
    GET /livez                 只判断进程是否存活
    GET /readyz                判断服务是否可以接收流量
    GET /health/dependencies   显示各项依赖的详细状态
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from aiohttp import web

from day30_config_management import (
    AppConfig,
    ConfigurationError,
    load_config,
)


EXPECTED_SCHEMA_VERSION = 3
STARTED_AT = time.monotonic()


@dataclass(frozen=True)
class CheckResult:
    """单项依赖检查结果，适合直接序列化为JSON。"""

    name: str
    status: str
    critical: bool
    latency_ms: float
    detail: str

    @property
    def healthy(self) -> bool:
        return self.status == "pass"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_config_arguments() -> argparse.Namespace:
    """复用第1步配置加载器，但本程序不增加配置覆盖参数。"""

    return argparse.Namespace(
        environment=None,
        host=None,
        port=None,
        log_level=None,
        request_timeout_seconds=None,
        max_concurrency=None,
    )


def timed_check(
    name: str,
    critical: bool,
    function: Callable[[], str],
) -> CheckResult:
    """统一测量耗时，并把异常转成结构化失败结果。"""

    started = time.perf_counter()
    try:
        detail = function()
        status = "pass"
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        status = "fail"

    return CheckResult(
        name=name,
        status=status,
        critical=critical,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        detail=detail,
    )


def check_configuration(config: AppConfig) -> str:
    """确认关键配置已通过第1步的类型和安全校验。"""

    if not config.model.strip():
        raise RuntimeError("模型名称为空。")
    if not config.api_key.reveal().strip():
        raise RuntimeError("模型密钥为空。")
    return (
        f"配置有效；environment={config.environment}，"
        f"model={config.model}，api_key=********"
    )


def check_database_connection(config: AppConfig) -> str:
    """用只读连接执行最小查询，不修改业务数据。"""

    database_uri = f"file:{config.database_path}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True, timeout=2)
    try:
        result = connection.execute("SELECT 1").fetchone()
        if result != (1,):
            raise RuntimeError("SELECT 1返回了异常结果。")
        return f"SQLite可连接；path={config.database_path}"
    finally:
        connection.close()


def check_database_schema(config: AppConfig) -> str:
    """只有数据库Schema达到程序要求的版本，服务才算就绪。"""

    database_uri = f"file:{config.database_path}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True, timeout=2)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required = {"schema_migrations", "rag_requests", "rag_sources"}
        missing = sorted(required - tables)
        if version != EXPECTED_SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema版本={version}，程序要求={EXPECTED_SCHEMA_VERSION}。"
            )
        if missing:
            raise RuntimeError(f"缺少数据表：{missing}")
        return f"Schema版本={version}；必需表均存在"
    finally:
        connection.close()


def check_model_configuration(config: AppConfig) -> str:
    """本步不付费调用模型，只确认调用模型所需信息齐全。"""

    if not config.base_url.startswith(("https://", "http://")):
        raise RuntimeError("模型base_url格式无效。")
    return (
        f"模型客户端配置齐全；provider_url={config.base_url}；"
        "未发送真实模型请求"
    )


def collect_dependency_checks(config: AppConfig) -> list[CheckResult]:
    """critical=True的失败会让/readyz返回503。"""

    return [
        timed_check("configuration", True, lambda: check_configuration(config)),
        timed_check(
            "database_connection",
            True,
            lambda: check_database_connection(config),
        ),
        timed_check(
            "database_schema",
            True,
            lambda: check_database_schema(config),
        ),
        timed_check(
            "model_configuration",
            True,
            lambda: check_model_configuration(config),
        ),
    ]


def is_ready(checks: list[CheckResult]) -> bool:
    """非关键检查可以失败；任一关键检查失败则不能接流量。"""

    return all(check.healthy for check in checks if check.critical)


def live_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "status": "alive",
        "service": config.app_name,
        "timestamp": utc_now(),
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 3),
    }


def readiness_payload(
    config: AppConfig,
    checks: list[CheckResult],
) -> dict[str, Any]:
    ready = is_ready(checks)
    return {
        "status": "ready" if ready else "not_ready",
        "service": config.app_name,
        "timestamp": utc_now(),
        "checks": [asdict(check) for check in checks],
    }


async def live_handler(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    return web.json_response(live_payload(config), status=200)


async def ready_handler(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    checks = collect_dependency_checks(config)
    ready = is_ready(checks)
    return web.json_response(
        readiness_payload(config, checks),
        status=200 if ready else 503,
    )


async def dependencies_handler(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    checks = collect_dependency_checks(config)
    return web.json_response(
        {
            "status": "healthy" if is_ready(checks) else "degraded",
            "service": config.app_name,
            "schema_expected": EXPECTED_SCHEMA_VERSION,
            "dependencies": [asdict(check) for check in checks],
        },
        status=200 if is_ready(checks) else 503,
    )


def create_app(config: AppConfig) -> web.Application:
    app = web.Application()
    app["config"] = config
    app.router.add_get("/livez", live_handler)
    app.router.add_get("/readyz", ready_handler)
    app.router.add_get("/health/dependencies", dependencies_handler)
    return app


def print_check_report(config: AppConfig) -> int:
    checks = collect_dependency_checks(config)
    ready = is_ready(checks)

    print("Day30 存活、就绪与依赖健康检查")
    print("服务：", config.app_name)
    print("数据库：", config.database_path)
    print("=" * 96)
    print("\n存活检查 /livez：")
    print(json.dumps(live_payload(config), ensure_ascii=False, indent=2))

    print("\n" + "=" * 96)
    print("依赖检查：")
    for check in checks:
        print(
            f"- {check.name:<24} {check.status.upper():<5} "
            f"critical={str(check.critical):<5} "
            f"latency={check.latency_ms:.3f} ms"
        )
        print("  ", check.detail)

    print("\n" + "=" * 96)
    print("就绪检查 /readyz：")
    print("HTTP状态码：", 200 if ready else 503)
    print("服务状态：", "ready" if ready else "not_ready")

    print("\n三种检查的职责：")
    print("/livez               → 只回答进程是否活着，不访问外部依赖")
    print("/readyz               → 关键依赖可用时才允许负载均衡器转发流量")
    print("/health/dependencies  → 给开发和运维查看每项依赖的详细原因")

    print("\n本步数据流：")
    print("HTTP请求或CLI → health handler → 各Check函数")
    print("→ CheckResult列表 → critical结果聚合")
    print("→ ready返回200；not_ready返回503；live始终独立返回200")
    return 0 if ready else 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day30健康检查练习")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("check", "serve"),
        default="check",
        help="check执行一次检查；serve启动HTTP服务。",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    try:
        config = load_config(empty_config_arguments()).config
    except ConfigurationError as error:
        print("配置加载失败：", error)
        raise SystemExit(2) from error

    if arguments.mode == "check":
        raise SystemExit(print_check_report(config))

    print("Day30健康检查HTTP服务已启动。")
    print(f"访问地址：http://{config.host}:{config.port}")
    print("存活检查：GET /livez")
    print("就绪检查：GET /readyz")
    print("依赖详情：GET /health/dependencies")
    print("按 Ctrl+C 停止服务。")
    web.run_app(
        create_app(config),
        host=config.host,
        port=config.port,
        print=None,
    )


if __name__ == "__main__":
    main()
