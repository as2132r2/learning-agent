"""Day12 第八步：实现liveness和readiness健康检查。"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class HealthReport:
    status: str
    checks: list[CheckResult]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": [asdict(check) for check in self.checks]
        }


def check_configuration() -> CheckResult:
    required = [
        "DEEPSEEK_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL"
    ]
    missing = [name for name in required if not os.getenv(name)]

    if missing:
        return CheckResult(
            name="configuration",
            status="failed",
            detail="缺少：" + ", ".join(missing)
        )

    return CheckResult(
        name="configuration",
        status="ok",
        detail="模型配置和密钥已存在"
    )


def check_database(database_path: Path) -> CheckResult:
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS health_check ("
                "id INTEGER PRIMARY KEY, checked_at TEXT)"
            )
            connection.execute(
                "INSERT OR REPLACE INTO health_check (id, checked_at) "
                "VALUES (1, CURRENT_TIMESTAMP)"
            )
    except sqlite3.Error as error:
        return CheckResult(
            name="database",
            status="failed",
            detail=str(error)
        )

    return CheckResult(
        name="database",
        status="ok",
        detail=str(database_path)
    )


def check_model_endpoint() -> CheckResult:
    base_url = os.getenv("DEEPSEEK_BASE_URL", "")

    if not base_url.startswith(("http://", "https://")):
        return CheckResult(
            name="model_endpoint",
            status="failed",
            detail="BASE_URL格式不正确"
        )

    # 就绪检查只验证配置格式，不在健康接口中发起模型请求。
    return CheckResult(
        name="model_endpoint",
        status="ok",
        detail="URL格式正确，未主动发送网络请求"
    )


def run_readiness_checks(
    database_path: Path,
    extra_checks: list[Callable[[], CheckResult]] | None = None
) -> HealthReport:
    """readiness：检查服务是否具备接收流量的条件。"""

    checks = [
        check_configuration(),
        check_database(database_path),
        check_model_endpoint()
    ]

    for extra_check in extra_checks or []:
        checks.append(extra_check())

    overall_status = (
        "ok"
        if all(check.status == "ok" for check in checks)
        else "failed"
    )

    return HealthReport(overall_status, checks)


def run_liveness_check() -> HealthReport:
    """liveness：只确认Python进程还能执行代码。"""

    return HealthReport(
        status="ok",
        checks=[CheckResult(
            name="process",
            status="ok",
            detail="Python进程正在运行"
        )]
    )


load_dotenv()
project_directory = Path(__file__).resolve().parent.parent.parent
health_database = (
    project_directory
    / "day12"
    / "database"
    / "day12_health_check.db"
)

print("存活检查 liveness：")
print(run_liveness_check().to_dict())

print("\n就绪检查 readiness：")
readiness_report = run_readiness_checks(health_database)
print(readiness_report.to_dict())

print("\n健康状态：", readiness_report.status)


# ============================================================
# 数据流
# ============================================================
#
# GET /health/live
#     ↓
# 进程是否还能执行？
#     └── ok → 保留/重启容器的判断依据
#
# GET /health/ready
#     ↓
# 配置 + 数据库 + 依赖格式检查
#     ├── 全部ok → 允许接收流量
#     └── 任一failed → 暂停接收流量
#
# liveness失败通常表示进程需要重启；
# readiness失败通常表示进程还活着，但暂时不应接收新请求。
