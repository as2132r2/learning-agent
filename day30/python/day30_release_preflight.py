"""Day30 第 5 步：启动前检查与发布验收清单。

运行：
    python day30/python/day30_release_preflight.py

严格模式（警告也会阻止发布）：
    python day30/python/day30_release_preflight.py --strict

退出码：0=允许启动，1=检查未通过，2=配置无法加载。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from day30_config_management import (
    AppConfig,
    ConfigurationError,
    DOTENV_PATH,
    load_config,
)
from day30_database_backup import (
    BACKUP_DIR,
    BackupError,
    BackupManifest,
    inspect_database,
    load_manifest,
    sha256_file,
)
from day30_database_migrations import (
    MIGRATIONS,
    read_applied_migrations,
    validate_history,
)


# 数据库迁移文件以MIGRATIONS定义作为唯一目标版本来源。
TARGET_SCHEMA_VERSION = len(MIGRATIONS)
MAX_BACKUP_AGE_HOURS = 24
REQUIRED_PACKAGES = (
    "aiohttp",
    "langchain-core",
    "langchain-openai",
    "langgraph",
    "pydantic",
    "python-dotenv",
)


@dataclass(frozen=True)
class PreflightResult:
    number: int
    name: str
    status: str
    required: bool
    duration_ms: float
    detail: str


def empty_config_arguments() -> argparse.Namespace:
    return argparse.Namespace(
        environment=None,
        host=None,
        port=None,
        log_level=None,
        request_timeout_seconds=None,
        max_concurrency=None,
    )


def run_check(
    number: int,
    name: str,
    required: bool,
    function: Callable[[], tuple[str, str]],
) -> PreflightResult:
    """检查函数返回(status, detail)，异常统一视为FAIL。"""

    started = time.perf_counter()
    try:
        status, detail = function()
        if status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"检查返回了未知状态：{status}")
    except Exception as error:
        status = "FAIL"
        detail = f"{type(error).__name__}: {error}"
    return PreflightResult(
        number=number,
        name=name,
        status=status,
        required=required,
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        detail=detail,
    )


def check_python_runtime() -> tuple[str, str]:
    current = sys.version_info
    if current < (3, 11):
        return "FAIL", f"当前Python={platform.python_version()}，要求>=3.11"
    return "PASS", f"Python={platform.python_version()}，executable={sys.executable}"


def check_packages() -> tuple[str, str]:
    versions: list[str] = []
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            versions.append(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        return "FAIL", f"缺少依赖：{missing}"
    return "PASS", "；".join(versions)


def check_configuration(config: AppConfig) -> tuple[str, str]:
    # config能构造出来，说明第1步的类型、范围和密钥规则已经通过。
    return (
        "PASS",
        f"environment={config.environment}，host={config.host}，"
        f"port={config.port}，model={config.model}，api_key=********",
    )


def check_environment_policy(config: AppConfig) -> tuple[str, str]:
    if config.environment != "production":
        return (
            "WARN",
            f"当前environment={config.environment}；正式发布前应明确改为production。",
        )
    if config.host in {"127.0.0.1", "localhost"}:
        return "WARN", "生产环境只监听本机地址，外部流量无法直接访问。"
    return "PASS", "生产环境标识和监听地址符合发布要求。"


def check_secret_file_permissions() -> tuple[str, str]:
    if not DOTENV_PATH.exists():
        return "FAIL", f"密钥文件不存在：{DOTENV_PATH}"
    mode = DOTENV_PATH.stat().st_mode & 0o777
    if mode & 0o077:
        return (
            "WARN",
            f".env权限={oct(mode)}，建议执行 chmod 600 .env，避免其他用户读取。",
        )
    return "PASS", f".env权限={oct(mode)}；未输出任何密钥原文。"


def check_database(config: AppConfig) -> tuple[str, str]:
    snapshot = inspect_database(config.database_path)
    if snapshot.schema_version != TARGET_SCHEMA_VERSION:
        return (
            "FAIL",
            f"数据库版本={snapshot.schema_version}，程序目标={TARGET_SCHEMA_VERSION}",
        )
    return (
        "PASS",
        f"integrity={snapshot.integrity}，schema={snapshot.schema_version}，"
        f"rows={snapshot.table_counts}",
    )


def check_migration_history(config: AppConfig) -> tuple[str, str]:
    connection = sqlite3.connect(
        f"file:{config.database_path}?mode=ro",
        uri=True,
        timeout=3,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        # 正式检查只读，不调用create_migration_table，不偷偷修复缺失状态。
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "schema_migrations" not in tables:
            return "FAIL", "缺少schema_migrations迁移历史表。"
        applied = read_applied_migrations(connection)
        validate_history(applied, MIGRATIONS)
        return "PASS", f"迁移版本连续且checksum有效：{sorted(applied)}"
    finally:
        connection.close()


def latest_manifest() -> Path:
    manifests = sorted(
        BACKUP_DIR.glob("*.manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise BackupError("没有找到数据库备份清单，请先执行第4步。")
    return manifests[0]


def check_latest_backup() -> tuple[str, str]:
    manifest_path = latest_manifest()
    manifest: BackupManifest = load_manifest(manifest_path)
    backup_path = manifest_path.parent / manifest.backup_file
    if not backup_path.is_file():
        return "FAIL", f"清单对应的备份文件不存在：{backup_path.name}"
    if backup_path.stat().st_size != manifest.size_bytes:
        return "FAIL", "备份大小与清单不一致。"
    if sha256_file(backup_path) != manifest.sha256:
        return "FAIL", "备份SHA-256与清单不一致。"

    snapshot = inspect_database(backup_path, immutable=True)
    if snapshot.schema_version != manifest.schema_version:
        return "FAIL", "备份Schema版本与清单不一致。"
    if snapshot.table_counts != manifest.table_counts:
        return "FAIL", "备份关键表行数与清单不一致。"

    created_at = datetime.fromisoformat(manifest.created_at)
    age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
    detail = (
        f"backup={backup_path.name}，age={age_hours:.2f}小时，"
        f"sha256={manifest.sha256[:12]}…"
    )
    if age_hours > MAX_BACKUP_AGE_HOURS:
        return "WARN", detail + f"；超过{MAX_BACKUP_AGE_HOURS}小时。"
    return "PASS", detail


def check_writable_directories(config: AppConfig) -> tuple[str, str]:
    directories = {config.database_path.parent, BACKUP_DIR}
    checked: list[str] = []
    for directory in sorted(directories, key=str):
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".preflight-",
            dir=directory,
        ) as file:
            file.write(b"day30-preflight")
            file.flush()
            os.fsync(file.fileno())
        checked.append(str(directory))
    return "PASS", "目录可写：" + "，".join(checked)


def check_port(config: AppConfig) -> tuple[str, str]:
    """只校验端口取值；真正绑定由第6步服务负责。"""

    if not 1 <= config.port <= 65535:
        return "FAIL", f"端口超出范围：{config.port}"
    return "PASS", f"计划监听{config.host}:{config.port}；启动时还会实际绑定。"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day30发布前检查")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把WARN也视为阻止发布。",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    try:
        config = load_config(empty_config_arguments()).config
    except ConfigurationError as error:
        print("配置加载失败：", error)
        raise SystemExit(2) from error

    definitions: list[tuple[str, bool, Callable[[], tuple[str, str]]]] = [
        ("Python运行时", True, check_python_runtime),
        ("Python依赖包", True, check_packages),
        ("类型化配置与密钥", True, lambda: check_configuration(config)),
        ("环境发布策略", False, lambda: check_environment_policy(config)),
        ("密钥文件权限", False, check_secret_file_permissions),
        ("数据库完整性与Schema", True, lambda: check_database(config)),
        ("迁移历史与校验和", True, lambda: check_migration_history(config)),
        ("最新备份有效性", True, check_latest_backup),
        ("数据目录写权限", True, lambda: check_writable_directories(config)),
        ("服务监听参数", True, lambda: check_port(config)),
    ]
    results = [
        run_check(index, name, required, function)
        for index, (name, required, function) in enumerate(definitions, start=1)
    ]

    failures = [result for result in results if result.status == "FAIL"]
    warnings = [result for result in results if result.status == "WARN"]
    blocked = bool(failures) or (arguments.strict and bool(warnings))

    print("Day30 启动前检查与发布验收清单")
    print("模式：", "严格（WARN也阻止）" if arguments.strict else "普通")
    print("=" * 110)
    for result in results:
        requirement = "必须" if result.required else "建议"
        print(
            f"[{result.status:<4}] {result.number:02d}. {result.name:<22} "
            f"{requirement:<4} {result.duration_ms:>8.3f} ms"
        )
        print("       ", result.detail)

    print("\n" + "=" * 110)
    print(
        f"统计：PASS={sum(r.status == 'PASS' for r in results)}，"
        f"WARN={len(warnings)}，FAIL={len(failures)}"
    )
    print("发布结论：", "允许启动" if not blocked else "禁止启动")
    print("进程退出码：", 0 if not blocked else 1)

    print("\n判定规则：")
    print("- 必须项FAIL：始终禁止启动")
    print("- 建议项WARN：普通模式允许，--strict模式禁止")
    print("- 检查程序只验证状态，不自动修改配置、迁移数据库或恢复备份")

    print("\n本步数据流：")
    print("运行环境 + 配置 + 密钥权限 + 数据库 + 迁移历史 + 最新备份")
    print("→ 10项PreflightResult → PASS/WARN/FAIL聚合")
    print("→ 发布结论 + 进程退出码 → 启动脚本或CI/CD决定是否继续")
    raise SystemExit(1 if blocked else 0)


if __name__ == "__main__":
    main()
