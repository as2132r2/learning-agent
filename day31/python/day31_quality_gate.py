"""Day31 第1步：自动化质量门禁。

运行：
    python day31/python/day31_quality_gate.py

显示每项完整输出：
    python day31/python/day31_quality_gate.py --show-output

退出码：
    0 = 所有门禁通过
    1 = 至少一个门禁失败
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY30_PYTHON = PROJECT_ROOT / "day30" / "python"
DAY31_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = DAY31_DIR / "logs" / "day31_quality_gate_report.json"
COMMAND_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class GateDefinition:
    """一项门禁的静态定义。"""

    name: str
    description: str
    command: tuple[str, ...]
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS


@dataclass(frozen=True)
class GateResult:
    """一项门禁的实际执行结果。"""

    name: str
    description: str
    status: str
    command: list[str]
    exit_code: int | None
    duration_seconds: float
    stdout: str
    stderr: str
    error: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def discover_day30_python_files() -> list[Path]:
    """动态寻找Day30代码，新增文件也会自动进入编译门禁。"""

    files = sorted(DAY30_PYTHON.glob("*.py"))
    if not files:
        raise RuntimeError(f"没有找到Day30 Python文件：{DAY30_PYTHON}")
    return files


def build_gate_definitions() -> list[GateDefinition]:
    python_files = discover_day30_python_files()
    return [
        GateDefinition(
            name="python_compile",
            description=f"编译检查Day30的{len(python_files)}个Python文件",
            command=(
                sys.executable,
                "-m",
                "py_compile",
                *(str(path) for path in python_files),
            ),
        ),
        GateDefinition(
            name="configuration",
            description="验证分层配置、类型约束和密钥隔离",
            command=(
                sys.executable,
                str(DAY30_PYTHON / "day30_config_management.py"),
            ),
        ),
        GateDefinition(
            name="database_migrations",
            description="验证迁移历史、checksum、幂等性和失败回滚",
            command=(
                sys.executable,
                str(DAY30_PYTHON / "day30_database_migrations.py"),
            ),
        ),
        GateDefinition(
            name="health_readiness",
            description="验证存活、数据库、Schema和模型配置检查",
            command=(
                sys.executable,
                str(DAY30_PYTHON / "day30_health_checks.py"),
            ),
        ),
        GateDefinition(
            name="release_preflight",
            description="执行普通模式发布前验收；FAIL会阻止通过",
            command=(
                sys.executable,
                str(DAY30_PYTHON / "day30_release_preflight.py"),
            ),
        ),
    ]


def safe_environment() -> dict[str, str]:
    """继承运行环境，但强制关闭Python输出缓冲且不打印密钥。"""

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def run_gate(definition: GateDefinition) -> GateResult:
    """在独立子进程中执行门禁，隔离SystemExit和全局状态。"""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            definition.command,
            cwd=PROJECT_ROOT,
            env=safe_environment(),
            text=True,
            capture_output=True,
            timeout=definition.timeout_seconds,
            check=False,
        )
        duration = time.perf_counter() - started
        return GateResult(
            name=definition.name,
            description=definition.description,
            status="passed" if completed.returncode == 0 else "failed",
            command=list(definition.command),
            exit_code=completed.returncode,
            duration_seconds=round(duration, 3),
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=None,
        )
    except subprocess.TimeoutExpired as error:
        duration = time.perf_counter() - started
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return GateResult(
            name=definition.name,
            description=definition.description,
            status="failed",
            command=list(definition.command),
            exit_code=None,
            duration_seconds=round(duration, 3),
            stdout=stdout,
            stderr=stderr,
            error=f"超过{definition.timeout_seconds}秒，子进程已终止。",
        )
    except OSError as error:
        duration = time.perf_counter() - started
        return GateResult(
            name=definition.name,
            description=definition.description,
            status="failed",
            command=list(definition.command),
            exit_code=None,
            duration_seconds=round(duration, 3),
            stdout="",
            stderr="",
            error=f"无法启动命令：{error}",
        )


def tail_text(text: str, line_count: int = 5) -> str:
    """终端默认只显示末尾几行，完整输出仍保存在JSON报告。"""

    lines = text.rstrip().splitlines()
    if not lines:
        return "（无输出）"
    selected = lines[-line_count:]
    prefix = "……\n" if len(lines) > line_count else ""
    return prefix + "\n".join(selected)


def build_report(results: list[GateResult], started_at: str) -> dict[str, object]:
    passed = sum(result.status == "passed" for result in results)
    failed = len(results) - passed
    return {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "python_executable": sys.executable,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "quality_gate": "passed" if failed == 0 else "failed",
            "total_duration_seconds": round(
                sum(result.duration_seconds for result in results), 3
            ),
        },
        "results": [asdict(result) for result in results],
    }


def write_report_atomically(report: dict[str, object]) -> None:
    """先写临时文件，再原子替换，避免留下半份JSON。"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".quality-gate-",
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day31自动化质量门禁")
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="显示每项命令的完整stdout和stderr。",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    definitions = build_gate_definitions()
    started_at = utc_now()

    print("Day31 自动化质量门禁")
    print("项目目录：", PROJECT_ROOT)
    print("Python：", sys.executable)
    print("门禁数量：", len(definitions))
    print("=" * 108)

    results: list[GateResult] = []
    for index, definition in enumerate(definitions, start=1):
        print(f"\n[{index}/{len(definitions)}] {definition.name}")
        print("说明：", definition.description)
        result = run_gate(definition)
        results.append(result)

        label = "通过" if result.status == "passed" else "失败"
        print("结果：", label)
        print("退出码：", result.exit_code)
        print("耗时：", result.duration_seconds, "秒")

        if arguments.show_output:
            print("stdout：")
            print(result.stdout or "（无输出）")
            print("stderr：")
            print(result.stderr or "（无输出）")
        elif result.status == "failed":
            print("stdout末尾：")
            print(tail_text(result.stdout))
            print("stderr末尾：")
            print(tail_text(result.stderr))

        if result.error:
            print("门禁错误：", result.error)

    report = build_report(results, started_at)
    write_report_atomically(report)
    summary = report["summary"]
    assert isinstance(summary, dict)

    print("\n" + "=" * 108)
    print("质量门禁汇总：")
    print("通过：", summary["passed"])
    print("失败：", summary["failed"])
    print("总耗时：", summary["total_duration_seconds"], "秒")
    print("最终结论：", summary["quality_gate"])
    print("机器可读报告：", REPORT_PATH)

    print("\n本步数据流：")
    print("GateDefinition列表 → subprocess.run逐项启动独立Python进程")
    print("→ 捕获exit_code/stdout/stderr/耗时 → GateResult列表")
    print("→ 汇总质量结论 → 临时JSON → os.replace原子发布报告")
    print("→ 任一门禁失败时，本程序退出码为1，可阻止后续CI发布")

    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
