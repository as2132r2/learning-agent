"""Day11 第六步：统一运行测试并生成质量门禁报告。"""

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    command: list[str]
    passed: bool
    return_code: int
    elapsed_seconds: float
    output_tail: str


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

CHECKS = [
    (
        "单元测试",
        PROJECT_DIRECTORY
        / "day11"
        / "python"
        / "day11_unit_test_basic.py"
    ),
    (
        "Mock测试",
        PROJECT_DIRECTORY
        / "day11"
        / "python"
        / "day11_mock_test.py"
    ),
    (
        "LangGraph测试",
        PROJECT_DIRECTORY
        / "day11"
        / "python"
        / "day11_graph_test.py"
    ),
    (
        "RAG集成测试",
        PROJECT_DIRECTORY
        / "day11"
        / "python"
        / "day11_rag_integration_test.py"
    ),
    (
        "RAG回归评测",
        PROJECT_DIRECTORY
        / "day11"
        / "python"
        / "day11_regression_eval.py"
    ),
    (
        "故障注入测试",
        PROJECT_DIRECTORY
        / "day11"
        / "python"
        / "day11_failure_injection.py"
    ),
    (
        "异步测试",
        PROJECT_DIRECTORY
        / "day11"
        / "python"
        / "day11_async_test.py"
    )
]


def run_check(name: str, script_path: Path) -> CheckResult:
    command = [PYTHON, str(script_path)]
    started_at = time.perf_counter()

    completed = subprocess.run(
        command,
        cwd=PROJECT_DIRECTORY,
        text=True,
        capture_output=True,
        check=False
    )

    elapsed_seconds = time.perf_counter() - started_at
    output = (completed.stdout + completed.stderr).strip()

    return CheckResult(
        name=name,
        command=command,
        passed=completed.returncode == 0,
        return_code=completed.returncode,
        elapsed_seconds=round(elapsed_seconds, 3),
        output_tail=output[-500:]
    )


print("开始执行 Day11 统一质量门禁……")
results: list[CheckResult] = []

for name, script_path in CHECKS:
    print(f"\n[执行] {name}")
    result = run_check(name, script_path)
    results.append(result)

    status = "通过" if result.passed else "失败"
    print(
        f"[结果] {status} | "
        f"耗时={result.elapsed_seconds:.3f}s | "
        f"return_code={result.return_code}"
    )

    if not result.passed:
        print("输出末尾：")
        print(result.output_tail)


passed_count = sum(1 for result in results if result.passed)
report = {
    "project": "langchain-learning",
    "day": 11,
    "passed": passed_count == len(results),
    "passed_count": passed_count,
    "total_count": len(results),
    "checks": [asdict(result) for result in results]
}

report_path = (
    PROJECT_DIRECTORY
    / "day11"
    / "database"
    / "day11_quality_gate.json"
)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("质量门禁汇总")
print(f"通过：{passed_count}/{len(results)}")
print("报告：", report_path)

if not report["passed"]:
    print("质量门禁失败：禁止继续发布当前版本。")
    raise SystemExit(1)

print("质量门禁通过：当前版本可以继续。")


# ============================================================
# 数据流
# ============================================================
#
# 多个独立检查脚本
#       ↓ subprocess.run()
# CheckResult
#       ↓
# 全部通过？
#   ├── 是 → passed=True → 可以继续
#   └── 否 → return_code=1 → 阻止继续
#       ↓
# day11_quality_gate.json
