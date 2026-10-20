"""Day31 第6步：完整CI/CD发布流水线。

运行：
    python day31/python/day31_ci_cd_pipeline.py

指定发布版本：
    python day31/python/day31_ci_cd_pipeline.py --version 31.0.1

流水线采用fail-fast：任一阶段失败，后续阶段和发布提升都不会执行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY31_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = DAY31_DIR / "python"
DIST_DIR = DAY31_DIR / "dist"
RELEASES_DIR = DAY31_DIR / "releases"
LOG_ROOT = DAY31_DIR / "logs" / "ci-runs"
LATEST_REPORT_PATH = DAY31_DIR / "logs" / "day31_ci_pipeline_report.json"
STAGE_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class StageDefinition:
    name: str
    description: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class StageResult:
    name: str
    description: str
    status: str
    exit_code: int | None
    duration_seconds: float
    command: list[str]
    log_path: str
    error: str | None


@dataclass(frozen=True)
class PromotionResult:
    status: str
    version: str
    release_directory: str | None
    artifact: str | None
    artifact_sha256: str | None
    detail: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def validate_version(value: str) -> str:
    import re

    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?", value) is None:
        raise argparse.ArgumentTypeError("版本格式应类似31.0.0或31.0.0-rc1。")
    return value


def build_stages(version: str) -> list[StageDefinition]:
    """顺序就是发布依赖关系，不能随意交换。"""

    return [
        StageDefinition(
            name="quality_gate",
            description="静态编译、配置、迁移、健康与发布前检查",
            command=(
                sys.executable,
                str(PYTHON_DIR / "day31_quality_gate.py"),
            ),
        ),
        StageDefinition(
            name="api_contracts",
            description="验证HTTP状态码、字段名称、类型和错误行为",
            command=(
                sys.executable,
                str(PYTHON_DIR / "day31_api_contract_tests.py"),
            ),
        ),
        StageDefinition(
            name="fake_integration",
            description="验证真实RAG控制流、守门、持久化和失败路径",
            command=(
                sys.executable,
                str(PYTHON_DIR / "day31_fake_integration_tests.py"),
            ),
        ),
        StageDefinition(
            name="load_slo",
            description="并发压测并验证成功率、P95、吞吐量和持久化",
            command=(
                sys.executable,
                str(PYTHON_DIR / "day31_load_slo_test.py"),
            ),
        ),
        StageDefinition(
            name="build_artifact",
            description="构建无密钥、可重复并经过哈希验证的发布制品",
            command=(
                sys.executable,
                str(PYTHON_DIR / "day31_release_artifact.py"),
                "--version",
                version,
            ),
        ),
    ]


def ci_environment(run_id: str, version: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CI": "true",
            "PYTHONUNBUFFERED": "1",
            "DAY31_CI_RUN_ID": run_id,
            "DAY31_RELEASE_VERSION": version,
        }
    )
    return environment


def write_stage_log(
    path: Path,
    definition: StageDefinition,
    stdout: str,
    stderr: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"stage: {definition.name}",
            f"description: {definition.description}",
            "command: " + " ".join(definition.command),
            "",
            "================ STDOUT ================",
            stdout or "（无输出）",
            "",
            "================ STDERR ================",
            stderr or "（无输出）",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def run_stage(
    definition: StageDefinition,
    run_directory: Path,
    environment: dict[str, str],
) -> StageResult:
    started = time.perf_counter()
    log_path = run_directory / f"{definition.name}.log"
    try:
        completed = subprocess.run(
            definition.command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=STAGE_TIMEOUT_SECONDS,
            check=False,
        )
        duration = time.perf_counter() - started
        write_stage_log(
            log_path,
            definition,
            completed.stdout,
            completed.stderr,
        )
        return StageResult(
            name=definition.name,
            description=definition.description,
            status="passed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
            duration_seconds=round(duration, 3),
            command=list(definition.command),
            log_path=str(log_path.relative_to(PROJECT_ROOT)),
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
        write_stage_log(log_path, definition, stdout, stderr)
        return StageResult(
            name=definition.name,
            description=definition.description,
            status="failed",
            exit_code=None,
            duration_seconds=round(duration, 3),
            command=list(definition.command),
            log_path=str(log_path.relative_to(PROJECT_ROOT)),
            error=f"阶段超过{STAGE_TIMEOUT_SECONDS}秒，已终止。",
        )
    except OSError as error:
        duration = time.perf_counter() - started
        write_stage_log(log_path, definition, "", str(error))
        return StageResult(
            name=definition.name,
            description=definition.description,
            status="failed",
            exit_code=None,
            duration_seconds=round(duration, 3),
            command=list(definition.command),
            log_path=str(log_path.relative_to(PROJECT_ROOT)),
            error=f"无法启动阶段：{error}",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def copy_file_atomically(source: Path, target: Path) -> None:
    """复制到同目录临时文件，校验后再原子提升。"""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary_path)
        if sha256_file(source) != sha256_file(temporary_path):
            raise RuntimeError(f"制品复制后哈希不一致：{source.name}")
        os.replace(temporary_path, target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def promote_release(version: str) -> PromotionResult:
    """只提升已经构建验证的制品；同版本不同哈希禁止覆盖。"""

    artifact = DIST_DIR / f"rag-service-{version}.tar.gz"
    artifact_manifest = DIST_DIR / f"rag-service-{version}.artifact.json"
    if not artifact.is_file() or not artifact_manifest.is_file():
        return PromotionResult(
            status="failed",
            version=version,
            release_directory=None,
            artifact=None,
            artifact_sha256=None,
            detail="构建阶段没有生成完整制品和外部清单。",
        )

    manifest = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    actual_hash = sha256_file(artifact)
    if manifest.get("artifact_sha256") != actual_hash:
        return PromotionResult(
            status="failed",
            version=version,
            release_directory=None,
            artifact=artifact.name,
            artifact_sha256=actual_hash,
            detail="制品哈希与artifact.json不一致，拒绝提升。",
        )

    release_directory = RELEASES_DIR / version
    release_directory.mkdir(parents=True, exist_ok=True)
    target_artifact = release_directory / artifact.name
    target_manifest = release_directory / artifact_manifest.name

    if target_artifact.exists():
        existing_hash = sha256_file(target_artifact)
        if existing_hash != actual_hash:
            return PromotionResult(
                status="failed",
                version=version,
                release_directory=str(release_directory),
                artifact=target_artifact.name,
                artifact_sha256=actual_hash,
                detail="同一版本已存在不同哈希，版本不可变策略拒绝覆盖。",
            )
        detail = "相同版本和哈希已经发布，本次幂等跳过复制。"
    else:
        copy_file_atomically(artifact, target_artifact)
        copy_file_atomically(artifact_manifest, target_manifest)
        detail = "制品和校验清单已原子提升到发布区。"

    return PromotionResult(
        status="promoted",
        version=version,
        release_directory=str(release_directory.relative_to(PROJECT_ROOT)),
        artifact=target_artifact.name,
        artifact_sha256=actual_hash,
        detail=detail,
    )


def write_json_atomically(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day31 CI/CD发布流水线")
    parser.add_argument("--version", type=validate_version, default="31.0.0")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_directory = LOG_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    environment = ci_environment(run_id, arguments.version)
    stages = build_stages(arguments.version)

    print("Day31 完整CI/CD发布流水线")
    print("运行编号：", run_id)
    print("发布版本：", arguments.version)
    print("阶段数量：", len(stages))
    print("失败策略：fail-fast")
    print("=" * 104)

    started_at = utc_now()
    wall_started = time.perf_counter()
    results: list[StageResult] = []
    failed_stage: str | None = None

    for index, stage in enumerate(stages, start=1):
        print(f"\n[{index}/{len(stages)}] {stage.name}")
        print("说明：", stage.description)
        result = run_stage(stage, run_directory, environment)
        results.append(result)
        print("状态：", result.status)
        print("退出码：", result.exit_code)
        print("耗时：", result.duration_seconds, "秒")
        print("日志：", result.log_path)
        if result.error:
            print("错误：", result.error)
        if result.status != "passed":
            failed_stage = result.name
            print("fail-fast：停止执行后续阶段，不允许发布提升。")
            break

    if failed_stage is None:
        promotion = promote_release(arguments.version)
    else:
        promotion = PromotionResult(
            status="skipped",
            version=arguments.version,
            release_directory=None,
            artifact=None,
            artifact_sha256=None,
            detail=f"阶段{failed_stage}失败，未执行发布提升。",
        )

    pipeline_passed = (
        len(results) == len(stages)
        and all(result.status == "passed" for result in results)
        and promotion.status == "promoted"
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "version": arguments.version,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_time_seconds": round(time.perf_counter() - wall_started, 3),
        "pipeline_status": "passed" if pipeline_passed else "failed",
        "failed_stage": failed_stage,
        "stage_count_expected": len(stages),
        "stage_count_executed": len(results),
        "stages": [asdict(result) for result in results],
        "promotion": asdict(promotion),
    }
    run_report_path = run_directory / "pipeline-report.json"
    write_json_atomically(run_report_path, report)
    write_json_atomically(LATEST_REPORT_PATH, report)

    print("\n" + "=" * 104)
    print("发布提升状态：", promotion.status)
    print("发布说明：", promotion.detail)
    if promotion.release_directory:
        print("发布目录：", promotion.release_directory)
    if promotion.artifact_sha256:
        print("制品SHA-256：", promotion.artifact_sha256)
    print("流水线最终状态：", report["pipeline_status"])
    print("实际墙钟耗时：", report["wall_time_seconds"], "秒")
    print("本次报告：", run_report_path)
    print("最新报告：", LATEST_REPORT_PATH)

    print("\n本步数据流：")
    print("质量门禁 → API契约 → Fake端到端集成 → 并发SLO → 制品构建")
    print("→ 全阶段exit_code均为0 → 校验artifact SHA-256")
    print("→ 检查同版本不可变规则 → 原子复制到releases/<version>")
    print("→ 每阶段独立日志 + pipeline-report.json + CI最终退出码")

    raise SystemExit(0 if pipeline_passed else 1)


if __name__ == "__main__":
    main()
