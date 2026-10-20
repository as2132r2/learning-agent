"""Day32 第6步：带供应链门禁的本地发布流水线。

python day32/python/day32_supply_chain_pipeline.py
python day32/python/day32_supply_chain_pipeline.py --scan-mode online --version 32.0.0

demo为默认模式，虚构扫描风险会阻断签名与发布。online查询失败同样阻断。
只扫描制品声明的直接依赖，尚不代表完整供应链审计。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "day31/python"))

from day31_ci_cd_pipeline import (  # noqa: E402
    build_stages, ci_environment, run_stage, validate_version,
)
from day32_dependency_scan import demo_scan, online_scan  # noqa: E402
from day32_ed25519_artifact import (  # noqa: E402
    PUBLIC_PATH, ensure_public_key, load_private_key, sign_artifact,
)
from day32_sbom_basic import (  # noqa: E402
    DAY32_DIR, build_sbom, sha256_file, write_json_atomically,
)
from day32_signature_gate import POLICY_PATH, evaluate_gate  # noqa: E402


def promote_bundle(version: str, files: list[Path], expected_hash: str) -> Path:
    """在同一文件系统临时目录完成复制和校验，最后发布整个目录。"""
    releases = DAY32_DIR / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    target = releases / version
    if target.exists():
        raise ValueError("该版本已经发布，请使用新版本号；不覆盖既有发布。")
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=releases) as directory:
        bundle = Path(directory) / "bundle"
        bundle.mkdir()
        for source in files:
            destination = bundle / source.name
            shutil.copyfile(source, destination)
            if sha256_file(source) != sha256_file(destination):
                raise ValueError(f"复制校验失败：{source.name}")
        if sha256_file(bundle / files[0].name) != expected_hash:
            raise ValueError("待发布制品与已扫描制品哈希不同。")
        # 公钥不从制品中取，仍使用本地受信任公钥和策略。
        check = evaluate_gate(
            bundle / files[0].name, bundle / files[1].name,
            PUBLIC_PATH, POLICY_PATH,
        )
        if check["decision"] != "allow":
            raise ValueError("复制到发布区后的制品未通过验签。")
        bundle.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Day32供应链发布流水线")
    parser.add_argument("--version", type=validate_version, default="32.0.0")
    parser.add_argument("--scan-mode", choices=("demo", "online"), default="demo")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = DAY32_DIR / "logs" / "ci-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "run_id": run_id, "version": args.version,
        "scan_mode": args.scan_mode, "status": "running",
        "stages": [], "promotion": "skipped",
        "coverage": "declared_direct_dependencies_only",
    }
    exit_code = 1
    print("Day32 供应链门禁CI/CD流水线")
    print("版本：", args.version, "扫描模式：", args.scan_mode)
    print("运行目录：", run_dir)
    try:
        # 复用Day31测试与构建定义，不调用Day31的发布提升函数。
        for definition in build_stages(args.version):
            result = run_stage(
                definition, run_dir, ci_environment(run_id, args.version)
            )
            report["stages"].append(asdict(result))
            print(f"[{definition.name}] {result.status}")
            if result.status != "passed":
                raise ValueError(f"基础阶段失败：{definition.name}；详见{result.log_path}")

        # 将构建结果复制到当前运行目录，后续都处理这份制品快照。
        built = PROJECT_ROOT / "day31/dist" / f"rag-service-{args.version}.tar.gz"
        artifact = run_dir / built.name
        shutil.copyfile(built, artifact)
        sbom = build_sbom(artifact)
        expected_hash = sbom["artifact"]["sha256"]
        sbom_path = run_dir / "sbom.json"
        write_json_atomically(sbom_path, sbom)
        report["stages"].append({"name": "sbom", "status": "passed"})
        print("[sbom] passed")

        results = (
            demo_scan(sbom["dependencies"])
            if args.scan_mode == "demo" else online_scan(sbom["dependencies"])
        )
        unknown = any(item["status"] == "unknown" for item in results)
        findings = any(item["findings"] for item in results)
        scan_allowed = args.scan_mode == "online" and not unknown and not findings
        scan = {
            "mode": args.scan_mode, "is_demo": args.scan_mode == "demo",
            "artifact_sha256": expected_hash, "results": results,
            "decision": "allow" if scan_allowed else "deny",
        }
        scan_path = run_dir / "dependency-scan.json"
        write_json_atomically(scan_path, scan)
        report["stages"].append({
            "name": "dependency_scan", "status": "passed" if scan_allowed else "failed"
        })
        print("[dependency_scan]", scan["decision"])
        if not scan_allowed:
            raise ValueError("演示模式、风险条目或unknown阻断发布；详见dependency-scan.json。")

        if sha256_file(artifact) != expected_hash:
            raise ValueError("扫描后的制品内容发生变化。")
        private_key = load_private_key(create=False)
        ensure_public_key(private_key)
        signature_path = run_dir / "artifact.ed25519.json"
        write_json_atomically(signature_path, sign_artifact(artifact, private_key))
        report["stages"].append({"name": "sign", "status": "passed"})

        gate = evaluate_gate(artifact, signature_path, PUBLIC_PATH, POLICY_PATH)
        gate_path = run_dir / "signature-gate.json"
        write_json_atomically(gate_path, gate)
        report["stages"].append({
            "name": "signature_gate", "status": "passed" if gate["decision"] == "allow" else "failed"
        })
        if gate["decision"] != "allow":
            raise ValueError("发布前验签失败。")
        target = promote_bundle(
            args.version,
            [artifact, signature_path, sbom_path, scan_path, gate_path],
            expected_hash,
        )
        report.update(status="passed", promotion="promoted", release_directory=str(target))
        exit_code = 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print("流水线停止：", report["error"])
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomically(run_dir / "pipeline-report.json", report)
        write_json_atomically(DAY32_DIR / "reports/day32-supply-chain-pipeline.json", report)
    print("最终状态：", report["status"], "发布提升：", report["promotion"])
    print("报告：", run_dir / "pipeline-report.json")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
