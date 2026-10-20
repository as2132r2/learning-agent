"""Day32 第5步：固定信任公钥的发布验签门禁。

python day32/python/day32_signature_gate.py

退出码0：验签通过；1：拒绝。此门禁不生成密钥、不重新签名、不提升制品。
release_trust.json必须由发布方独立维护，不能从待验证签名自动生成。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from day32_ed25519_artifact import (
    PUBLIC_PATH, SIGNATURE_PATH, fingerprint, load_public_key, verify_artifact,
)
from day32_sbom_basic import (
    DAY32_DIR, DEFAULT_ARTIFACT, read_verified_members, write_json_atomically,
)


POLICY_PATH = DAY32_DIR / "config" / "release_trust.json"
REPORT_PATH = DAY32_DIR / "reports" / "day32-signature-gate.json"


def load_trust_policy(path: Path) -> str:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version", "algorithm", "trusted_public_key_sha256"
    }:
        raise ValueError("信任策略格式无效。")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("信任策略版本不支持。")
    if policy["algorithm"] != "Ed25519":
        raise ValueError("信任策略算法不支持。")
    expected = policy["trusted_public_key_sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("公钥指纹格式无效。")
    return expected


def evaluate_gate(
    artifact: Path,
    signature_path: Path,
    public_key_path: Path,
    policy_path: Path,
) -> dict:
    """顺序执行检查；首次失败停止，未执行项明确标记skipped。"""
    checks = [
        {"name": name, "status": "skipped", "detail": "前置检查尚未通过。"}
        for name in (
            "trust_policy", "pinned_public_key", "signature_and_artifact",
            "internal_manifest",
        )
    ]
    index = 0
    try:
        expected = load_trust_policy(policy_path)
        checks[0].update(status="passed", detail="独立信任策略格式有效。")

        index = 1
        public_key = load_public_key(public_key_path)
        actual = fingerprint(public_key)
        if actual != expected:
            raise ValueError("公钥与固定指纹不一致，拒绝信任替换后的公钥。")
        checks[1].update(status="passed", detail=f"公钥指纹匹配：{actual}")

        index = 2
        envelope = json.loads(signature_path.read_text(encoding="utf-8"))
        if not verify_artifact(artifact, public_key, envelope):
            raise ValueError("签名无效，或制品名称、大小、SHA-256不匹配。")
        checks[2].update(status="passed", detail="Ed25519签名与实际制品一致。")

        index = 3
        manifest, members = read_verified_members(artifact)
        checks[3].update(
            status="passed",
            detail=f"内部清单校验通过；版本={manifest['version']}，文件数={len(members)}。",
        )
    except Exception as error:
        checks[index].update(
            status="failed", detail=f"{type(error).__name__}: {error}"
        )

    allowed = all(check["status"] == "passed" for check in checks)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": str(artifact),
        "signature_file": str(signature_path),
        "public_key_file": str(public_key_path),
        "trust_policy_file": str(policy_path),
        "checks": checks,
        "decision": "allow" if allowed else "deny",
        "exit_code": 0 if allowed else 1,
        "scope": "signature_gate_only_not_full_release_approval",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Day32发布验签门禁")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--signature", type=Path, default=SIGNATURE_PATH)
    parser.add_argument("--public-key", type=Path, default=PUBLIC_PATH)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = evaluate_gate(
        args.artifact.resolve(), args.signature.resolve(),
        args.public_key.resolve(), args.policy.resolve(),
    )
    write_json_atomically(args.report, report)
    print("Day32 发布前强制验签门禁")
    for check in report["checks"]:
        print(f"- {check['name']}: {check['status']} | {check['detail']}")
    print("\n验签门禁决定：", report["decision"])
    print("退出码：", report["exit_code"])
    print("报告：", args.report)
    print("\n信任策略 → 公钥指纹 → 声明签名与制品哈希 → 内部清单 → allow/deny")
    print("此处allow仅表示验签通过，依赖风险等其他门禁仍须通过。")
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
