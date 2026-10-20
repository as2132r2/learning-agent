"""Day32 第4步：Ed25519公私钥制品签名。

python day32/python/day32_ed25519_artifact.py
python day32/python/day32_ed25519_artifact.py --action verify

签名内容是包含制品SHA-256的规范化声明。验证时使用事先信任的公钥文件。
依赖：cryptography（当前虚拟环境已安装）。
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import secrets
import stat
import tempfile
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from day32_sbom_basic import DEFAULT_ARTIFACT, DAY32_DIR, sha256_file, write_json_atomically


PRIVATE_PATH = DAY32_DIR / "private" / "artifact-ed25519.pem"
PUBLIC_PATH = DAY32_DIR / "public" / "artifact-ed25519-public.pem"
SIGNATURE_PATH = DAY32_DIR / "signatures" / "rag-service-31.0.0.ed25519.json"
DOMAIN = b"day32-ed25519-artifact-v1\x00"


def public_bytes(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def fingerprint(public_key: Ed25519PublicKey) -> str:
    """公钥指纹可供人工核对，不包含私钥信息。"""
    return hashlib.sha256(public_bytes(public_key)).hexdigest()


def load_private_key(*, create: bool) -> Ed25519PrivateKey:
    if create:
        PRIVATE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(
                PRIVATE_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            pass
        else:
            key = Ed25519PrivateKey.generate()
            # 教学用本机PEM；文件权限600，private目录已被Git忽略。
            data = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
    metadata = PRIVATE_PATH.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise ValueError("私钥必须是权限600的普通文件。")
    key = serialization.load_pem_private_key(PRIVATE_PATH.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("私钥必须使用Ed25519。")
    return key


def ensure_public_key(private_key: Ed25519PrivateKey) -> None:
    """首次导出公钥；已有公钥不一致时拒绝覆盖。"""
    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    try:
        with PUBLIC_PATH.open("xb") as file:
            file.write(data)
    except FileExistsError:
        if public_bytes(load_public_key(PUBLIC_PATH)) != public_bytes(private_key.public_key()):
            raise ValueError("已有公钥与私钥不配对，拒绝自动替换。")


def load_public_key(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("公钥必须使用Ed25519。")
    return key


def canonical_statement(statement: dict) -> bytes:
    return DOMAIN + json.dumps(
        statement, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sign_artifact(artifact: Path, private_key: Ed25519PrivateKey) -> dict:
    statement = {
        "schema_version": 1,
        "algorithm": "Ed25519",
        "artifact_name": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
        "public_key_sha256": fingerprint(private_key.public_key()),
    }
    signature = private_key.sign(canonical_statement(statement))
    return {
        "statement": statement,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_artifact(artifact: Path, public_key: Ed25519PublicKey, envelope: dict) -> bool:
    """先验证声明签名，再核对实际文件。公钥来自独立可信路径。"""
    try:
        if not isinstance(envelope, dict) or set(envelope) != {"statement", "signature_base64"}:
            return False
        statement = envelope["statement"]
        if not isinstance(statement, dict) or set(statement) != {
            "schema_version", "algorithm", "artifact_name", "size_bytes",
            "sha256", "public_key_sha256",
        }:
            return False
        if type(statement["schema_version"]) is not int or statement["schema_version"] != 1:
            return False
        if statement["algorithm"] != "Ed25519":
            return False
        if statement["public_key_sha256"] != fingerprint(public_key):
            return False
        signature = base64.b64decode(envelope["signature_base64"], validate=True)
        public_key.verify(signature, canonical_statement(statement))
        return (
            statement["artifact_name"] == artifact.name
            and type(statement["size_bytes"]) is int
            and statement["size_bytes"] == artifact.stat().st_size
            and statement["sha256"] == sha256_file(artifact)
        )
    except (InvalidSignature, ValueError, TypeError, binascii.Error):
        return False


def demonstrate_rejection(artifact: Path, public_key: Ed25519PublicKey, envelope: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="day32-ed25519-") as directory:
        modified = Path(directory) / artifact.name
        data = artifact.read_bytes()
        if not data:
            raise ValueError("演示要求制品非空。")
        modified.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        altered_valid = verify_artifact(modified, public_key, envelope)
        wrong_key = Ed25519PrivateKey.generate().public_key()
        wrong_valid = verify_artifact(artifact, wrong_key, envelope)
        # 直接翻转签名字节，验证密码学验签失败路径。
        damaged = dict(envelope)
        raw = base64.b64decode(damaged["signature_base64"])
        damaged["signature_base64"] = base64.b64encode(
            bytes([raw[0] ^ 1]) + raw[1:]
        ).decode("ascii")
        signature_valid = verify_artifact(artifact, public_key, damaged)
        for label, valid in (
            ("篡改制品", altered_valid), ("错误公钥", wrong_valid),
            ("篡改签名", signature_valid),
        ):
            print(f"{label}：", "错误放行" if valid else "拒绝（预期）")
        if any((altered_valid, wrong_valid, signature_valid)):
            raise RuntimeError("拦截演示失败。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day32 Ed25519数字签名")
    parser.add_argument("--action", choices=("demo", "sign", "verify"), default="demo")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--signature", type=Path, default=SIGNATURE_PATH)
    parser.add_argument("--public-key", type=Path, default=PUBLIC_PATH)
    args = parser.parse_args()
    try:
        artifact = args.artifact.resolve()
        if args.action != "verify":
            private_key = load_private_key(create=True)
            ensure_public_key(private_key)
            write_json_atomically(args.signature, sign_artifact(artifact, private_key))
        # verify路径完全不读取私钥。
        public_key = load_public_key(args.public_key)
        envelope = json.loads(args.signature.read_text(encoding="utf-8"))
        print("Day32 Ed25519公钥／私钥签名")
        print("制品：", artifact)
        print("可信公钥：", args.public_key)
        print("公钥SHA-256指纹：", fingerprint(public_key))
        print("签名文件：", args.signature)
        valid = verify_artifact(artifact, public_key, envelope)
        print("原制品验签：", "通过" if valid else "拒绝")
        if not valid:
            raise SystemExit(1)
        if args.action == "demo":
            demonstrate_rejection(artifact, public_key, envelope)
    except (OSError, ValueError, TypeError) as error:
        print("处理失败：", error)
        raise SystemExit(2) from error
    print("\n签名：制品 → SHA-256 → 规范化声明 → 私钥签名 → JSON")
    print("验签：可信公钥 → 验证声明签名 → 核对制品名称、大小与SHA-256")
    print("公钥应通过可信渠道分发；验证通过表示该密钥认可这些字节。")


if __name__ == "__main__":
    main()
