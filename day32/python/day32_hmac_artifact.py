"""Day32 第3步：HMAC-SHA256制品签名与验证。

python day32/python/day32_hmac_artifact.py
python day32/python/day32_hmac_artifact.py --action sign
python day32/python/day32_hmac_artifact.py --action verify

默认demo签名并验证，再在临时目录演示篡改和错误密钥。
HMAC使用共享密钥：持有验证密钥的人同样可以生成签名。
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
from pathlib import Path

from day32_sbom_basic import DEFAULT_ARTIFACT, DAY32_DIR, write_json_atomically


KEY_PATH = DAY32_DIR / "private" / "artifact-hmac.key"
SIGNATURE_PATH = DAY32_DIR / "signatures" / "rag-service-31.0.0.hmac.json"
KEY_SIZE = 32
DOMAIN = b"day32-artifact-hmac-v1\x00"


def load_key(path: Path, *, create: bool = False) -> bytes:
    """只在签名时允许创建32字节随机密钥；现有密钥绝不覆盖。"""
    if create:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as file:
                file.write(secrets.token_bytes(KEY_SIZE))
                file.flush()
                os.fsync(file.fileno())

    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("密钥路径必须是普通文件，不能是符号链接。")
    if metadata.st_mode & 0o077:
        raise ValueError("密钥权限过宽，请将该密钥文件权限设为600。")
    key = path.read_bytes()
    if len(key) != KEY_SIZE:
        raise ValueError("密钥长度必须是32字节。")
    return key


def canonical_metadata(metadata: dict) -> bytes:
    """排序并固定JSON编码，签名与验证双方必须使用相同规则。"""
    return json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def calculate_tag(artifact: Path, key: bytes, metadata: dict) -> str:
    """同时认证元数据和完整制品内容，分块读取以控制内存。"""
    authenticator = hmac.new(key, digestmod=hashlib.sha256)
    authenticator.update(DOMAIN)
    authenticator.update(canonical_metadata(metadata))
    authenticator.update(b"\x00")
    with artifact.open("rb") as file:
        while block := file.read(1024 * 1024):
            authenticator.update(block)
    return authenticator.hexdigest()


def sign_artifact(artifact: Path, key: bytes) -> dict:
    metadata = {
        "schema_version": 1,
        "algorithm": "HMAC-SHA256",
        "artifact_name": artifact.name,
        "size_bytes": artifact.stat().st_size,
    }
    return {
        "metadata": metadata,
        "tag": calculate_tag(artifact, key, metadata),
    }


def verify_artifact(artifact: Path, key: bytes, signature: dict) -> bool:
    """未知格式拒绝；使用compare_digest比较认证码。"""
    if not isinstance(signature, dict) or set(signature) != {"metadata", "tag"}:
        return False
    metadata = signature["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version", "algorithm", "artifact_name", "size_bytes"
    }:
        return False
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        return False
    if metadata["algorithm"] != "HMAC-SHA256":
        return False
    if metadata["artifact_name"] != artifact.name:
        return False
    if type(metadata["size_bytes"]) is not int:
        return False
    if metadata["size_bytes"] != artifact.stat().st_size:
        return False
    tag = signature["tag"]
    if not isinstance(tag, str) or len(tag) != 64:
        return False
    if any(character not in "0123456789abcdef" for character in tag):
        return False
    actual_tag = calculate_tag(artifact, key, metadata)
    return hmac.compare_digest(actual_tag, tag)


def demonstrate_tampering(artifact: Path, key: bytes, signature: dict) -> None:
    """改动临时副本的一个字节，保留文件名和大小，确保检查认证码。"""
    with tempfile.TemporaryDirectory(prefix="day32-hmac-") as directory:
        altered = Path(directory) / artifact.name
        original_bytes = artifact.read_bytes()
        if not original_bytes:
            raise ValueError("演示要求制品非空。")
        altered.write_bytes(
            bytes([original_bytes[0] ^ 1]) + original_bytes[1:]
        )
        tampered_passed = verify_artifact(altered, key, signature)
        wrong_key_passed = verify_artifact(
            artifact, bytes([key[0] ^ 1]) + key[1:], signature
        )
        print("修改一个字节后的验签：", "通过" if tampered_passed else "拒绝（预期）")
        print("使用错误密钥的验签：", "通过" if wrong_key_passed else "拒绝（预期）")
        if tampered_passed or wrong_key_passed:
            raise RuntimeError("篡改拦截演示失败。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day32 HMAC制品签名")
    parser.add_argument("--action", choices=("demo", "sign", "verify"), default="demo")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--signature", type=Path, default=SIGNATURE_PATH)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    print("Day32 HMAC-SHA256制品签名基础")
    print("制品：", artifact)
    try:
        key = load_key(KEY_PATH, create=args.action != "verify")
        print("本机密钥：", KEY_PATH, "（内容不显示）")
        if args.action != "verify":
            signature = sign_artifact(artifact, key)
            write_json_atomically(args.signature, signature)
            print("签名文件：", args.signature)
        # 从磁盘重新读取，验证实际保存的签名文件。
        signature = json.loads(args.signature.read_text(encoding="utf-8"))
        valid = verify_artifact(artifact, key, signature)
        print("原制品验签：", "通过" if valid else "拒绝")
        if not valid:
            raise SystemExit(1)
        if args.action == "demo":
            demonstrate_tampering(artifact, key, signature)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("处理失败：", error)
        raise SystemExit(2) from error
    print("\n数据流：共享密钥 + 元数据 + 制品字节 → HMAC tag → 签名JSON")
    print("验证：重新计算HMAC → compare_digest → 通过或拒绝")
    print("HMAC不加密制品；签名文件可分发，共享密钥必须私下保管。")


if __name__ == "__main__":
    main()
