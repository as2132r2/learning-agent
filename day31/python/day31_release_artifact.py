"""Day31 第5步：构建可验证、可重复的发布制品。

运行：
    python day31/python/day31_release_artifact.py

指定版本：
    python day31/python/day31_release_artifact.py --version 31.0.1

安全原则：发布包只包含代码、普通配置模板和课程内置示例资料；不包含
.env、密钥、运行数据库、用户上传、备份、日志、缓存和虚拟环境。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import re
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY31_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = DAY31_DIR / "dist"
ROOT_DOTENV = PROJECT_ROOT / ".env"
ARCHIVE_ROOT = "rag-service"
MANIFEST_SCHEMA_VERSION = 1


# 明确白名单比“打包整个项目后再排除”更安全。
SOURCE_FILES: tuple[tuple[Path, str], ...] = (
    # Day30最终服务及生产化支撑模块。
    *tuple(
        (path, f"day30/python/{path.name}")
        for path in sorted((PROJECT_ROOT / "day30" / "python").glob("*.py"))
    ),
    (PROJECT_ROOT / "day30/config/app.json", "day30/config/app.json"),
    (
        PROJECT_ROOT / "day30/config/.env.example",
        "day30/config/.env.example",
    ),
    # 最终服务实际导入的Day22混合检索代码。
    *tuple(
        (path, f"day22/python/{path.name}")
        for path in sorted((PROJECT_ROOT / "day22" / "python").glob("*.py"))
    ),
    # Day22向量检索继续依赖的Day21文档与索引代码。
    *tuple(
        (path, f"day21/python/{path.name}")
        for path in sorted((PROJECT_ROOT / "day21" / "python").glob("*.py"))
    ),
    # 只发布课程内置样例，不发布uploads或uploads_validated。
    (
        PROJECT_ROOT / "day21/documents/day21_sample.txt",
        "day21/documents/day21_sample.txt",
    ),
)


RUNTIME_PACKAGES = (
    "aiohttp",
    "fastembed",
    "langchain-community",
    "langchain-core",
    "langchain-openai",
    "numpy",
    "pydantic",
    "pypdf",
    "python-dotenv",
    "sqlite-vec",
)


@dataclass(frozen=True)
class PackagedFile:
    path: str
    size_bytes: int
    sha256: str
    source: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_version(value: str) -> str:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?", value) is None:
        raise argparse.ArgumentTypeError(
            "版本格式应类似31.0.0或31.0.0-rc1。"
        )
    return value


def installed_requirements() -> bytes:
    lines: list[str] = []
    missing: list[str] = []
    for package in RUNTIME_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
        else:
            lines.append(f"{package}=={version}")
    if missing:
        raise RuntimeError(f"当前环境缺少运行依赖，不能构建：{missing}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def deployment_guide(version: str) -> bytes:
    text = f"""# RAG Service {version}

此制品只包含不可变应用代码和课程内置示例资料。

## 明确未包含

- `.env` 和任何真实密钥；
- Day30运行数据库、WAL和SHM；
- 数据库备份和日志；
- 用户上传及其向量/FTS索引；
- `.venv`、`__pycache__` 和模型缓存。

## 部署初始化顺序

1. 创建Python虚拟环境并安装 `requirements.lock.txt`；
2. 从安全密钥系统注入DEEPSEEK环境变量；
3. 挂载独立的知识库、索引和业务数据库持久卷；
4. 执行数据库迁移；
5. 创建并验证第一份数据库备份；
6. 运行发布前检查；
7. 启动 `day30/python/day30_production_rag_service.py`。

代码制品与运行数据分离，升级代码时不会把开发机数据覆盖到生产环境。
"""
    return text.encode("utf-8")


def load_real_secret_values() -> list[bytes]:
    """读取本机密钥只用于泄露扫描，永远不写入报告。"""

    if not ROOT_DOTENV.exists():
        return []
    values = dotenv_values(ROOT_DOTENV)
    secrets: list[bytes] = []
    for key, value in values.items():
        if value and any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET")):
            encoded = value.encode("utf-8")
            if len(encoded) >= 8:
                secrets.append(encoded)
    return secrets


def assert_no_secret_leak(path: str, data: bytes, secrets: list[bytes]) -> None:
    forbidden_names = {".env", "id_rsa", "id_ed25519"}
    if PurePosixPath(path).name in forbidden_names:
        raise RuntimeError(f"禁止打包敏感文件：{path}")
    for secret in secrets:
        if secret in data:
            raise RuntimeError(f"检测到本机密钥泄露到文件：{path}")


def collect_payloads(version: str) -> dict[str, tuple[bytes, str]]:
    """返回归档相对路径 → (文件内容, 来源说明)。"""

    payloads: dict[str, tuple[bytes, str]] = {}
    for source_path, archive_path in SOURCE_FILES:
        if not source_path.is_file():
            raise FileNotFoundError(f"发布白名单文件不存在：{source_path}")
        if archive_path in payloads:
            raise RuntimeError(f"归档路径重复：{archive_path}")
        payloads[archive_path] = (
            source_path.read_bytes(),
            str(source_path.relative_to(PROJECT_ROOT)),
        )

    payloads["requirements.lock.txt"] = (
        installed_requirements(),
        "generated:installed-package-versions",
    )
    payloads["DEPLOYMENT.md"] = (
        deployment_guide(version),
        "generated:deployment-guide",
    )
    return payloads


def build_internal_manifest(
    version: str,
    payloads: dict[str, tuple[bytes, str]],
) -> tuple[bytes, list[PackagedFile]]:
    entries = [
        PackagedFile(
            path=path,
            size_bytes=len(data),
            sha256=sha256_bytes(data),
            source=source,
        )
        for path, (data, source) in sorted(payloads.items())
    ]
    # build_epoch固定为0，使相同版本和相同源码能够生成相同归档字节。
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "application": "rag-service",
        "version": version,
        "build_epoch": 0,
        "file_count": len(entries),
        "files": [asdict(entry) for entry in entries],
    }
    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    return data, entries


def normalized_tar_info(path: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=f"{ARCHIVE_ROOT}/{path}")
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def write_deterministic_archive(
    output_path: Path,
    payloads: dict[str, tuple[bytes, str]],
    manifest_data: bytes,
) -> None:
    """固定tar元数据和gzip时间戳，实现字节级可重复构建。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".artifact-",
        suffix=".tar.gz",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as raw_file:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_file,
                mtime=0,
            ) as gzip_file:
                with tarfile.open(mode="w", fileobj=gzip_file) as archive:
                    all_payloads = dict(payloads)
                    all_payloads["MANIFEST.json"] = (
                        manifest_data,
                        "generated:internal-manifest",
                    )
                    for path, (data, _) in sorted(all_payloads.items()):
                        archive.addfile(
                            normalized_tar_info(path, len(data)),
                            io.BytesIO(data),
                        )
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def verify_archive(
    artifact_path: Path,
    secrets: list[bytes],
) -> tuple[int, int]:
    """重新读取归档，防止路径穿越、链接、哈希错误和密钥泄露。"""

    with tarfile.open(artifact_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        manifest_name = f"{ARCHIVE_ROOT}/MANIFEST.json"
        if manifest_name not in names:
            raise RuntimeError("归档缺少内部MANIFEST.json。")

        extracted_manifest = archive.extractfile(manifest_name)
        if extracted_manifest is None:
            raise RuntimeError("无法读取内部MANIFEST.json。")
        manifest = json.loads(extracted_manifest.read().decode("utf-8"))
        expected = {entry["path"]: entry for entry in manifest["files"]}

        verified_count = 0
        total_uncompressed = 0
        for member in members:
            pure_path = PurePosixPath(member.name)
            if pure_path.is_absolute() or ".." in pure_path.parts:
                raise RuntimeError(f"归档包含不安全路径：{member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"归档禁止包含链接：{member.name}")
            if not member.isfile() or member.name == manifest_name:
                continue

            relative = str(pure_path.relative_to(ARCHIVE_ROOT))
            file_object = archive.extractfile(member)
            if file_object is None:
                raise RuntimeError(f"无法读取归档成员：{member.name}")
            data = file_object.read()
            assert_no_secret_leak(relative, data, secrets)

            entry = expected.get(relative)
            if entry is None:
                raise RuntimeError(f"归档文件未登记在manifest：{relative}")
            if len(data) != entry["size_bytes"]:
                raise RuntimeError(f"文件大小校验失败：{relative}")
            if sha256_bytes(data) != entry["sha256"]:
                raise RuntimeError(f"文件哈希校验失败：{relative}")
            verified_count += 1
            total_uncompressed += len(data)

        if verified_count != manifest["file_count"]:
            raise RuntimeError("实际文件数与manifest不一致。")
        return verified_count, total_uncompressed


def write_external_manifest(
    path: Path,
    version: str,
    artifact_path: Path,
    file_count: int,
    uncompressed_size: int,
) -> None:
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "application": "rag-service",
        "version": version,
        "artifact": artifact_path.name,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "artifact_sha256": sha256_file(artifact_path),
        "verified_file_count": file_count,
        "uncompressed_payload_bytes": uncompressed_size,
        "excluded_runtime_data": [
            ".env and secrets",
            "day30/database",
            "day21/uploads",
            "day21/uploads_validated",
            "day21/database and day22/database indexes",
            "logs, backups, caches, virtual environments",
        ],
        "verification": "passed",
    }
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day31发布制品构建")
    parser.add_argument("--version", type=validate_version, default="31.0.0")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    artifact_path = DIST_DIR / f"rag-service-{arguments.version}.tar.gz"
    external_manifest_path = DIST_DIR / (
        f"rag-service-{arguments.version}.artifact.json"
    )

    print("Day31 构建发布制品与校验清单")
    print("版本：", arguments.version)
    print("输出目录：", DIST_DIR)
    print("=" * 100)

    secrets = load_real_secret_values()
    payloads = collect_payloads(arguments.version)
    for path, (data, _) in payloads.items():
        assert_no_secret_leak(path, data, secrets)

    internal_manifest, entries = build_internal_manifest(
        arguments.version,
        payloads,
    )
    write_deterministic_archive(artifact_path, payloads, internal_manifest)
    verified_count, uncompressed_size = verify_archive(artifact_path, secrets)
    write_external_manifest(
        external_manifest_path,
        arguments.version,
        artifact_path,
        verified_count,
        uncompressed_size,
    )

    print("白名单文件数：", len(entries))
    print("验证通过文件数：", verified_count)
    print("未压缩内容大小：", uncompressed_size, "bytes")
    print("制品大小：", artifact_path.stat().st_size, "bytes")
    print("制品SHA-256：", sha256_file(artifact_path))
    print("\n发布制品：", artifact_path)
    print("外部校验清单：", external_manifest_path)

    print("\n安全排除：")
    print("- .env与真实密钥")
    print("- 运行数据库、WAL、SHM、备份和日志")
    print("- 用户上传、现有向量索引和FTS索引")
    print("- .venv、__pycache__和模型缓存")

    print("\n可重复构建设计：")
    print("归档路径排序 + mtime=0 + uid/gid=0 + 固定权限 + gzip mtime=0")
    print("相同版本、相同源码和相同依赖版本应得到相同SHA-256。")

    print("\n本步数据流：")
    print("发布白名单 → 读取文件 → 本机真实密钥泄露扫描")
    print("→ 每文件SHA-256 → 内部MANIFEST.json → 规范化tar + gzip")
    print("→ 重新打开归档 → 路径/链接/文件数/大小/哈希/密钥校验")
    print("→ tar.gz + 外部artifact.json，可交给下一步CI/CD发布")


if __name__ == "__main__":
    main()
