"""Day32 第1步：从已发布制品生成教学用软件物料清单。

运行：python day32/python/day32_sbom_basic.py
此JSON使用课程自定义格式，不宣称符合CycloneDX或SPDX标准。
依赖仅限制品requirements中声明的直接依赖，不代表完整传递依赖树。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAY32_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    PROJECT_ROOT / "day31/releases/31.0.0/rag-service-31.0.0.tar.gz"
)


def sha256_file(path: Path) -> str:
    """分块计算整个制品的内容指纹。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_verified_members(artifact: Path) -> tuple[dict, dict[str, bytes]]:
    """读取归档而不解压，并验证内部清单对应的每个文件。"""
    members: dict[str, bytes] = {}
    with tarfile.open(artifact, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"归档路径不安全：{member.name}")
            if not member.isfile():
                raise ValueError(f"只接受普通文件：{member.name}")
            if not path.parts or path.parts[0] != "rag-service":
                raise ValueError("归档根目录必须为rag-service。")
            relative = str(path.relative_to("rag-service"))
            if relative in members:
                raise ValueError(f"归档路径重复：{relative}")
            # 当前制品很小；限制读取体积，避免异常归档耗尽内存。
            if member.size > 10 * 1024 * 1024:
                raise ValueError("归档单文件超过10MiB。")
            if sum(len(data) for data in members.values()) + member.size > 50 * 1024 * 1024:
                raise ValueError("归档总内容超过50MiB。")
            file = archive.extractfile(member)
            if file is None:
                raise ValueError(f"无法读取：{relative}")
            members[relative] = file.read()

    manifest = json.loads(members.pop("MANIFEST.json"))
    entries = manifest["files"]
    expected = {entry["path"]: entry for entry in entries}
    if len(expected) != len(entries) or set(expected) != set(members):
        raise ValueError("内部清单文件集合与实际文件不一致。")
    if manifest["file_count"] != len(members):
        raise ValueError("内部清单文件数不一致。")
    for path, data in members.items():
        entry = expected[path]
        if len(data) != entry["size_bytes"]:
            raise ValueError(f"大小校验失败：{path}")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"哈希校验失败：{path}")
    return manifest, members


def parse_dependencies(data: bytes) -> list[dict[str, str]]:
    """从制品内requirements读取版本，不读取本机pip环境。"""
    packages: dict[str, str] = {}
    for raw_line in data.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise ValueError(f"依赖行必须精确固定版本：{line}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in packages:
            raise ValueError(f"重复声明依赖：{name}")
        packages[name] = match[2]
    return [
        {"name": name, "version": version, "ecosystem": "PyPI", "scope": "direct"}
        for name, version in sorted(packages.items())
    ]


def build_sbom(artifact: Path) -> dict:
    manifest, members = read_verified_members(artifact)
    dependencies = parse_dependencies(members["requirements.lock.txt"])
    return {
        "format": "day32-teaching-sbom",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "application": {
            "name": manifest["application"],
            "version": manifest["version"],
        },
        "artifact": {
            "filename": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "dependency_source": "artifact:requirements.lock.txt",
        "dependency_coverage": "declared_direct_only",
        "dependencies": dependencies,
        "files": [
            {
                "path": path,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for path, data in sorted(members.items())
        ],
        "verification": "internal_manifest_verified",
        "limitations": [
            "课程自定义格式；不是标准CycloneDX/SPDX文档。",
            "未解析传递依赖、许可证和依赖包分发文件哈希。",
            "内部哈希一致性不能证明发布者身份；后续步骤增加签名验证。",
        ],
    }


def write_json_atomically(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Day32教学用SBOM生成")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    arguments = parser.parse_args()
    artifact = arguments.artifact.resolve()
    sbom = build_sbom(artifact)
    output = DAY32_DIR / "reports" / f"{artifact.name.removesuffix('.tar.gz')}.sbom.json"
    write_json_atomically(output, sbom)

    print("Day32 软件物料清单基础")
    print("应用：", sbom["application"])
    print("制品：", artifact)
    print("制品SHA-256：", sbom["artifact"]["sha256"])
    print("内部文件校验：通过")
    print("清单文件数：", len(sbom["files"]))
    print("直接依赖数：", len(sbom["dependencies"]))
    print("\n制品声明的依赖：")
    for package in sbom["dependencies"]:
        print(f"- {package['name']}=={package['version']}")
    print("\nSBOM文件：", output)
    print("\n本步数据流：")
    print("已发布tar.gz → 内部MANIFEST逐文件校验 → requirements精确版本解析")
    print("→ 应用身份 + 制品哈希 + 直接依赖 + 文件哈希 → 原子写入SBOM JSON")
    print("\n当前只覆盖声明的直接依赖，完整依赖树将在后续工具化时扩展。")


if __name__ == "__main__":
    main()
