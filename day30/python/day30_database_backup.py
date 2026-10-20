"""Day30 第 4 步：SQLite备份、校验与恢复演练。

运行：
    python day30/python/day30_database_backup.py

程序不会覆盖正式数据库。恢复演练在临时目录中完成并自动清理。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DAY30_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = DAY30_DIR / "database" / "day30_production.db"
BACKUP_DIR = DAY30_DIR / "database" / "backups"
EXPECTED_SCHEMA_VERSION = 3
REQUIRED_TABLES = ("schema_migrations", "rag_requests", "rag_sources")


class BackupError(RuntimeError):
    """备份、校验或恢复演练不符合要求。"""


@dataclass(frozen=True)
class DatabaseSnapshot:
    """某个SQLite文件在检查时的关键状态。"""

    integrity: str
    schema_version: int
    table_counts: dict[str, int]


@dataclass(frozen=True)
class BackupManifest:
    """与备份文件一起保存的可审计清单。"""

    format_version: int
    created_at: str
    source_database: str
    backup_file: str
    size_bytes: int
    sha256: str
    schema_version: int
    table_counts: dict[str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    """分块读取文件，避免大型数据库一次性进入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect_database(
    path: Path,
    *,
    immutable: bool = False,
) -> DatabaseSnapshot:
    """以只读方式检查数据库完整性、版本和关键表行数。"""

    if not path.is_file():
        raise BackupError(f"数据库文件不存在：{path}")

    query = "mode=ro&immutable=1" if immutable else "mode=ro"
    connection = sqlite3.connect(f"file:{path}?{query}", uri=True, timeout=5)
    try:
        integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
        integrity = "; ".join(str(row[0]) for row in integrity_rows)
        if integrity.lower() != "ok":
            raise BackupError(f"数据库完整性检查失败：{integrity}")

        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        existing_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = sorted(set(REQUIRED_TABLES) - existing_tables)
        if missing:
            raise BackupError(f"数据库缺少必需表：{missing}")

        table_counts = {
            table: connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            for table in REQUIRED_TABLES
        }
        return DatabaseSnapshot(
            integrity=integrity,
            schema_version=schema_version,
            table_counts=table_counts,
        )
    finally:
        connection.close()


def create_consistent_backup(source_path: Path, backup_dir: Path) -> Path:
    """使用SQLite Online Backup API获得一致性快照，再原子发布文件。"""

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final_path = backup_dir / f"day30-production-{timestamp}.db"

    # 临时文件与最终文件处于同一目录，os.replace可以原子地发布备份。
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".backup-in-progress-",
        suffix=".db",
        dir=backup_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True, timeout=5)
    destination = sqlite3.connect(temporary_path)
    try:
        source.backup(destination, pages=100)
        destination.commit()
    except Exception:
        destination.close()
        source.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        destination.close()
        source.close()

    os.replace(temporary_path, final_path)
    return final_path


def build_manifest(
    source_path: Path,
    backup_path: Path,
    snapshot: DatabaseSnapshot,
) -> BackupManifest:
    return BackupManifest(
        format_version=1,
        created_at=utc_now(),
        source_database=str(source_path.resolve()),
        backup_file=backup_path.name,
        size_bytes=backup_path.stat().st_size,
        sha256=sha256_file(backup_path),
        schema_version=snapshot.schema_version,
        table_counts=snapshot.table_counts,
    )


def write_manifest(backup_path: Path, manifest: BackupManifest) -> Path:
    manifest_path = backup_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_manifest(path: Path) -> BackupManifest:
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return BackupManifest(**data)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise BackupError(f"备份清单无效：{error}") from error


def validate_backup(
    backup_path: Path,
    manifest_path: Path,
    source_snapshot: DatabaseSnapshot,
) -> DatabaseSnapshot:
    """同时验证文件哈希、SQLite内部完整性和业务数据规模。"""

    manifest = load_manifest(manifest_path)
    actual_size = backup_path.stat().st_size
    actual_hash = sha256_file(backup_path)
    if manifest.backup_file != backup_path.name:
        raise BackupError("清单记录的文件名与备份文件不一致。")
    if manifest.size_bytes != actual_size:
        raise BackupError("备份文件大小与清单不一致。")
    if manifest.sha256 != actual_hash:
        raise BackupError("备份SHA-256与清单不一致，文件可能损坏。")

    # 备份文件已经封存，不会再变化；immutable模式不会生成-wal/-shm旁文件。
    backup_snapshot = inspect_database(backup_path, immutable=True)
    if backup_snapshot.schema_version != EXPECTED_SCHEMA_VERSION:
        raise BackupError(
            f"备份Schema版本={backup_snapshot.schema_version}，"
            f"程序要求={EXPECTED_SCHEMA_VERSION}。"
        )
    if backup_snapshot != source_snapshot:
        raise BackupError("备份快照与备份前的源数据库状态不一致。")
    if manifest.schema_version != backup_snapshot.schema_version:
        raise BackupError("清单Schema版本与备份数据库不一致。")
    if manifest.table_counts != backup_snapshot.table_counts:
        raise BackupError("清单行数与备份数据库不一致。")
    return backup_snapshot


def restore_to_path(backup_path: Path, restored_path: Path) -> None:
    """从已校验备份恢复到指定的新数据库，不覆盖源库。"""

    source = sqlite3.connect(
        f"file:{backup_path}?mode=ro&immutable=1",
        uri=True,
        timeout=5,
    )
    destination = sqlite3.connect(restored_path)
    try:
        source.backup(destination)
        destination.commit()
    finally:
        destination.close()
        source.close()


def perform_restore_drill(
    source_database_path: Path,
    backup_path: Path,
    expected_snapshot: DatabaseSnapshot,
) -> tuple[DatabaseSnapshot, int, int]:
    """临时恢复、写入隔离记录，并证明正式数据库没有被修改。"""

    source_count_before = expected_snapshot.table_counts["rag_requests"]
    with tempfile.TemporaryDirectory(prefix="day30-restore-drill-") as directory:
        restored_path = Path(directory) / "restored.db"
        restore_to_path(backup_path, restored_path)
        restored_snapshot = inspect_database(restored_path, immutable=True)
        if restored_snapshot != expected_snapshot:
            raise BackupError("恢复后的数据库状态与备份不一致。")

        connection = sqlite3.connect(restored_path)
        try:
            connection.execute(
                """
                INSERT INTO rag_requests
                    (request_id, question, status, created_at, trace_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "req-restore-drill-only",
                    "这条记录是否只存在于恢复演练数据库？",
                    "pending",
                    utc_now(),
                    "trace-restore-drill",
                ),
            )
            connection.commit()
            restored_count_after = connection.execute(
                "SELECT COUNT(*) FROM rag_requests"
            ).fetchone()[0]
        finally:
            connection.close()

    source_count_after = inspect_database(source_database_path).table_counts[
        "rag_requests"
    ]
    if source_count_after != source_count_before:
        raise BackupError("恢复演练意外修改了正式数据库。")
    return restored_snapshot, restored_count_after, source_count_after


def main() -> None:
    print("Day30 SQLite备份、校验与恢复演练")
    print("源数据库：", DATABASE_PATH)
    print("备份目录：", BACKUP_DIR)
    print("=" * 100)

    started = time.perf_counter()
    source_snapshot = inspect_database(DATABASE_PATH)
    print("备份前完整性：", source_snapshot.integrity)
    print("Schema版本：", source_snapshot.schema_version)
    print("关键表行数：", source_snapshot.table_counts)

    backup_path = create_consistent_backup(DATABASE_PATH, BACKUP_DIR)
    manifest = build_manifest(DATABASE_PATH, backup_path, source_snapshot)
    manifest_path = write_manifest(backup_path, manifest)

    print("\n" + "=" * 100)
    print("备份已经原子发布：")
    print("数据库备份：", backup_path)
    print("备份清单：", manifest_path)
    print("文件大小：", manifest.size_bytes, "bytes")
    print("SHA-256：", manifest.sha256)

    backup_snapshot = validate_backup(
        backup_path,
        manifest_path,
        source_snapshot,
    )
    print("\n备份校验：通过")
    print("备份完整性：", backup_snapshot.integrity)
    print("备份关键表行数：", backup_snapshot.table_counts)

    restored, restored_count, source_count = perform_restore_drill(
        DATABASE_PATH,
        backup_path,
        backup_snapshot,
    )
    print("\n" + "=" * 100)
    print("隔离恢复演练：通过")
    print("恢复后完整性：", restored.integrity)
    print("恢复库加入测试记录后的rag_requests行数：", restored_count)
    print("正式库rag_requests行数：", source_count)
    print("临时恢复目录已经自动清理，正式数据库未被覆盖。")

    print("\n总耗时：", round(time.perf_counter() - started, 3), "秒")
    print("\n本步数据流：")
    print("正式SQLite → integrity_check → SQLite Backup API → 临时文件")
    print("→ os.replace原子发布 → SHA-256 + manifest → 二次完整性校验")
    print("→ 临时目录恢复 → 行数与Schema对比 → 隔离写入验证 → 自动清理")


if __name__ == "__main__":
    main()
