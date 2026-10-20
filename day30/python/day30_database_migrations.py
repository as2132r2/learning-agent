"""Day30 第 2 步：SQLite Schema 版本与数据库迁移。

运行：
    python day30/python/day30_database_migrations.py

再次运行不会重复修改表结构，而是验证已有迁移记录。
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DAY30_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = DAY30_DIR / "database" / "day30_production.db"


class MigrationError(RuntimeError):
    """迁移版本、校验和或执行过程不符合要求。"""


@dataclass(frozen=True)
class Migration:
    """一份不可变的数据库迁移定义。"""

    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        """SQL内容指纹，用于发现已经发布的迁移被修改。"""

        canonical = "\n".join(statement.strip() for statement in self.statements)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="create_rag_requests",
        statements=(
            """
            CREATE TABLE rag_requests (
                request_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT,
                status TEXT NOT NULL
                    CHECK (status IN ('pending', 'success', 'failed')),
                created_at TEXT NOT NULL
            )
            """,
        ),
    ),
    Migration(
        version=2,
        name="add_trace_id",
        statements=(
            "ALTER TABLE rag_requests ADD COLUMN trace_id TEXT",
            "CREATE INDEX idx_rag_requests_trace_id ON rag_requests(trace_id)",
        ),
    ),
    Migration(
        version=3,
        name="create_rag_sources",
        statements=(
            """
            CREATE TABLE rag_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                relevance_score REAL,
                FOREIGN KEY (request_id)
                    REFERENCES rag_requests(request_id)
                    ON DELETE CASCADE,
                UNIQUE (request_id, chunk_id)
            )
            """,
            "CREATE INDEX idx_rag_sources_request_id ON rag_sources(request_id)",
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def open_database(path: Path | str) -> sqlite3.Connection:
    """创建连接，并启用生产环境常用的SQLite选项。"""

    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if path != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


def create_migration_table(connection: sqlite3.Connection) -> None:
    """该管理表记录哪些迁移已经成功完成。"""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            execution_ms REAL NOT NULL
        )
        """
    )


def read_applied_migrations(
    connection: sqlite3.Connection,
) -> dict[int, sqlite3.Row]:
    rows = connection.execute(
        "SELECT * FROM schema_migrations ORDER BY version"
    ).fetchall()
    return {row["version"]: row for row in rows}


def validate_history(
    applied: dict[int, sqlite3.Row],
    migrations: tuple[Migration, ...],
) -> None:
    """启动前检查版本连续性和历史SQL是否被偷偷修改。"""

    expected_versions = [migration.version for migration in migrations]
    if expected_versions != list(range(1, len(migrations) + 1)):
        raise MigrationError("代码中的迁移版本必须从1开始连续递增。")

    applied_versions = sorted(applied)
    if applied_versions != list(range(1, len(applied_versions) + 1)):
        raise MigrationError(f"数据库迁移版本不连续：{applied_versions}")

    definitions = {migration.version: migration for migration in migrations}
    for version, row in applied.items():
        migration = definitions.get(version)
        if migration is None:
            raise MigrationError(
                f"数据库版本v{version}高于当前程序认识的版本，禁止降级启动。"
            )
        if row["name"] != migration.name:
            raise MigrationError(f"迁移v{version}名称与数据库记录不一致。")
        if row["checksum"] != migration.checksum:
            raise MigrationError(
                f"迁移v{version}校验和不一致；已发布迁移不能直接修改。"
            )


def apply_one_migration(
    connection: sqlite3.Connection,
    migration: Migration,
) -> float:
    """结构变化和版本记录在同一个事务里，要么全成功，要么全回滚。"""

    started = time.perf_counter()
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in migration.statements:
            connection.execute(statement)

        execution_ms = (time.perf_counter() - started) * 1000
        connection.execute(
            """
            INSERT INTO schema_migrations
                (version, name, checksum, applied_at, execution_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                migration.version,
                migration.name,
                migration.checksum,
                utc_now(),
                execution_ms,
            ),
        )
        connection.execute(f"PRAGMA user_version = {migration.version}")
        connection.execute("COMMIT")
        return execution_ms
    except Exception:
        connection.execute("ROLLBACK")
        raise


def migrate_database(
    connection: sqlite3.Connection,
    migrations: tuple[Migration, ...] = MIGRATIONS,
) -> tuple[int, int]:
    """验证历史并依次执行待处理迁移，返回执行前后版本。"""

    create_migration_table(connection)
    applied = read_applied_migrations(connection)
    validate_history(applied, migrations)
    before_version = max(applied, default=0)

    for migration in migrations:
        if migration.version in applied:
            print(f"[跳过] v{migration.version} {migration.name}（已经执行）")
            continue

        print(f"[迁移] v{migration.version} {migration.name}")
        try:
            elapsed_ms = apply_one_migration(connection, migration)
        except Exception as error:
            raise MigrationError(
                f"迁移v{migration.version}失败，事务已经回滚：{error}"
            ) from error
        print(f"       完成，耗时 {elapsed_ms:.3f} ms")

    final_applied = read_applied_migrations(connection)
    validate_history(final_applied, migrations)
    after_version = max(final_applied, default=0)
    return before_version, after_version


def seed_and_read_example(connection: sqlite3.Connection) -> sqlite3.Row:
    """写入一条演示数据，证明最新Schema可以正常使用。"""

    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            INSERT INTO rag_requests
                (request_id, question, answer, status, created_at, trace_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                answer = excluded.answer,
                status = excluded.status,
                trace_id = excluded.trace_id
            """,
            (
                "req-day30-demo-001",
                "什么是数据库迁移？",
                "数据库迁移是按版本演进Schema的过程。",
                "success",
                utc_now(),
                "trace-day30-demo-001",
            ),
        )
        connection.execute(
            """
            INSERT INTO rag_sources
                (request_id, chunk_id, source_path, relevance_score)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(request_id, chunk_id) DO UPDATE SET
                relevance_score = excluded.relevance_score
            """,
            (
                "req-day30-demo-001",
                "chunk-migration-001",
                "day30/notes/database_migration.md",
                0.98,
            ),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise

    return connection.execute(
        """
        SELECT r.request_id, r.status, r.trace_id,
               s.chunk_id, s.source_path, s.relevance_score
        FROM rag_requests AS r
        JOIN rag_sources AS s ON s.request_id = r.request_id
        WHERE r.request_id = ?
        """,
        ("req-day30-demo-001",),
    ).fetchone()


def demonstrate_atomic_rollback() -> tuple[bool, int]:
    """在内存数据库中故意失败，避免污染正式演示数据库。"""

    connection = open_database(":memory:")
    try:
        migrate_database(connection)
        broken = Migration(
            version=4,
            name="broken_demo",
            statements=(
                "CREATE TABLE should_be_rolled_back (id INTEGER PRIMARY KEY)",
                "INSERT INTO table_that_does_not_exist VALUES (1)",
            ),
        )
        try:
            apply_one_migration(connection, broken)
        except sqlite3.Error:
            pass

        table_exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'should_be_rolled_back'
            """
        ).fetchone() is not None
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        return table_exists, version
    finally:
        connection.close()


def print_history(connection: sqlite3.Connection) -> None:
    print("\n迁移历史：")
    for row in connection.execute(
        "SELECT * FROM schema_migrations ORDER BY version"
    ):
        print(
            f"- v{row['version']} | {row['name']} | "
            f"checksum={row['checksum'][:12]} | {row['execution_ms']:.3f} ms"
        )


def main() -> None:
    print("Day30 SQLite Schema版本与数据库迁移")
    print("数据库：", DATABASE_PATH)
    print("程序目标版本：", len(MIGRATIONS))
    print("=" * 88)

    connection = open_database(DATABASE_PATH)
    try:
        before, after = migrate_database(connection)
        print("\n迁移前版本：", before)
        print("迁移后版本：", after)
        print("PRAGMA user_version：", connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0])
        print_history(connection)

        example = seed_and_read_example(connection)
        print("\n" + "=" * 88)
        print("最新Schema读写验证：")
        print(dict(example))
    finally:
        connection.close()

    table_exists, failure_version = demonstrate_atomic_rollback()
    print("\n" + "=" * 88)
    print("故意失败迁移的原子性验证（使用内存数据库）：")
    print("失败迁移创建的表是否残留：", table_exists)
    print("失败后的数据库版本：", failure_version)
    print("结论：SQL结构修改和版本记录一起回滚。")

    print("\n" + "=" * 88)
    print("本步数据流：")
    print("Migration定义 → checksum校验 → 查询schema_migrations")
    print("→ BEGIN IMMEDIATE → 顺序执行SQL → 写入迁移记录 → COMMIT")
    print("→ 失败时ROLLBACK；再次启动时已完成版本只校验、不重复执行")


if __name__ == "__main__":
    main()
