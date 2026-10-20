"""Day26 第 1 步：A/B 实验的稳定分流与粘性分组。

运行：
    python day26/python/day26_stable_assignment.py

核心要求：
1. 同一用户在同一实验中始终进入同一版本；
2. 分流结果可以复现；
3. 已分配用户不会因后续权重调整突然换组；
4. 分配结果持久化，便于审计。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DAY26_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = DAY26_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "day26_experiments.db"


@dataclass(frozen=True, slots=True)
class Variant:
    """一个实验版本及其流量权重。"""

    name: str
    weight: int


@dataclass(frozen=True, slots=True)
class Experiment:
    """不可变的实验定义。"""

    experiment_id: str
    name: str
    allocation_version: str
    variants: tuple[Variant, ...]


@dataclass(frozen=True, slots=True)
class Assignment:
    """一个用户在实验中的持久化分组结果。"""

    experiment_id: str
    user_id: str
    variant: str
    bucket: int
    allocation_version: str
    assignment_source: str
    assigned_at: str


EXPERIMENT = Experiment(
    experiment_id="rag-answer-prompt-exp-001",
    name="RAG 回答提示词 v1 与 v2 对比",
    allocation_version="allocation-v1",
    variants=(
        Variant(name="control", weight=50),
        Variant(name="candidate", weight=50),
    ),
)


def open_database() -> sqlite3.Connection:
    """打开 SQLite，并启用按列名读取。"""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """创建实验定义与用户分配表。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id       TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            allocation_version  TEXT NOT NULL,
            variants_json       TEXT NOT NULL,
            status              TEXT NOT NULL
                                CHECK (status IN ('draft', 'running', 'stopped')),
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS experiment_assignments (
            experiment_id       TEXT NOT NULL,
            user_id             TEXT NOT NULL,
            variant             TEXT NOT NULL,
            bucket              INTEGER NOT NULL
                                CHECK (bucket BETWEEN 0 AND 9999),
            allocation_version  TEXT NOT NULL,
            assignment_source   TEXT NOT NULL
                                CHECK (assignment_source IN ('hash', 'database')),
            assigned_at         TEXT NOT NULL,
            PRIMARY KEY (experiment_id, user_id),
            FOREIGN KEY (experiment_id)
                REFERENCES experiments(experiment_id)
        );

        CREATE INDEX IF NOT EXISTS idx_assignments_variant
        ON experiment_assignments(experiment_id, variant);
        """
    )
    connection.commit()


def validate_experiment(experiment: Experiment) -> None:
    """检查实验定义，防止错误权重进入分流逻辑。"""

    if not experiment.experiment_id.strip():
        raise ValueError("experiment_id 不能为空。")
    if not experiment.variants:
        raise ValueError("实验至少需要一个版本。")

    names = [variant.name for variant in experiment.variants]
    if len(names) != len(set(names)):
        raise ValueError("版本名称不能重复。")
    if any(variant.weight <= 0 for variant in experiment.variants):
        raise ValueError("每个版本权重必须大于 0。")
    if sum(variant.weight for variant in experiment.variants) != 100:
        raise ValueError("版本权重总和必须等于 100。")


def save_experiment(
    connection: sqlite3.Connection,
    experiment: Experiment,
) -> None:
    """保存实验定义；已有实验只更新名称和更新时间，不改正在运行的分配快照。"""

    validate_experiment(experiment)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    variants_json = json.dumps(
        [asdict(variant) for variant in experiment.variants],
        ensure_ascii=False,
        sort_keys=True,
    )

    connection.execute(
        """
        INSERT INTO experiments (
            experiment_id, name, allocation_version,
            variants_json, status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, 'running', ?, ?)
        ON CONFLICT(experiment_id) DO UPDATE SET
            name = excluded.name,
            updated_at = excluded.updated_at
        """,
        (
            experiment.experiment_id,
            experiment.name,
            experiment.allocation_version,
            variants_json,
            now,
            now,
        ),
    )
    connection.commit()


def stable_bucket(experiment_id: str, user_id: str) -> int:
    """把用户稳定映射到 0～9999，共一万个桶。"""

    if not user_id.strip():
        raise ValueError("user_id 不能为空。")
    raw = f"{experiment_id}|{user_id}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    # 前 8 字节转成大整数，再对 10000 取模。
    return int.from_bytes(digest[:8], byteorder="big") % 10_000


def variant_for_bucket(
    bucket: int,
    variants: tuple[Variant, ...],
) -> str:
    """按照权重的累计区间把桶映射到具体版本。"""

    # 权重 50 表示 5000 个桶，权重 10 表示 1000 个桶。
    upper_bound = 0
    for variant in variants:
        upper_bound += variant.weight * 100
        if bucket < upper_bound:
            return variant.name
    raise RuntimeError("权重区间没有覆盖当前桶，请检查实验定义。")


def row_to_assignment(
    row: sqlite3.Row,
    source: str,
) -> Assignment:
    return Assignment(
        experiment_id=row["experiment_id"],
        user_id=row["user_id"],
        variant=row["variant"],
        bucket=row["bucket"],
        allocation_version=row["allocation_version"],
        assignment_source=source,
        assigned_at=row["assigned_at"],
    )


def assign_user(
    connection: sqlite3.Connection,
    experiment: Experiment,
    user_id: str,
) -> Assignment:
    """优先返回数据库旧分组；只有新用户才执行哈希分流。"""

    old_row = connection.execute(
        """
        SELECT *
        FROM experiment_assignments
        WHERE experiment_id = ? AND user_id = ?
        """,
        (experiment.experiment_id, user_id),
    ).fetchone()
    if old_row is not None:
        return row_to_assignment(old_row, source="database")

    bucket = stable_bucket(experiment.experiment_id, user_id)
    variant = variant_for_bucket(bucket, experiment.variants)
    assigned_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # INSERT OR IGNORE 可以处理两个并发请求同时给同一用户分组的情况。
    with connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO experiment_assignments (
                experiment_id, user_id, variant, bucket,
                allocation_version, assignment_source, assigned_at
            )
            VALUES (?, ?, ?, ?, ?, 'hash', ?)
            """,
            (
                experiment.experiment_id,
                user_id,
                variant,
                bucket,
                experiment.allocation_version,
                assigned_at,
            ),
        )

    # 再查一次，以数据库最终结果为准。
    final_row = connection.execute(
        """
        SELECT *
        FROM experiment_assignments
        WHERE experiment_id = ? AND user_id = ?
        """,
        (experiment.experiment_id, user_id),
    ).fetchone()
    if final_row is None:
        raise RuntimeError("用户分组保存失败。")
    return row_to_assignment(final_row, source="hash")


def assignment_statistics(
    connection: sqlite3.Connection,
    experiment_id: str,
) -> dict[str, int]:
    """统计当前实验各版本已分配用户数。"""

    rows = connection.execute(
        """
        SELECT variant, COUNT(*) AS users
        FROM experiment_assignments
        WHERE experiment_id = ?
        GROUP BY variant
        ORDER BY variant
        """,
        (experiment_id,),
    ).fetchall()
    return {row["variant"]: row["users"] for row in rows}


def main() -> None:
    print("Day26 A/B 实验稳定分流")
    print("数据库：", DATABASE_PATH)
    print("实验：", EXPERIMENT.experiment_id)
    print("分配版本：", EXPERIMENT.allocation_version)
    print("流量权重：", {item.name: item.weight for item in EXPERIMENT.variants})
    print("=" * 88)

    users = [
        "user-xiaolin",
        "user-leo",
        "user-001",
        "user-002",
        "user-003",
        "user-004",
        "user-005",
        "user-006",
    ]

    with open_database() as connection:
        create_schema(connection)
        save_experiment(connection, EXPERIMENT)

        print("第一次分配：")
        first_assignments: dict[str, Assignment] = {}
        for user_id in users:
            assignment = assign_user(connection, EXPERIMENT, user_id)
            first_assignments[user_id] = assignment
            print(
                f"- {user_id:<15} → {assignment.variant:<9} "
                f"bucket={assignment.bucket:04d} "
                f"source={assignment.assignment_source}"
            )

        print("\n" + "=" * 88)
        print("第二次请求相同用户，验证粘性分组：")
        all_stable = True
        for user_id in users:
            second = assign_user(connection, EXPERIMENT, user_id)
            first = first_assignments[user_id]
            stable = first.variant == second.variant and first.bucket == second.bucket
            all_stable = all_stable and stable
            print(
                f"- {user_id:<15} → {second.variant:<9} "
                f"bucket={second.bucket:04d} source={second.assignment_source} "
                f"稳定={stable}"
            )

        statistics = assignment_statistics(connection, EXPERIMENT.experiment_id)

    print("\n" + "=" * 88)
    print("分配统计：", statistics)
    print("所有用户重复分配均保持一致：", all_stable)

    print("\n本步数据流：")
    print("experiment_id + user_id")
    print("→ SHA-256 → 0～9999稳定bucket")
    print("→ 按权重区间选择variant")
    print("→ INSERT OR IGNORE持久化Assignment")
    print("→ 后续请求优先读取数据库 → 用户保持在原版本")

    print("\n重要：少量用户的实际比例不一定正好50/50；用户增多后才会逐渐接近配置权重。")


if __name__ == "__main__":
    main()
