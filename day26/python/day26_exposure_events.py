"""Day26 第 3 步：记录实验曝光、回答结果和用户反馈事件。

运行：
    python day26/python/day26_exposure_events.py

本步使用模拟请求，不调用大模型。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from day26_stable_assignment import (
    EXPERIMENT,
    assign_user,
    create_schema,
    open_database,
    save_experiment,
)
from day26_variant_config import (
    VARIANT_CONFIGS,
    create_config_schema,
    freeze_variant_configs,
    load_frozen_config,
)


OutcomeStatus = Literal["success", "rejected", "error", "cancelled"]
Rating = Literal["up", "down", "neutral"]


@dataclass(frozen=True, slots=True)
class OutcomeInput:
    """一次实验请求完成后的客观运行结果。"""

    request_id: str
    status: OutcomeStatus
    quality_passed: bool
    latency_ms: int
    model_calls: int
    total_tokens: int
    cache_hits: int
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class FeedbackEventInput:
    """用户对一次已曝光请求给出的主观评价。"""

    request_id: str
    user_id: str
    rating: Rating
    reason: str | None = None


def create_event_schema(connection: sqlite3.Connection) -> None:
    """创建曝光、结果和反馈事件表。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiment_exposures (
            request_id          TEXT PRIMARY KEY,
            experiment_id       TEXT NOT NULL,
            allocation_version  TEXT NOT NULL,
            user_id             TEXT NOT NULL,
            variant             TEXT NOT NULL,
            bucket              INTEGER NOT NULL,
            prompt_version      TEXT NOT NULL,
            config_hash         TEXT NOT NULL,
            exposed_at          TEXT NOT NULL,
            FOREIGN KEY (experiment_id, user_id)
                REFERENCES experiment_assignments(experiment_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS experiment_outcomes (
            request_id       TEXT PRIMARY KEY,
            status           TEXT NOT NULL
                             CHECK (status IN (
                                'success', 'rejected', 'error', 'cancelled'
                             )),
            quality_passed   INTEGER NOT NULL CHECK (quality_passed IN (0, 1)),
            latency_ms       INTEGER NOT NULL CHECK (latency_ms >= 0),
            model_calls      INTEGER NOT NULL CHECK (model_calls >= 0),
            total_tokens     INTEGER NOT NULL CHECK (total_tokens >= 0),
            cache_hits       INTEGER NOT NULL CHECK (cache_hits >= 0),
            error_type       TEXT,
            completed_at     TEXT NOT NULL,
            FOREIGN KEY (request_id)
                REFERENCES experiment_exposures(request_id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS experiment_feedback_events (
            request_id       TEXT NOT NULL,
            user_id          TEXT NOT NULL,
            rating           TEXT NOT NULL
                             CHECK (rating IN ('up', 'down', 'neutral')),
            reason           TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            PRIMARY KEY (request_id, user_id),
            FOREIGN KEY (request_id)
                REFERENCES experiment_exposures(request_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_exposures_variant
        ON experiment_exposures(experiment_id, allocation_version, variant);

        CREATE INDEX IF NOT EXISTS idx_outcomes_status
        ON experiment_outcomes(status);

        CREATE INDEX IF NOT EXISTS idx_experiment_feedback_rating
        ON experiment_feedback_events(rating);
        """
    )
    connection.commit()


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_exposure(
    connection: sqlite3.Connection,
    request_id: str,
    user_id: str,
) -> tuple[str, bool]:
    """记录版本真正参与一次请求；返回 (variant, 是否新增)。"""

    if not request_id.strip() or not user_id.strip():
        raise ValueError("request_id 和 user_id 不能为空。")

    assignment = assign_user(connection, EXPERIMENT, user_id)
    config, digest = load_frozen_config(
        connection,
        assignment.experiment_id,
        assignment.allocation_version,
        assignment.variant,
    )

    old = connection.execute(
        "SELECT * FROM experiment_exposures WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    if old is not None:
        # request_id 是幂等键，但不能被另一个用户或版本复用。
        if old["user_id"] != user_id or old["variant"] != assignment.variant:
            raise ValueError("request_id 已经关联到其他用户或版本。")
        return old["variant"], False

    with connection:
        connection.execute(
            """
            INSERT INTO experiment_exposures (
                request_id, experiment_id, allocation_version,
                user_id, variant, bucket, prompt_version,
                config_hash, exposed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                assignment.experiment_id,
                assignment.allocation_version,
                user_id,
                assignment.variant,
                assignment.bucket,
                config.prompt_version,
                digest,
                now_text(),
            ),
        )
    return assignment.variant, True


def validate_outcome(outcome: OutcomeInput) -> None:
    if outcome.status not in {"success", "rejected", "error", "cancelled"}:
        raise ValueError("无效的 outcome status。")
    if min(
        outcome.latency_ms,
        outcome.model_calls,
        outcome.total_tokens,
        outcome.cache_hits,
    ) < 0:
        raise ValueError("耗时、调用、Token 和缓存命中不能为负数。")
    if outcome.quality_passed and outcome.status != "success":
        raise ValueError("只有 success 结果才能 quality_passed=True。")


def log_outcome(
    connection: sqlite3.Connection,
    outcome: OutcomeInput,
) -> bool:
    """记录一次客观结果；同一request_id只允许一条最终结果。"""

    validate_outcome(outcome)
    with connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO experiment_outcomes (
                request_id, status, quality_passed, latency_ms,
                model_calls, total_tokens, cache_hits,
                error_type, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.request_id,
                outcome.status,
                int(outcome.quality_passed),
                outcome.latency_ms,
                outcome.model_calls,
                outcome.total_tokens,
                outcome.cache_hits,
                outcome.error_type,
                now_text(),
            ),
        )
    return cursor.rowcount == 1


def log_feedback(
    connection: sqlite3.Connection,
    feedback: FeedbackEventInput,
) -> bool:
    """保存或修改反馈，并检查反馈用户就是被曝光用户。"""

    if feedback.rating not in {"up", "down", "neutral"}:
        raise ValueError("rating 必须是 up、down 或 neutral。")

    exposure = connection.execute(
        "SELECT user_id FROM experiment_exposures WHERE request_id = ?",
        (feedback.request_id,),
    ).fetchone()
    if exposure is None:
        raise LookupError("没有找到对应曝光，不能记录反馈。")
    if exposure["user_id"] != feedback.user_id:
        raise PermissionError("反馈用户与曝光用户不一致。")

    old = connection.execute(
        """
        SELECT 1 FROM experiment_feedback_events
        WHERE request_id = ? AND user_id = ?
        """,
        (feedback.request_id, feedback.user_id),
    ).fetchone()
    timestamp = now_text()
    with connection:
        connection.execute(
            """
            INSERT INTO experiment_feedback_events (
                request_id, user_id, rating, reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id, user_id) DO UPDATE SET
                rating = excluded.rating,
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (
                feedback.request_id,
                feedback.user_id,
                feedback.rating,
                feedback.reason,
                timestamp,
                timestamp,
            ),
        )
    return old is None


def load_event_stream(connection: sqlite3.Connection) -> list[dict[str, object]]:
    """JOIN 三类事件，形成后续指标计算所需的宽表视图。"""

    rows = connection.execute(
        """
        SELECT
            e.request_id,
            e.user_id,
            e.variant,
            e.bucket,
            e.prompt_version,
            substr(e.config_hash, 1, 12) AS config_hash_short,
            o.status,
            o.quality_passed,
            o.latency_ms,
            o.model_calls,
            o.total_tokens,
            f.rating,
            f.reason
        FROM experiment_exposures AS e
        LEFT JOIN experiment_outcomes AS o
            ON o.request_id = e.request_id
        LEFT JOIN experiment_feedback_events AS f
            ON f.request_id = e.request_id
           AND f.user_id = e.user_id
        WHERE e.experiment_id = ?
          AND e.allocation_version = ?
        ORDER BY e.request_id
        """,
        (EXPERIMENT.experiment_id, EXPERIMENT.allocation_version),
    ).fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    print("Day26 实验曝光、结果与反馈事件")
    print("实验：", EXPERIMENT.experiment_id)
    print("=" * 104)

    demonstrations = [
        ("req-exp-001", "user-xiaolin", "success", True, 2100, 2, 1900, "up"),
        ("req-exp-002", "user-leo", "success", True, 2450, 2, 2050, "up"),
        ("req-exp-003", "user-001", "success", True, 1800, 2, 1750, "up"),
        ("req-exp-004", "user-002", "rejected", False, 1300, 1, 900, "down"),
        ("req-exp-005", "user-004", "success", True, 2900, 3, 2600, "neutral"),
        ("req-exp-006", "user-005", "cancelled", False, 700, 1, 500, None),
    ]

    with open_database() as connection:
        create_schema(connection)
        create_config_schema(connection)
        create_event_schema(connection)
        save_experiment(connection, EXPERIMENT)
        freeze_variant_configs(connection, EXPERIMENT, VARIANT_CONFIGS)

        for (
            request_id,
            user_id,
            status,
            quality_passed,
            latency_ms,
            model_calls,
            total_tokens,
            rating,
        ) in demonstrations:
            variant, exposure_created = log_exposure(
                connection, request_id, user_id
            )
            outcome_created = log_outcome(
                connection,
                OutcomeInput(
                    request_id=request_id,
                    status=status,  # type: ignore[arg-type]
                    quality_passed=quality_passed,
                    latency_ms=latency_ms,
                    model_calls=model_calls,
                    total_tokens=total_tokens,
                    cache_hits=0,
                ),
            )
            if rating is not None:
                feedback_created = log_feedback(
                    connection,
                    FeedbackEventInput(
                        request_id=request_id,
                        user_id=user_id,
                        rating=rating,  # type: ignore[arg-type]
                        reason=f"模拟{rating}反馈",
                    ),
                )
            else:
                feedback_created = None

            print(
                f"{request_id} | {user_id:<13} | {variant:<9} | "
                f"曝光={'新增' if exposure_created else '已有'} | "
                f"结果={'新增' if outcome_created else '已有'} | "
                f"反馈={rating or '未提交'}"
                + (
                    f"({'新增' if feedback_created else '更新'})"
                    if feedback_created is not None
                    else ""
                )
            )

        stream = load_event_stream(connection)

    print("\n" + "=" * 104)
    print("JOIN 后的实验事件流：")
    print(json.dumps(stream, ensure_ascii=False, indent=2))

    print("\n" + "=" * 104)
    print("本步数据流：")
    print("Assignment（用户属于哪组）")
    print("→ 版本真正开始服务 → Exposure（实验分母）")
    print("→ 请求完成 → Outcome（质量、延迟、成本、错误）")
    print("→ 用户主动评价 → FeedbackEvent（主观结果，可能为空）")
    print("→ 三表LEFT JOIN → 下一步按variant聚合A/B指标")


if __name__ == "__main__":
    main()
