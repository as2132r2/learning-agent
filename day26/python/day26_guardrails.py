"""Day26 第 5 步：A/B 实验质量、错误率、延迟和成本护栏。

运行：
    python day26/python/day26_guardrails.py

程序会评估真实数据库，但只对真实数据应用状态变更；
假想故障场景仅用于演示，不会修改实验状态。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal
from uuid import uuid4

from day26_ab_metrics import VariantMetrics, aggregate_metrics
from day26_exposure_events import create_event_schema
from day26_stable_assignment import EXPERIMENT, create_schema, open_database
from day26_variant_config import create_config_schema


GuardrailStatus = Literal["pass", "fail", "not_enough_data"]
GuardrailAction = Literal["continue", "continue_collecting", "stop"]

MIN_EXPOSURES_FOR_STANDARD_RULES = 20
MIN_EXPOSURES_FOR_CATASTROPHIC_RULE = 3
MAX_ABSOLUTE_ERROR_RATE = 0.50
MAX_QUALITY_REGRESSION_POINTS = 5.0
MAX_REJECTION_REGRESSION_POINTS = 10.0
MAX_LATENCY_RATIO = 1.25
MAX_TOKEN_RATIO = 1.30


@dataclass(slots=True)
class SafetyMetrics:
    """护栏所需的客观指标。"""

    variant: str
    exposures: int
    error_count: int
    rejection_count: int
    error_rate: float
    rejection_rate: float
    quality_pass_rate: float
    average_latency_ms: float
    average_total_tokens: float


@dataclass(slots=True)
class GuardrailCheck:
    """一条护栏的判断结果。"""

    name: str
    status: GuardrailStatus
    observed: str
    threshold: str
    explanation: str


@dataclass(slots=True)
class GuardrailDecision:
    """全部护栏合并后的实验动作。"""

    action: GuardrailAction
    checks: list[GuardrailCheck]
    reason: str


def create_guardrail_schema(connection: sqlite3.Connection) -> None:
    """保存每次护栏评估，形成审计历史。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiment_guardrail_decisions (
            decision_id         TEXT PRIMARY KEY,
            experiment_id       TEXT NOT NULL,
            allocation_version  TEXT NOT NULL,
            action              TEXT NOT NULL
                                CHECK (action IN (
                                    'continue', 'continue_collecting', 'stop'
                                )),
            reason              TEXT NOT NULL,
            checks_json         TEXT NOT NULL,
            applied             INTEGER NOT NULL CHECK (applied IN (0, 1)),
            evaluated_at        TEXT NOT NULL,
            FOREIGN KEY (experiment_id)
                REFERENCES experiments(experiment_id)
        );
        """
    )
    connection.commit()


def load_safety_metrics(
    connection: sqlite3.Connection,
    basic: dict[str, VariantMetrics],
) -> dict[str, SafetyMetrics]:
    """在第 4 步指标基础上补充 error 和 rejected。"""

    rows = connection.execute(
        """
        SELECT
            e.variant,
            SUM(CASE WHEN o.status = 'error' THEN 1 ELSE 0 END) AS errors,
            SUM(CASE WHEN o.status = 'rejected' THEN 1 ELSE 0 END) AS rejections
        FROM experiment_exposures AS e
        LEFT JOIN experiment_outcomes AS o
            ON o.request_id = e.request_id
        WHERE e.experiment_id = ?
          AND e.allocation_version = ?
        GROUP BY e.variant
        """,
        (EXPERIMENT.experiment_id, EXPERIMENT.allocation_version),
    ).fetchall()
    counts = {
        row["variant"]: (int(row["errors"] or 0), int(row["rejections"] or 0))
        for row in rows
    }

    result: dict[str, SafetyMetrics] = {}
    for variant, item in basic.items():
        errors, rejections = counts.get(variant, (0, 0))
        exposures = item.exposures
        result[variant] = SafetyMetrics(
            variant=variant,
            exposures=exposures,
            error_count=errors,
            rejection_count=rejections,
            error_rate=errors / exposures if exposures else 0.0,
            rejection_rate=rejections / exposures if exposures else 0.0,
            quality_pass_rate=item.quality_pass_rate or 0.0,
            average_latency_ms=item.average_latency_ms or 0.0,
            average_total_tokens=item.average_total_tokens or 0.0,
        )
    return result


def ratio(candidate: float, control: float) -> float | None:
    if control <= 0:
        return None
    return candidate / control


def evaluate_guardrails(
    control: SafetyMetrics,
    candidate: SafetyMetrics,
) -> GuardrailDecision:
    """先判断灾难性错误，再判断需要较多样本的标准护栏。"""

    checks: list[GuardrailCheck] = []

    catastrophic_ready = candidate.exposures >= MIN_EXPOSURES_FOR_CATASTROPHIC_RULE
    catastrophic_failed = (
        catastrophic_ready and candidate.error_rate >= MAX_ABSOLUTE_ERROR_RATE
    )
    checks.append(
        GuardrailCheck(
            name="catastrophic_error_rate",
            status=(
                "fail"
                if catastrophic_failed
                else "pass"
                if catastrophic_ready
                else "not_enough_data"
            ),
            observed=f"candidate错误率={candidate.error_rate:.1%}",
            threshold=f"错误率必须低于{MAX_ABSOLUTE_ERROR_RATE:.0%}",
            explanation="灾难性规则只需少量曝光即可触发。",
        )
    )
    if catastrophic_failed:
        return GuardrailDecision(
            action="stop",
            checks=checks,
            reason="candidate灾难性错误率达到停止线。",
        )

    standard_ready = (
        control.exposures >= MIN_EXPOSURES_FOR_STANDARD_RULES
        and candidate.exposures >= MIN_EXPOSURES_FOR_STANDARD_RULES
    )
    if not standard_ready:
        checks.append(
            GuardrailCheck(
                name="standard_sample_gate",
                status="not_enough_data",
                observed=(
                    f"control={control.exposures}, candidate={candidate.exposures}"
                ),
                threshold=f"每组至少{MIN_EXPOSURES_FOR_STANDARD_RULES}次曝光",
                explanation="普通波动不能在极小样本下触发自动停止。",
            )
        )
        return GuardrailDecision(
            action="continue_collecting",
            checks=checks,
            reason="未触发灾难性规则，但标准护栏样本仍不足。",
        )

    quality_regression = (
        control.quality_pass_rate - candidate.quality_pass_rate
    ) * 100
    rejection_regression = (
        candidate.rejection_rate - control.rejection_rate
    ) * 100
    latency_ratio = ratio(
        candidate.average_latency_ms, control.average_latency_ms
    )
    token_ratio = ratio(
        candidate.average_total_tokens, control.average_total_tokens
    )

    standard_values = [
        (
            "quality_regression",
            quality_regression > MAX_QUALITY_REGRESSION_POINTS,
            f"质量通过率下降={quality_regression:.2f}个百分点",
            f"不得下降超过{MAX_QUALITY_REGRESSION_POINTS:.1f}个百分点",
        ),
        (
            "rejection_regression",
            rejection_regression > MAX_REJECTION_REGRESSION_POINTS,
            f"拒绝率增加={rejection_regression:.2f}个百分点",
            f"不得增加超过{MAX_REJECTION_REGRESSION_POINTS:.1f}个百分点",
        ),
        (
            "latency_ratio",
            latency_ratio is not None and latency_ratio > MAX_LATENCY_RATIO,
            f"candidate/control={latency_ratio:.3f}"
            if latency_ratio is not None
            else "control延迟为0，无法计算",
            f"比值不得超过{MAX_LATENCY_RATIO}",
        ),
        (
            "token_ratio",
            token_ratio is not None and token_ratio > MAX_TOKEN_RATIO,
            f"candidate/control={token_ratio:.3f}"
            if token_ratio is not None
            else "control Token为0，无法计算",
            f"比值不得超过{MAX_TOKEN_RATIO}",
        ),
    ]

    for name, failed, observed, threshold in standard_values:
        checks.append(
            GuardrailCheck(
                name=name,
                status="fail" if failed else "pass",
                observed=observed,
                threshold=threshold,
                explanation="标准护栏已达到最低样本要求。",
            )
        )

    failed_names = [check.name for check in checks if check.status == "fail"]
    if failed_names:
        return GuardrailDecision(
            action="stop",
            checks=checks,
            reason="触发护栏：" + "、".join(failed_names),
        )
    return GuardrailDecision(
        action="continue",
        checks=checks,
        reason="所有已启用护栏均通过。",
    )


def save_and_apply_decision(
    connection: sqlite3.Connection,
    decision: GuardrailDecision,
) -> str:
    """保存真实决策；若action=stop，则原子地停止实验。"""

    decision_id = f"guard-{uuid4().hex[:12]}"
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection:
        connection.execute(
            """
            INSERT INTO experiment_guardrail_decisions (
                decision_id, experiment_id, allocation_version,
                action, reason, checks_json, applied, evaluated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                decision_id,
                EXPERIMENT.experiment_id,
                EXPERIMENT.allocation_version,
                decision.action,
                decision.reason,
                json.dumps(
                    [asdict(check) for check in decision.checks],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                now,
            ),
        )
        if decision.action == "stop":
            connection.execute(
                """
                UPDATE experiments
                SET status = 'stopped', updated_at = ?
                WHERE experiment_id = ?
                """,
                (now, EXPERIMENT.experiment_id),
            )
    return decision_id


def show_decision(title: str, decision: GuardrailDecision) -> None:
    print("\n" + "=" * 104)
    print(title)
    print("动作：", decision.action)
    print("原因：", decision.reason)
    for check in decision.checks:
        print(
            f"- {check.name:<28} {check.status:<16} "
            f"{check.observed}；{check.threshold}"
        )


def hypothetical_failure() -> tuple[SafetyMetrics, SafetyMetrics]:
    """构造足量且candidate明显退化的演示场景。"""

    control = SafetyMetrics(
        variant="control",
        exposures=100,
        error_count=2,
        rejection_count=5,
        error_rate=0.02,
        rejection_rate=0.05,
        quality_pass_rate=0.92,
        average_latency_ms=2000,
        average_total_tokens=1800,
    )
    candidate = SafetyMetrics(
        variant="candidate",
        exposures=100,
        error_count=8,
        rejection_count=22,
        error_rate=0.08,
        rejection_rate=0.22,
        quality_pass_rate=0.76,
        average_latency_ms=2900,
        average_total_tokens=2550,
    )
    return control, candidate


def main() -> None:
    print("Day26 A/B 实验安全护栏")
    print("实验：", EXPERIMENT.experiment_id)
    print("标准护栏最低曝光：", MIN_EXPOSURES_FOR_STANDARD_RULES, "/ 组")
    print("灾难性错误停止线：", f">={MAX_ABSOLUTE_ERROR_RATE:.0%}")

    with open_database() as connection:
        create_schema(connection)
        create_config_schema(connection)
        create_event_schema(connection)
        create_guardrail_schema(connection)

        basic = aggregate_metrics(connection)
        if "control" not in basic or "candidate" not in basic:
            raise RuntimeError("缺少实验事件，请先运行day26_exposure_events.py。")
        real_metrics = load_safety_metrics(connection, basic)
        real_decision = evaluate_guardrails(
            real_metrics["control"], real_metrics["candidate"]
        )
        decision_id = save_and_apply_decision(connection, real_decision)
        experiment_status = connection.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (EXPERIMENT.experiment_id,),
        ).fetchone()["status"]

    show_decision("真实数据库护栏结果", real_decision)
    print("决策编号：", decision_id)
    print("实验状态：", experiment_status)

    failed_control, failed_candidate = hypothetical_failure()
    failed_decision = evaluate_guardrails(failed_control, failed_candidate)
    show_decision("假想故障场景（仅演示，不写数据库）", failed_decision)
    print("假想场景若应用，将把experiments.status原子更新为stopped。")

    print("\n" + "=" * 104)
    print("本步数据流：")
    print("客观Outcome指标 → SafetyMetrics")
    print("→ 灾难性护栏（少量样本也可停止）")
    print("→ 标准样本门槛 → 质量/拒绝率/延迟/Token护栏")
    print("→ GuardrailDecision审计记录")
    print("→ action=stop时原子更新实验状态")


if __name__ == "__main__":
    main()
