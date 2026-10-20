"""Day26 第 4 步：A/B 指标聚合与基础显著性判断。

运行前先生成第 3 步事件：
    python day26/python/day26_exposure_events.py

然后运行：
    python day26/python/day26_ab_metrics.py

输出：
    day26/reports/day26_ab_metrics.json
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

from day26_exposure_events import create_event_schema
from day26_stable_assignment import EXPERIMENT, create_schema, open_database
from day26_variant_config import create_config_schema


DAY26_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = DAY26_DIR / "reports"
REPORT_PATH = REPORT_DIR / "day26_ab_metrics.json"
ALPHA = 0.05
MINIMUM_SAMPLE_PER_VARIANT = 30


@dataclass(slots=True)
class VariantMetrics:
    """单个实验版本的聚合指标。"""

    variant: str
    exposures: int
    outcomes: int
    successes: int
    quality_passes: int
    feedback_count: int
    up_count: int
    down_count: int
    neutral_count: int
    success_rate: float | None
    quality_pass_rate: float | None
    feedback_response_rate: float | None
    positive_feedback_rate: float | None
    average_latency_ms: float | None
    average_total_tokens: float | None
    average_model_calls: float | None


@dataclass(slots=True)
class ProportionTest:
    """两版本比例指标的比较结果。"""

    metric: str
    control_successes: int
    control_total: int
    candidate_successes: int
    candidate_total: int
    control_rate: float | None
    candidate_rate: float | None
    absolute_lift_points: float | None
    relative_lift_percent: float | None
    control_wilson_95: tuple[float, float] | None
    candidate_wilson_95: tuple[float, float] | None
    z_score: float | None
    p_value: float | None
    statistically_significant: bool
    decision: str


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    """计算二项比例的 Wilson 95% 置信区间。"""

    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def two_proportion_z_test(
    first_successes: int,
    first_total: int,
    second_successes: int,
    second_total: int,
) -> tuple[float | None, float | None]:
    """计算双侧双比例 z 检验；返回 (z_score, p_value)。"""

    if first_total == 0 or second_total == 0:
        return None, None

    first_rate = first_successes / first_total
    second_rate = second_successes / second_total
    pooled = (first_successes + second_successes) / (first_total + second_total)
    standard_error = math.sqrt(
        pooled * (1 - pooled) * (1 / first_total + 1 / second_total)
    )
    if standard_error == 0:
        # 两组都全成功或全失败时没有可估计方差。
        return 0.0, 1.0

    z_score = (second_rate - first_rate) / standard_error
    p_value = 2 * (1 - NormalDist().cdf(abs(z_score)))
    return round(z_score, 4), round(p_value, 6)


def aggregate_metrics(connection) -> dict[str, VariantMetrics]:
    """直接以 Exposure 为分母，LEFT JOIN 结果与反馈。"""

    rows = connection.execute(
        """
        SELECT
            e.variant,
            COUNT(DISTINCT e.request_id) AS exposures,
            COUNT(DISTINCT o.request_id) AS outcomes,
            SUM(CASE WHEN o.status = 'success' THEN 1 ELSE 0 END) AS successes,
            SUM(CASE WHEN o.quality_passed = 1 THEN 1 ELSE 0 END)
                AS quality_passes,
            COUNT(f.request_id) AS feedback_count,
            SUM(CASE WHEN f.rating = 'up' THEN 1 ELSE 0 END) AS up_count,
            SUM(CASE WHEN f.rating = 'down' THEN 1 ELSE 0 END) AS down_count,
            SUM(CASE WHEN f.rating = 'neutral' THEN 1 ELSE 0 END)
                AS neutral_count,
            AVG(o.latency_ms) AS average_latency_ms,
            AVG(o.total_tokens) AS average_total_tokens,
            AVG(o.model_calls) AS average_model_calls
        FROM experiment_exposures AS e
        LEFT JOIN experiment_outcomes AS o
            ON o.request_id = e.request_id
        LEFT JOIN experiment_feedback_events AS f
            ON f.request_id = e.request_id
           AND f.user_id = e.user_id
        WHERE e.experiment_id = ?
          AND e.allocation_version = ?
        GROUP BY e.variant
        ORDER BY e.variant
        """,
        (EXPERIMENT.experiment_id, EXPERIMENT.allocation_version),
    ).fetchall()

    result: dict[str, VariantMetrics] = {}
    for row in rows:
        exposures = int(row["exposures"] or 0)
        outcomes = int(row["outcomes"] or 0)
        successes = int(row["successes"] or 0)
        quality_passes = int(row["quality_passes"] or 0)
        feedback_count = int(row["feedback_count"] or 0)
        up_count = int(row["up_count"] or 0)
        down_count = int(row["down_count"] or 0)
        neutral_count = int(row["neutral_count"] or 0)

        metrics = VariantMetrics(
            variant=row["variant"],
            exposures=exposures,
            outcomes=outcomes,
            successes=successes,
            quality_passes=quality_passes,
            feedback_count=feedback_count,
            up_count=up_count,
            down_count=down_count,
            neutral_count=neutral_count,
            success_rate=safe_rate(successes, exposures),
            quality_pass_rate=safe_rate(quality_passes, exposures),
            feedback_response_rate=safe_rate(feedback_count, exposures),
            # 中立也属于已反馈用户，因此作为非正向计入分母。
            positive_feedback_rate=safe_rate(up_count, feedback_count),
            average_latency_ms=(
                round(float(row["average_latency_ms"]), 1)
                if row["average_latency_ms"] is not None
                else None
            ),
            average_total_tokens=(
                round(float(row["average_total_tokens"]), 1)
                if row["average_total_tokens"] is not None
                else None
            ),
            average_model_calls=(
                round(float(row["average_model_calls"]), 2)
                if row["average_model_calls"] is not None
                else None
            ),
        )
        result[metrics.variant] = metrics
    return result


def compare_proportion(
    metric: str,
    control_successes: int,
    control_total: int,
    candidate_successes: int,
    candidate_total: int,
) -> ProportionTest:
    """计算提升、置信区间、p值，并给出保守决策。"""

    control_rate = safe_rate(control_successes, control_total)
    candidate_rate = safe_rate(candidate_successes, candidate_total)
    z_score, p_value = two_proportion_z_test(
        control_successes,
        control_total,
        candidate_successes,
        candidate_total,
    )

    absolute_lift = None
    relative_lift = None
    if control_rate is not None and candidate_rate is not None:
        absolute_lift = round((candidate_rate - control_rate) * 100, 2)
        if control_rate != 0:
            relative_lift = round(
                (candidate_rate - control_rate) / control_rate * 100,
                2,
            )

    enough_samples = (
        control_total >= MINIMUM_SAMPLE_PER_VARIANT
        and candidate_total >= MINIMUM_SAMPLE_PER_VARIANT
    )
    statistically_significant = bool(
        enough_samples and p_value is not None and p_value < ALPHA
    )

    if not enough_samples:
        decision = (
            f"样本不足：每组至少需要{MINIMUM_SAMPLE_PER_VARIANT}个有效样本，"
            "当前不能下结论。"
        )
    elif p_value is None:
        decision = "数据不足，无法计算显著性。"
    elif p_value >= ALPHA:
        decision = f"p={p_value}，未达到显著水平，继续收集数据。"
    elif candidate_rate is not None and control_rate is not None:
        decision = (
            "candidate 显著更好。"
            if candidate_rate > control_rate
            else "candidate 显著更差。"
        )
    else:
        decision = "无法判断。"

    return ProportionTest(
        metric=metric,
        control_successes=control_successes,
        control_total=control_total,
        candidate_successes=candidate_successes,
        candidate_total=candidate_total,
        control_rate=control_rate,
        candidate_rate=candidate_rate,
        absolute_lift_points=absolute_lift,
        relative_lift_percent=relative_lift,
        control_wilson_95=wilson_interval(control_successes, control_total),
        candidate_wilson_95=wilson_interval(candidate_successes, candidate_total),
        z_score=z_score,
        p_value=p_value,
        statistically_significant=statistically_significant,
        decision=decision,
    )


def save_report(
    metrics: dict[str, VariantMetrics],
    tests: list[ProportionTest],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "report_version": "day26-ab-metrics-v1",
        "experiment_id": EXPERIMENT.experiment_id,
        "allocation_version": EXPERIMENT.allocation_version,
        "alpha": ALPHA,
        "minimum_sample_per_variant": MINIMUM_SAMPLE_PER_VARIANT,
        "variants": {name: asdict(value) for name, value in metrics.items()},
        "proportion_tests": [asdict(test) for test in tests],
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def main() -> None:
    print("Day26 A/B 指标聚合与基础显著性判断")
    print("实验：", EXPERIMENT.experiment_id)
    print("显著水平：", ALPHA)
    print("每组最低有效样本：", MINIMUM_SAMPLE_PER_VARIANT)
    print("=" * 112)

    with open_database() as connection:
        create_schema(connection)
        create_config_schema(connection)
        create_event_schema(connection)
        metrics = aggregate_metrics(connection)

    if "control" not in metrics or "candidate" not in metrics:
        raise RuntimeError(
            "缺少control或candidate曝光，请先运行day26_exposure_events.py。"
        )

    print(
        f"{'版本':<12}{'曝光':>6}{'成功率':>12}{'质量通过率':>14}"
        f"{'反馈响应率':>14}{'正向反馈率':>14}{'平均延迟':>12}{'平均Token':>12}"
    )
    print("-" * 112)
    for variant in ("control", "candidate"):
        item = metrics[variant]
        print(
            f"{variant:<12}{item.exposures:>6}"
            f"{format_percent(item.success_rate):>12}"
            f"{format_percent(item.quality_pass_rate):>14}"
            f"{format_percent(item.feedback_response_rate):>14}"
            f"{format_percent(item.positive_feedback_rate):>14}"
            f"{str(item.average_latency_ms) + 'ms':>12}"
            f"{item.average_total_tokens:>12}"
        )

    control = metrics["control"]
    candidate = metrics["candidate"]
    tests = [
        compare_proportion(
            "success_rate",
            control.successes,
            control.exposures,
            candidate.successes,
            candidate.exposures,
        ),
        compare_proportion(
            "quality_pass_rate",
            control.quality_passes,
            control.exposures,
            candidate.quality_passes,
            candidate.exposures,
        ),
        compare_proportion(
            "positive_feedback_rate",
            control.up_count,
            control.feedback_count,
            candidate.up_count,
            candidate.feedback_count,
        ),
    ]

    print("\n" + "=" * 112)
    print("比例指标比较：")
    for test in tests:
        print(f"\n指标：{test.metric}")
        print(
            f"control={format_percent(test.control_rate)} "
            f"({test.control_successes}/{test.control_total})"
        )
        print(
            f"candidate={format_percent(test.candidate_rate)} "
            f"({test.candidate_successes}/{test.candidate_total})"
        )
        print("绝对提升：", test.absolute_lift_points, "个百分点")
        print("95% Wilson区间：", test.control_wilson_95, test.candidate_wilson_95)
        print("z：", test.z_score, "p：", test.p_value)
        print("判断：", test.decision)

    save_report(metrics, tests)
    print("\n" + "=" * 112)
    print("报告：", REPORT_PATH)
    print("本步数据流：")
    print("Exposure LEFT JOIN Outcome LEFT JOIN Feedback")
    print("→ 按variant聚合客观指标与主观指标")
    print("→ Wilson区间 + 双比例z检验 + 最低样本门槛")
    print("→ A/B指标报告")
    print("\n注意：反馈率低时，正向反馈率存在自选择偏差，不能代表全部曝光用户。")


if __name__ == "__main__":
    main()
