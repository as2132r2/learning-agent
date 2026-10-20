"""Day24 第五步：模型调用次数、Token预算与安全降级策略。

默认运行：
    python day24/python/day24_budget_controller.py

观察中风险安全降级：
    python day24/python/day24_budget_controller.py \
        --max-calls 2 --max-tokens 3000 --allow-downgrade

高风险预算不足会拒绝，不允许静默降级。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY22_PYTHON = PROJECT_DIR / "day22" / "python"
DAY23_PYTHON = PROJECT_DIR / "day23" / "python"
DAY24_PYTHON = PROJECT_DIR / "day24" / "python"
for directory in (DAY22_PYTHON, DAY23_PYTHON, DAY24_PYTHON):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from day22_final_hybrid_rag import retrieve_rrf_candidates  # noqa: E402
from day22_llm_reranker import rerank_candidates  # noqa: E402
from day23_answer_quality import (  # noqa: E402
    build_final_quality_report,
    evaluate_answer_quality,
)
from day23_citation_audit import (  # noqa: E402
    audit_citations,
    generate_structured_answer,
)
from day23_claim_groundedness import (  # noqa: E402
    build_groundedness_report,
    evaluate_claim_support,
)
from day23_structured_citations import (  # noqa: E402
    build_readable_answer,
    select_evidence,
)
from day24_adaptive_audit import (  # noqa: E402
    AuditPolicy,
    POLICIES,
    RiskDecision,
    classify_risk,
)
from day24_parallel_evaluators import (  # noqa: E402
    ModelCallResult,
    measured_model_call,
)


# 来自前面实测的保守整数估算。真实Token可能上下波动。
STAGE_TOKEN_ESTIMATES = {
    "Reranker": 1700,
    "StructuredAnswer": 900,
    "Groundedness": 1000,
    "AnswerQuality": 1100,
}


class BudgetExceeded(RuntimeError):
    """调用前预算不足，或调用后真实Token超过软预算。"""


@dataclass
class BudgetReservation:
    stage_name: str
    estimated_tokens: int


@dataclass
class BudgetController:
    """在模型调用前预留预算，调用后记录真实消耗。"""

    max_calls: int
    max_tokens: int
    reserved_calls: int = 0
    reserved_tokens: int = 0
    actual_calls: int = 0
    actual_tokens: int = 0
    reservations: list[BudgetReservation] = field(default_factory=list)
    completed_calls: list[ModelCallResult] = field(default_factory=list)

    def can_reserve(self, calls: int, tokens: int) -> bool:
        return (
            self.reserved_calls + calls <= self.max_calls
            and self.reserved_tokens + tokens <= self.max_tokens
        )

    def reserve(self, stage_name: str, estimated_tokens: int) -> None:
        if not self.can_reserve(1, estimated_tokens):
            raise BudgetExceeded(
                f"阶段{stage_name}预算不足：需要1次调用/约{estimated_tokens} Token，"
                f"剩余{self.max_calls - self.reserved_calls}次/"
                f"约{self.max_tokens - self.reserved_tokens} Token。"
            )
        self.reserved_calls += 1
        self.reserved_tokens += estimated_tokens
        self.reservations.append(
            BudgetReservation(stage_name, estimated_tokens)
        )

    def complete(self, call: ModelCallResult) -> None:
        self.actual_calls += call.model_calls
        self.actual_tokens += call.total_tokens
        self.completed_calls.append(call)
        if self.actual_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"真实Token累计{self.actual_tokens}，已经超过软预算"
                f"{self.max_tokens}。当前调用无法撤销，后续阶段停止。"
            )


@dataclass(frozen=True)
class PlanCost:
    risk_level: str
    calls: int
    estimated_tokens: int


@dataclass
class BudgetedRunResult:
    status: str
    answer: str | None
    requested_risk: str
    executed_risk: str | None
    downgrade_reason: str | None
    controller: BudgetController
    wall_seconds: float
    audit_passed: bool
    messages: list[str]


def estimate_policy_cost(policy: AuditPolicy) -> PlanCost:
    stages = ["Reranker", "StructuredAnswer"]
    if policy.run_groundedness:
        stages.append("Groundedness")
    if policy.run_answer_quality:
        stages.append("AnswerQuality")
    return PlanCost(
        risk_level=policy.risk_level,
        calls=len(stages),
        estimated_tokens=sum(STAGE_TOKEN_ESTIMATES[name] for name in stages),
    )


def choose_executable_policy(
    decision: RiskDecision,
    max_calls: int,
    max_tokens: int,
    allow_downgrade: bool,
) -> tuple[AuditPolicy | None, str | None]:
    """预算预检；高风险永不降级，中风险仅在显式允许时降级。"""

    desired = POLICIES[decision.risk_level]
    desired_cost = estimate_policy_cost(desired)
    if desired_cost.calls <= max_calls and desired_cost.estimated_tokens <= max_tokens:
        return desired, None

    if decision.risk_level == "high":
        return None, "高风险问题预算不足，安全策略禁止降低审计级别。"

    if decision.risk_level == "medium" and allow_downgrade:
        low = POLICIES["low"]
        low_cost = estimate_policy_cost(low)
        if low_cost.calls <= max_calls and low_cost.estimated_tokens <= max_tokens:
            return low, "中风险完整审计预算不足，用户显式允许降级到低风险策略。"

    return None, "所需审计预算不足，且没有可接受的授权降级方案。"


async def budgeted_call(
    controller: BudgetController,
    stage_name: str,
    function,
    *args,
) -> ModelCallResult:
    estimate = STAGE_TOKEN_ESTIMATES[stage_name]
    controller.reserve(stage_name, estimate)
    result = await asyncio.to_thread(
        measured_model_call,
        stage_name,
        function,
        *args,
    )
    controller.complete(result)
    return result


async def run_budgeted_pipeline(
    question: str,
    decision: RiskDecision,
    policy: AuditPolicy,
    downgrade_reason: str | None,
    max_calls: int,
    max_tokens: int,
) -> BudgetedRunResult:
    started = time.perf_counter()
    controller = BudgetController(max_calls=max_calls, max_tokens=max_tokens)
    messages: list[str] = []

    try:
        rrf_results, _ = await asyncio.to_thread(
            retrieve_rrf_candidates,
            question,
        )
        reranker_call = await budgeted_call(
            controller,
            "Reranker",
            rerank_candidates,
            question,
            rrf_results,
        )
        reranked_results, _ = reranker_call.value
        evidence = select_evidence(reranked_results)
        if not evidence:
            messages.append("没有资料达到Reranker阈值。")
            return BudgetedRunResult(
                "retrieval_rejected", None, decision.risk_level,
                policy.risk_level, downgrade_reason, controller,
                time.perf_counter() - started, False, messages,
            )

        answer_call = await budgeted_call(
            controller,
            "StructuredAnswer",
            generate_structured_answer,
            question,
            evidence,
        )
        answer = answer_call.value
        rendered = build_readable_answer(answer)
        citation = audit_citations(answer, len(evidence), rendered)
        if not citation.passed:
            messages.extend(citation.errors)
            return BudgetedRunResult(
                "audit_failed", None, decision.risk_level,
                policy.risk_level, downgrade_reason, controller,
                time.perf_counter() - started, False, messages,
            )

        groundedness_report = None
        quality_response = None

        if policy.run_groundedness and policy.run_answer_quality:
            # 两份预算先各自预留，再并行调用。
            groundedness_task = asyncio.create_task(
                budgeted_call(
                    controller,
                    "Groundedness",
                    evaluate_claim_support,
                    answer,
                    evidence,
                )
            )
            quality_task = asyncio.create_task(
                budgeted_call(
                    controller,
                    "AnswerQuality",
                    evaluate_answer_quality,
                    question,
                    answer,
                    evidence,
                )
            )
            groundedness_call, quality_call = await asyncio.gather(
                groundedness_task,
                quality_task,
            )
            groundedness_report = build_groundedness_report(
                answer,
                groundedness_call.value,
            )
            quality_response = quality_call.value
            final = build_final_quality_report(
                citation,
                groundedness_report,
                quality_response,
            )
            audit_passed = final.passed
            messages.extend(final.errors)
        elif policy.run_groundedness:
            groundedness_call = await budgeted_call(
                controller,
                "Groundedness",
                evaluate_claim_support,
                answer,
                evidence,
            )
            groundedness_report = build_groundedness_report(
                answer,
                groundedness_call.value,
            )
            audit_passed = groundedness_report.passed
            messages.extend(groundedness_report.errors)
            messages.append("预算策略未执行回答完整性评估。")
        else:
            audit_passed = citation.passed
            messages.append("预算策略只执行确定性引用审计。")

        return BudgetedRunResult(
            "success" if audit_passed else "audit_failed",
            rendered if audit_passed else None,
            decision.risk_level,
            policy.risk_level,
            downgrade_reason,
            controller,
            time.perf_counter() - started,
            audit_passed,
            messages,
        )
    except BudgetExceeded as error:
        messages.append(str(error))
        return BudgetedRunResult(
            "budget_exceeded", None, decision.risk_level,
            policy.risk_level, downgrade_reason, controller,
            time.perf_counter() - started, False, messages,
        )


def print_result(result: BudgetedRunResult) -> None:
    controller = result.controller
    print("\n" + "=" * 92)
    print("执行结果：", result.status)
    print("请求风险：", result.requested_risk)
    print("实际策略：", result.executed_risk)
    print("降级原因：", result.downgrade_reason or "无")
    print("审计通过：", result.audit_passed)
    print("墙钟耗时：", f"{result.wall_seconds:.3f}秒")

    print("\n预算使用：")
    print("最大调用数：", controller.max_calls)
    print("预留调用数：", controller.reserved_calls)
    print("实际调用数：", controller.actual_calls)
    print("Token软预算：", controller.max_tokens)
    print("预留Token估算：", controller.reserved_tokens)
    print("实际Token：", controller.actual_tokens)

    print("\n逐阶段预算：")
    actual_by_name = {
        call.name: call.total_tokens
        for call in controller.completed_calls
    }
    for reservation in controller.reservations:
        print(
            f"- {reservation.stage_name}: "
            f"估算={reservation.estimated_tokens}，"
            f"实际={actual_by_name.get(reservation.stage_name, '未完成')}"
        )

    if result.messages:
        print("\n系统消息：")
        for message in result.messages:
            print("-", message)
    if result.answer:
        print("\n最终回答：")
        print(result.answer)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day24预算控制练习")
    parser.add_argument("question", nargs="*", help="知识库问题")
    parser.add_argument(
        "--risk",
        choices=["auto", "low", "medium", "high"],
        default="auto",
    )
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--allow-downgrade", action="store_true")
    return parser.parse_args()


async def main() -> None:
    arguments = parse_arguments()
    if arguments.max_calls < 0 or arguments.max_tokens < 0:
        raise SystemExit("预算不能是负数。")

    question = " ".join(arguments.question).strip() or (
        "什么是 RAG？它的工作流程是什么？"
    )
    decision = classify_risk(question, arguments.risk)
    desired = POLICIES[decision.risk_level]
    desired_cost = estimate_policy_cost(desired)

    print("Day24 模型调用与Token预算控制")
    print("问题：", question)
    print("风险等级：", decision.risk_level)
    print("期望策略估算：", desired_cost)
    print("预算：", arguments.max_calls, "次 /", arguments.max_tokens, "Token")
    print("允许降级：", arguments.allow_downgrade)

    policy, downgrade_reason = choose_executable_policy(
        decision,
        arguments.max_calls,
        arguments.max_tokens,
        arguments.allow_downgrade,
    )
    if policy is None:
        print("\n预算预检拒绝执行：", downgrade_reason)
        return

    print("实际执行策略：", policy.risk_level)
    if downgrade_reason:
        print("策略变化：", downgrade_reason)

    result = await run_budgeted_pipeline(
        question,
        decision,
        policy,
        downgrade_reason,
        arguments.max_calls,
        arguments.max_tokens,
    )
    print_result(result)

    print("\n本步数据流：")
    print("风险策略 → 估算调用/Token → 预算预检")
    print("→ 每次模型调用前reserve → 调用后记录真实Token")
    print("→ 通过 / 授权降级 / 预算拒绝 / 运行中软预算超限")


if __name__ == "__main__":
    asyncio.run(main())
