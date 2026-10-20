"""Day24 第四步：根据问题风险等级选择不同审计深度。

自动判断：
    python day24/python/day24_adaptive_audit.py

手动指定：
    python day24/python/day24_adaptive_audit.py --risk high "什么是 RAG？"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
    AnswerQualityResponse,
    FinalQualityReport,
    build_final_quality_report,
    evaluate_answer_quality,
)
from day23_citation_audit import (  # noqa: E402
    CitationAudit,
    audit_citations,
    generate_structured_answer,
)
from day23_claim_groundedness import (  # noqa: E402
    GroundednessReport,
    GroundednessResponse,
    build_groundedness_report,
    evaluate_claim_support,
)
from day23_structured_citations import (  # noqa: E402
    build_readable_answer,
    select_evidence,
)
from day24_parallel_evaluators import (  # noqa: E402
    ModelCallResult,
    measured_model_call,
)


RiskLevel = Literal["low", "medium", "high"]

HIGH_RISK_KEYWORDS = {
    "医疗", "诊断", "药物", "剂量", "法律", "合同", "诉讼",
    "金融", "投资", "转账", "贷款", "密码", "密钥", "生产环境",
    "删除数据", "执行命令", "人身安全",
}

MEDIUM_RISK_KEYWORDS = {
    "为什么", "如何", "流程", "步骤", "比较", "区别", "分析",
    "详细", "总结", "优缺点", "设计", "方案",
}


@dataclass(frozen=True)
class AuditPolicy:
    """某一风险等级对应的审计开关。"""

    risk_level: RiskLevel
    run_groundedness: bool
    run_answer_quality: bool
    description: str


@dataclass(frozen=True)
class RiskDecision:
    risk_level: RiskLevel
    reasons: list[str]
    source: str


@dataclass
class AdaptiveAuditReport:
    """不同审计深度最终统一成同一个报告类型。"""

    decision: RiskDecision
    policy: AuditPolicy
    citation_audit: CitationAudit
    groundedness_report: GroundednessReport | None
    quality_response: AnswerQualityResponse | None
    final_quality_report: FinalQualityReport | None
    passed: bool
    model_calls: int
    total_tokens: int
    wall_seconds: float
    warnings: list[str]


POLICIES: dict[RiskLevel, AuditPolicy] = {
    "low": AuditPolicy(
        risk_level="low",
        run_groundedness=False,
        run_answer_quality=False,
        description="只执行确定性引用审计，适合低风险事实型问答。",
    ),
    "medium": AuditPolicy(
        risk_level="medium",
        run_groundedness=True,
        run_answer_quality=False,
        description="增加证据支持度检查，适合解释、流程和比较问题。",
    ),
    "high": AuditPolicy(
        risk_level="high",
        run_groundedness=True,
        run_answer_quality=True,
        description="执行完整审计，适合高影响或安全敏感问题。",
    ),
}


def classify_risk(question: str, override: str) -> RiskDecision:
    """用透明规则分类；手动覆盖优先且不消耗模型调用。"""

    if override != "auto":
        return RiskDecision(
            risk_level=override,  # type: ignore[arg-type]
            reasons=[f"命令行显式指定为{override}"],
            source="manual_override",
        )

    high_matches = sorted(
        keyword for keyword in HIGH_RISK_KEYWORDS if keyword in question
    )
    if high_matches:
        return RiskDecision(
            risk_level="high",
            reasons=[f"命中高风险关键词：{', '.join(high_matches)}"],
            source="deterministic_rules",
        )

    medium_matches = sorted(
        keyword for keyword in MEDIUM_RISK_KEYWORDS if keyword in question
    )
    if medium_matches:
        return RiskDecision(
            risk_level="medium",
            reasons=[f"命中复杂问题关键词：{', '.join(medium_matches)}"],
            source="deterministic_rules",
        )

    return RiskDecision(
        risk_level="low",
        reasons=["未命中高风险或复杂问题规则"],
        source="deterministic_rules",
    )


async def call_measured(
    name: str,
    function,
    *args,
) -> ModelCallResult:
    return await asyncio.to_thread(
        measured_model_call,
        name,
        function,
        *args,
    )


async def run_adaptive_audit(
    question: str,
    decision: RiskDecision,
) -> tuple[str, AdaptiveAuditReport]:
    """执行共同生成阶段，再按策略选择审计分支。"""

    started = time.perf_counter()
    policy = POLICIES[decision.risk_level]
    calls: list[ModelCallResult] = []
    warnings: list[str] = []

    rrf_results, _ = await asyncio.to_thread(
        retrieve_rrf_candidates,
        question,
    )
    reranker_call = await call_measured(
        "Reranker",
        rerank_candidates,
        question,
        rrf_results,
    )
    calls.append(reranker_call)
    reranked_results, _ = reranker_call.value
    evidence = select_evidence(reranked_results)
    if not evidence:
        raise RuntimeError("没有资料达到Reranker阈值，不能生成回答。")

    answer_call = await call_measured(
        "结构化回答",
        generate_structured_answer,
        question,
        evidence,
    )
    calls.append(answer_call)
    answer = answer_call.value
    rendered_answer = build_readable_answer(answer)
    citation_audit = audit_citations(
        answer,
        len(evidence),
        rendered_answer,
    )

    groundedness_report: GroundednessReport | None = None
    quality_response: AnswerQualityResponse | None = None
    final_quality_report: FinalQualityReport | None = None

    if not citation_audit.passed:
        passed = False
        warnings.append("引用结构审计失败，停止语义评估。")
    elif policy.run_groundedness and policy.run_answer_quality:
        # 高风险：两个互不依赖的评估器并行运行。
        groundedness_task = asyncio.create_task(
            call_measured(
                "Groundedness",
                evaluate_claim_support,
                answer,
                evidence,
            ),
            name="adaptive-groundedness",
        )
        quality_task = asyncio.create_task(
            call_measured(
                "AnswerQuality",
                evaluate_answer_quality,
                question,
                answer,
                evidence,
            ),
            name="adaptive-answer-quality",
        )
        groundedness_call, quality_call = await asyncio.gather(
            groundedness_task,
            quality_task,
        )
        calls.extend([groundedness_call, quality_call])
        groundedness_response: GroundednessResponse = groundedness_call.value
        quality_response = quality_call.value
        groundedness_report = build_groundedness_report(
            answer,
            groundedness_response,
        )
        final_quality_report = build_final_quality_report(
            citation_audit,
            groundedness_report,
            quality_response,
        )
        passed = final_quality_report.passed
    elif policy.run_groundedness:
        groundedness_call = await call_measured(
            "Groundedness",
            evaluate_claim_support,
            answer,
            evidence,
        )
        calls.append(groundedness_call)
        groundedness_report = build_groundedness_report(
            answer,
            groundedness_call.value,
        )
        passed = citation_audit.passed and groundedness_report.passed
        warnings.append("中风险策略未执行回答完整性评估。")
    else:
        passed = citation_audit.passed
        warnings.append(
            "低风险策略只证明引用结构合法，未验证语义支持度和完整性。"
        )

    report = AdaptiveAuditReport(
        decision=decision,
        policy=policy,
        citation_audit=citation_audit,
        groundedness_report=groundedness_report,
        quality_response=quality_response,
        final_quality_report=final_quality_report,
        passed=passed,
        model_calls=sum(call.model_calls for call in calls),
        total_tokens=sum(call.total_tokens for call in calls),
        wall_seconds=time.perf_counter() - started,
        warnings=warnings,
    )
    return rendered_answer, report


def print_report(answer: str, report: AdaptiveAuditReport) -> None:
    print("\n" + "=" * 88)
    print("风险决策：", report.decision.risk_level)
    print("决策来源：", report.decision.source)
    print("判断原因：", report.decision.reasons)
    print("审计策略：", report.policy.description)
    print("\n最终回答：")
    print(answer)

    print("\n审计结果：")
    print("引用覆盖率：", f"{report.citation_audit.claim_coverage_rate:.1%}")
    print(
        "严格证据支持率：",
        "未执行"
        if report.groundedness_report is None
        else f"{report.groundedness_report.groundedness_rate:.1%}",
    )
    print(
        "问题相关性：",
        "未执行"
        if report.quality_response is None
        else report.quality_response.relevance_score,
    )
    print(
        "回答完整性：",
        "未执行"
        if report.quality_response is None
        else report.quality_response.completeness_score,
    )
    print("最终判断：", "通过" if report.passed else "失败")
    print("模型调用数：", report.model_calls)
    print("Token总数：", report.total_tokens)
    print("墙钟耗时：", f"{report.wall_seconds:.3f}秒")

    if report.warnings:
        print("\n策略提示：")
        for warning in report.warnings:
            print("-", warning)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day24自适应审计练习")
    parser.add_argument("question", nargs="*", help="知识库问题")
    parser.add_argument(
        "--risk",
        choices=["auto", "low", "medium", "high"],
        default="auto",
        help="自动判断或手动覆盖风险等级",
    )
    return parser.parse_args()


async def main() -> None:
    arguments = parse_arguments()
    question = " ".join(arguments.question).strip() or (
        "什么是 RAG？它的工作流程是什么？"
    )
    decision = classify_risk(question, arguments.risk)

    print("Day24 风险分级与自适应审计")
    print("问题：", question)
    print("风险参数：", arguments.risk)
    print("分类结果：", decision.risk_level)
    print("分类原因：", decision.reasons)
    print("预计模型调用：", {"low": 2, "medium": 3, "high": 4}[decision.risk_level])

    answer, report = await run_adaptive_audit(question, decision)
    print_report(answer, report)

    print("\n本步数据流：")
    print("问题 → 确定性风险规则/人工覆盖 → AuditPolicy")
    print("→ 共同的Reranker和结构化回答")
    print("→ low / medium / high审计分支 → AdaptiveAuditReport")
    print("注意：减少审计意味着降低成本，也意味着接受更多残余风险。")


if __name__ == "__main__":
    asyncio.run(main())
