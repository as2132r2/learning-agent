"""Day23 第四步：综合评估回答的忠实度、相关性与完整性。

运行：
    python day23/python/day23_answer_quality.py "什么是 RAG？"

正常流程包含四次模型调用：Reranker、回答生成、Groundedness、质量评估。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY22_PYTHON = PROJECT_DIR / "day22" / "python"
DAY23_PYTHON = PROJECT_DIR / "day23" / "python"
for directory in (DAY22_PYTHON, DAY23_PYTHON):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from day22_final_hybrid_rag import (  # noqa: E402
    RERANK_THRESHOLD,
    retrieve_rrf_candidates,
)
from day22_llm_reranker import RerankedResult, rerank_candidates  # noqa: E402
from day23_citation_audit import (  # noqa: E402
    CitationAudit,
    audit_citations,
    generate_structured_answer,
)
from day23_claim_groundedness import (  # noqa: E402
    GroundednessReport,
    build_groundedness_report,
    evaluate_claim_support,
)
from day23_structured_citations import (  # noqa: E402
    StructuredRAGAnswer,
    build_readable_answer,
    format_evidence,
    select_evidence,
)


QUALITY_SCORE_THRESHOLD = 80


class AnswerQualityResponse(BaseModel):
    """质量评估模型返回的结构化结果。"""

    relevance_score: int = Field(
        ge=0,
        le=100,
        description="回答切中用户问题的程度",
    )
    completeness_score: int = Field(
        ge=0,
        le=100,
        description="在参考资料范围内覆盖问题重要要点的程度",
    )
    addressed_points: list[str] = Field(
        default_factory=list,
        description="回答已经覆盖的重要要点"
    )
    missing_points: list[str] = Field(
        default_factory=list,
        description="资料可以支持、问题也要求，但回答遗漏的要点"
    )
    irrelevant_points: list[str] = Field(
        default_factory=list,
        description="回答中偏离问题的内容"
    )
    explanation: str = Field(
        default="",
        description="整体评分理由；模型省略时使用空字符串",
    )


@dataclass
class FinalQualityReport:
    """合并确定性检查和两种模型评估后的质量报告。"""

    citation_coverage_rate: float
    groundedness_rate: float
    relevance_score: int
    completeness_score: int
    overall_score: float
    addressed_points: list[str]
    missing_points: list[str]
    irrelevant_points: list[str]
    explanation: str
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def create_quality_model():
    """创建只评估问题相关性和回答完整性的结构化模型。"""

    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    return llm.with_structured_output(
        AnswerQualityResponse,
        method="function_calling",
    )


def evaluate_answer_quality(
    question: str,
    answer: StructuredRAGAnswer,
    evidence: list[RerankedResult],
) -> AnswerQualityResponse:
    """基于用户问题与合格资料评估相关性和完整性。"""

    quality_model = create_quality_model()
    evidence_text = format_evidence(evidence)
    rendered_answer = build_readable_answer(answer)

    return quality_model.invoke(
        [
            SystemMessage(
                content=(
                    "你是RAG回答质量评估员，不重新回答问题。"
                    "只根据用户问题和参考资料判断现有回答。"
                    "relevance_score衡量回答是否直接回应问题；"
                    "completeness_score衡量资料中能够回答且问题要求的"
                    "重要要点是否都被覆盖。不要因回答省略与问题无关的"
                    "资料而扣分，也不得要求资料之外的外部知识。"
                    "missing_points只能填写资料支持且问题要求的遗漏内容。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：\n{question}\n\n"
                    f"参考资料：\n{evidence_text}\n\n"
                    f"待评估回答：\n{rendered_answer}"
                )
            ),
        ]
    )


def build_final_quality_report(
    citation_audit: CitationAudit,
    groundedness_report: GroundednessReport,
    quality_response: AnswerQualityResponse,
) -> FinalQualityReport:
    """按最短板原则汇总四类质量指标。"""

    errors: list[str] = []

    if not citation_audit.passed:
        errors.extend(
            f"引用审计：{error}"
            for error in citation_audit.errors
        )
    if not groundedness_report.passed:
        errors.extend(
            f"忠实度：{error}"
            for error in groundedness_report.errors
        )
    if quality_response.relevance_score < QUALITY_SCORE_THRESHOLD:
        errors.append(
            f"问题相关性只有{quality_response.relevance_score}分，"
            f"低于{QUALITY_SCORE_THRESHOLD}分。"
        )
    if quality_response.completeness_score < QUALITY_SCORE_THRESHOLD:
        errors.append(
            f"回答完整性只有{quality_response.completeness_score}分，"
            f"低于{QUALITY_SCORE_THRESHOLD}分。"
        )
    if quality_response.missing_points:
        errors.append(
            f"存在{len(quality_response.missing_points)}个重要遗漏点。"
        )
    if quality_response.irrelevant_points:
        errors.append(
            f"存在{len(quality_response.irrelevant_points)}个偏题内容。"
        )

    # 使用最短板而非平均值，避免高分指标掩盖某个严重缺陷。
    overall_score = min(
        citation_audit.claim_coverage_rate * 100,
        groundedness_report.groundedness_rate * 100,
        quality_response.relevance_score,
        quality_response.completeness_score,
    )

    return FinalQualityReport(
        citation_coverage_rate=citation_audit.claim_coverage_rate,
        groundedness_rate=groundedness_report.groundedness_rate,
        relevance_score=quality_response.relevance_score,
        completeness_score=quality_response.completeness_score,
        overall_score=overall_score,
        addressed_points=quality_response.addressed_points,
        missing_points=quality_response.missing_points,
        irrelevant_points=quality_response.irrelevant_points,
        explanation=quality_response.explanation,
        errors=errors,
    )


def print_final_report(report: FinalQualityReport) -> None:
    print("引用覆盖率：", f"{report.citation_coverage_rate:.1%}")
    print("严格证据支持率：", f"{report.groundedness_rate:.1%}")
    print("问题相关性：", report.relevance_score)
    print("回答完整性：", report.completeness_score)
    print("综合分（最短板）：", round(report.overall_score, 2))
    print("最终判断：", "通过" if report.passed else "失败")

    print("\n已经覆盖的要点：")
    if report.addressed_points:
        for point in report.addressed_points:
            print("-", point)
    else:
        print("- 无")

    print("\n遗漏的重要要点：")
    if report.missing_points:
        for point in report.missing_points:
            print("-", point)
    else:
        print("- 无")

    print("\n偏离问题的内容：")
    if report.irrelevant_points:
        for point in report.irrelevant_points:
            print("-", point)
    else:
        print("- 无")

    print("\n评估说明：", report.explanation)
    if report.errors:
        print("\n未通过原因：")
        for error in report.errors:
            print("-", error)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    print("Day23 RAG回答综合质量评估")
    print("问题：", question)
    print("相关性/完整性阈值：", QUALITY_SCORE_THRESHOLD)
    print("=" * 88)

    rrf_results, _ = retrieve_rrf_candidates(question)
    reranked_results, _ = rerank_candidates(question, rrf_results)
    evidence = select_evidence(reranked_results)

    print("Reranker最高分：", reranked_results[0].relevance_score)
    print("Reranker守门阈值：", RERANK_THRESHOLD)
    print("合格证据数：", len(evidence))
    if not evidence:
        print("没有合格证据，回答生成前已经拒绝。")
        return

    answer = generate_structured_answer(question, evidence)
    rendered_answer = build_readable_answer(answer)
    citation_audit = audit_citations(
        answer,
        len(evidence),
        rendered_answer,
    )

    if not citation_audit.passed:
        print("引用结构审计失败，停止后续质量评估：")
        for error in citation_audit.errors:
            print("-", error)
        return

    groundedness_response = evaluate_claim_support(answer, evidence)
    groundedness_report = build_groundedness_report(
        answer,
        groundedness_response,
    )

    print("正在评估问题相关性与回答完整性……")
    quality_response = evaluate_answer_quality(
        question,
        answer,
        evidence,
    )
    final_report = build_final_quality_report(
        citation_audit,
        groundedness_report,
        quality_response,
    )

    print("\n" + "=" * 88)
    print("待评估回答：")
    print(rendered_answer)

    print("\n" + "=" * 88)
    print("最终质量报告：")
    print_final_report(final_report)

    print("\n" + "=" * 88)
    print("AnswerQualityResponse类型：", type(quality_response))
    print("FinalQualityReport类型：", type(final_report))
    print("overall_score类型：", type(final_report.overall_score))

    print("\n本步数据流：")
    print("问题 + 合格资料 + 结构化回答")
    print("├→ CitationAudit：引用结构和覆盖率")
    print("├→ GroundednessReport：已说事实是否有证据")
    print("└→ AnswerQualityResponse：是否切题、是否完整")
    print("→ 最短板汇总 → FinalQualityReport")


if __name__ == "__main__":
    main()
