"""Day23 第五步：不合格RAG回答的受控修复与重新审计。

正常运行：
    python day23/python/day23_answer_repair.py

稳定演示一次修复：
    python day23/python/day23_answer_repair.py --demo-incomplete

指定问题：
    python day23/python/day23_answer_repair.py "什么是 RAG？"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage


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
from day23_answer_quality import (  # noqa: E402
    AnswerQualityResponse,
    FinalQualityReport,
    build_final_quality_report,
    evaluate_answer_quality,
)
from day23_citation_audit import CitationAudit, audit_citations  # noqa: E402
from day23_claim_groundedness import (  # noqa: E402
    GroundednessReport,
    build_groundedness_report,
    evaluate_claim_support,
)
from day23_structured_citations import (  # noqa: E402
    StructuredRAGAnswer,
    build_readable_answer,
    create_answer_model,
    format_evidence,
    select_evidence,
)
from day23_citation_audit import generate_structured_answer  # noqa: E402


MAX_REPAIR_ATTEMPTS = 1


@dataclass
class AnswerAssessment:
    """一版回答经过全部可执行检查后的结果。"""

    answer: StructuredRAGAnswer
    rendered_answer: str
    citation_audit: CitationAudit
    groundedness_report: GroundednessReport | None
    quality_response: AnswerQualityResponse | None
    final_report: FinalQualityReport | None

    @property
    def passed(self) -> bool:
        return (
            self.citation_audit.passed
            and self.final_report is not None
            and self.final_report.passed
        )


@dataclass
class AnswerVersion:
    """保存初稿或修复稿，便于比较每一版结果。"""

    version_number: int
    label: str
    assessment: AnswerAssessment


def assess_answer(
    question: str,
    answer: StructuredRAGAnswer,
    evidence: list[RerankedResult],
) -> AnswerAssessment:
    """依次运行引用审计、Groundedness和回答质量评估。"""

    rendered_answer = build_readable_answer(answer)
    citation_audit = audit_citations(
        answer,
        len(evidence),
        rendered_answer,
    )

    if not citation_audit.passed:
        return AnswerAssessment(
            answer=answer,
            rendered_answer=rendered_answer,
            citation_audit=citation_audit,
            groundedness_report=None,
            quality_response=None,
            final_report=None,
        )

    groundedness_response = evaluate_claim_support(answer, evidence)
    groundedness_report = build_groundedness_report(
        answer,
        groundedness_response,
    )
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

    return AnswerAssessment(
        answer=answer,
        rendered_answer=rendered_answer,
        citation_audit=citation_audit,
        groundedness_report=groundedness_report,
        quality_response=quality_response,
        final_report=final_report,
    )


def assessment_feedback(assessment: AnswerAssessment) -> str:
    """把机器可读报告转换成修复模型需要的具体反馈。"""

    issues: list[str] = []

    issues.extend(
        f"引用结构问题：{error}"
        for error in assessment.citation_audit.errors
    )

    if assessment.groundedness_report is not None:
        issues.extend(
            f"证据支持问题：{error}"
            for error in assessment.groundedness_report.errors
        )
        for judgment in assessment.groundedness_report.judgments:
            if not (
                judgment.verdict == "fully_supported"
                and judgment.support_score >= 80
            ):
                issues.append(
                    f"结论{judgment.claim_number}未被充分支持："
                    f"{judgment.explanation}"
                )

    if assessment.quality_response is not None:
        quality = assessment.quality_response
        issues.extend(
            f"遗漏要点：{point}"
            for point in quality.missing_points
        )
        issues.extend(
            f"偏题内容：{point}"
            for point in quality.irrelevant_points
        )
        if quality.relevance_score < 80:
            issues.append(f"问题相关性只有{quality.relevance_score}分。")
        if quality.completeness_score < 80:
            issues.append(f"回答完整性只有{quality.completeness_score}分。")

    if not issues:
        issues.append("自动评估未通过，请严格缩减为资料能直接支持的完整回答。")

    return "\n".join(f"- {issue}" for issue in issues)


def repair_answer(
    question: str,
    evidence: list[RerankedResult],
    assessment: AnswerAssessment,
) -> StructuredRAGAnswer:
    """只使用原证据和审计反馈修复当前回答。"""

    repair_model = create_answer_model()
    evidence_text = format_evidence(evidence)
    feedback = assessment_feedback(assessment)

    return repair_model.invoke(
        [
            SystemMessage(
                content=(
                    "你是RAG回答修复器。只能使用给定参考资料，不能"
                    "加入外部知识。根据审计反馈修复遗漏、偏题、不受支持"
                    "或引用错误。每条claim必须是单一、可验证的结论，"
                    "并只引用真正支持它的资料编号。不要解释修复过程，"
                    "直接返回修复后的结构化回答。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：\n{question}\n\n"
                    f"参考资料：\n{evidence_text}\n\n"
                    f"当前回答：\n{assessment.rendered_answer}\n\n"
                    f"必须修复的问题：\n{feedback}"
                )
            ),
        ]
    )


def make_incomplete_demo(
    answer: StructuredRAGAnswer,
) -> StructuredRAGAnswer:
    """教学模式：只保留第一条结论，稳定制造完整性缺陷。"""

    if not answer.claims:
        return answer

    first_claim = answer.claims[0]
    return StructuredRAGAnswer(
        can_answer=True,
        answer=first_claim.claim,
        claims=[first_claim],
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day23回答自动修复练习")
    parser.add_argument(
        "question",
        nargs="*",
        help="要向知识库提出的问题",
    )
    parser.add_argument(
        "--demo-incomplete",
        action="store_true",
        help="把模型初稿裁剪成不完整回答，稳定演示修复循环",
    )
    return parser.parse_args()


def print_version(version: AnswerVersion) -> None:
    assessment = version.assessment
    print("\n" + "=" * 88)
    print(f"版本{version.version_number}：{version.label}")
    print("=" * 88)
    print(assessment.rendered_answer)
    print("\n引用审计：", "通过" if assessment.citation_audit.passed else "失败")

    if assessment.groundedness_report is not None:
        print(
            "严格证据支持率：",
            f"{assessment.groundedness_report.groundedness_rate:.1%}",
        )
    if assessment.quality_response is not None:
        print("相关性：", assessment.quality_response.relevance_score)
        print("完整性：", assessment.quality_response.completeness_score)
        print("遗漏点：", assessment.quality_response.missing_points)
        print("偏题内容：", assessment.quality_response.irrelevant_points)
    if assessment.final_report is not None:
        print("综合分：", assessment.final_report.overall_score)

    print("本版结论：", "通过" if assessment.passed else "需要修复")
    if not assessment.passed:
        print("反馈给修复模型：")
        print(assessment_feedback(assessment))


def main() -> None:
    arguments = parse_arguments()
    question = " ".join(arguments.question).strip() or (
        "什么是 RAG？它的工作流程是什么？"
    )

    print("Day23 不合格回答受控修复")
    print("问题：", question)
    print("最大修复次数：", MAX_REPAIR_ATTEMPTS)
    print("教学缺陷模式：", arguments.demo_incomplete)
    print("=" * 88)

    rrf_results, _ = retrieve_rrf_candidates(question)
    reranked_results, _ = rerank_candidates(question, rrf_results)
    evidence = select_evidence(reranked_results)

    print("Reranker最高分：", reranked_results[0].relevance_score)
    print("Reranker守门阈值：", RERANK_THRESHOLD)
    print("合格证据数：", len(evidence))
    if not evidence:
        print("没有合格证据，不能生成或修复回答。")
        return

    initial_answer = generate_structured_answer(question, evidence)
    if arguments.demo_incomplete:
        initial_answer = make_incomplete_demo(initial_answer)
        initial_label = "教学模式裁剪后的不完整初稿"
    else:
        initial_label = "模型初稿"

    versions: list[AnswerVersion] = []
    current_assessment = assess_answer(
        question,
        initial_answer,
        evidence,
    )
    versions.append(AnswerVersion(1, initial_label, current_assessment))
    print_version(versions[-1])

    repair_attempt = 0
    while not current_assessment.passed and repair_attempt < MAX_REPAIR_ATTEMPTS:
        repair_attempt += 1
        print(f"\n开始第{repair_attempt}次修复……")
        repaired_answer = repair_answer(
            question,
            evidence,
            current_assessment,
        )
        current_assessment = assess_answer(
            question,
            repaired_answer,
            evidence,
        )
        version = AnswerVersion(
            version_number=len(versions) + 1,
            label=f"第{repair_attempt}次修复稿",
            assessment=current_assessment,
        )
        versions.append(version)
        print_version(version)

    print("\n" + "=" * 88)
    print("修复流程结束")
    print("总版本数：", len(versions))
    print("实际修复次数：", repair_attempt)
    print("最终状态：", "通过，可返回用户" if current_assessment.passed else "失败，应拒绝返回")

    print("\n本步数据流：")
    print("初稿 → 全量审计")
    print("├─ 通过 → 直接返回")
    print("└─ 失败 → 错误/遗漏/不支持结论组成修复反馈")
    print("          → 最多修复1次 → 再次全量审计")
    print("          ├─ 通过 → 返回修复稿")
    print("          └─ 失败 → 拒绝返回，禁止无限循环")


if __name__ == "__main__":
    main()
