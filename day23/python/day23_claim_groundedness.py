"""Day23 第三步：检查每条结论是否真的被所引用的资料支持。

运行：
    python day23/python/day23_claim_groundedness.py "什么是 RAG？"

正常流程包含三次模型调用：Reranker、回答生成、Groundedness评估。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
from day23_structured_citations import (  # noqa: E402
    StructuredRAGAnswer,
    build_readable_answer,
    select_evidence,
)


SUPPORT_SCORE_THRESHOLD = 80


class ClaimSupportJudgment(BaseModel):
    """评估模型对一条结论和其引用资料的判断。"""

    claim_number: int = Field(ge=1, description="从1开始的结论编号")
    verdict: Literal[
        "fully_supported",
        "partially_supported",
        "unsupported",
    ] = Field(description="引用资料对结论的支持程度")
    support_score: int = Field(
        ge=0,
        le=100,
        description="资料支持结论的程度，0到100分",
    )
    explanation: str = Field(description="根据引用资料给出的简短判断理由")


class GroundednessResponse(BaseModel):
    """一次批量证据支持度评估的结构化输出。"""

    judgments: list[ClaimSupportJudgment] = Field(
        description="按结论编号逐条返回判断，不能遗漏或重复"
    )


@dataclass
class GroundednessReport:
    """经过编号校验和统计后的最终证据支持度报告。"""

    claim_count: int
    fully_supported_count: int
    partially_supported_count: int
    unsupported_count: int
    grounded_claim_count: int
    groundedness_rate: float
    judgments: list[ClaimSupportJudgment]
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors and self.groundedness_rate == 1.0


def create_groundedness_model():
    """创建只负责判断证据支持度的结构化模型。"""

    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    return llm.with_structured_output(
        GroundednessResponse,
        method="function_calling",
    )


def format_claim_evidence_pairs(
    answer: StructuredRAGAnswer,
    evidence: list[RerankedResult],
) -> str:
    """每条结论只附带它声明引用的资料正文。"""

    sections: list[str] = []
    for claim_number, claim in enumerate(answer.claims, start=1):
        cited_blocks: list[str] = []
        for citation_number in claim.citation_numbers:
            # 引用审计已在调用该函数前保证编号合法。
            cited_item = evidence[citation_number - 1]
            cited_blocks.append(
                f"[资料{citation_number}｜{cited_item.location}]\n"
                f"{cited_item.content}"
            )

        sections.append(
            f"结论{claim_number}：{claim.claim}\n"
            f"该结论声明的引用：\n"
            + "\n\n".join(cited_blocks)
        )

    return ("\n\n" + "=" * 60 + "\n\n").join(sections)


def evaluate_claim_support(
    answer: StructuredRAGAnswer,
    evidence: list[RerankedResult],
) -> GroundednessResponse:
    """批量调用评估模型，判断每一条claim的证据支持度。"""

    evaluator = create_groundedness_model()
    pairs_text = format_claim_evidence_pairs(answer, evidence)

    return evaluator.invoke(
        [
            SystemMessage(
                content=(
                    "你是严格的RAG证据审查员，不回答用户问题。"
                    "请逐条判断引用正文是否直接支持结论。"
                    "fully_supported表示结论全部能从引用中得到；"
                    "partially_supported表示只有一部分能得到，或结论扩大了范围；"
                    "unsupported表示引用不支持或与结论矛盾。"
                    "不得使用外部常识，也不得使用该结论没有引用的其他资料。"
                    "必须逐条返回，保留原始结论编号。"
                )
            ),
            HumanMessage(
                content=f"需要审查的结论与引用如下：\n\n{pairs_text}"
            ),
        ]
    )


def build_groundedness_report(
    answer: StructuredRAGAnswer,
    response: GroundednessResponse,
) -> GroundednessReport:
    """校验评估模型的编号完整性，再计算严格支持率。"""

    errors: list[str] = []
    expected_numbers = set(range(1, len(answer.claims) + 1))
    returned_numbers = [item.claim_number for item in response.judgments]
    returned_set = set(returned_numbers)

    duplicate_numbers = sorted(
        number
        for number in returned_set
        if returned_numbers.count(number) > 1
    )
    missing_numbers = sorted(expected_numbers - returned_set)
    unknown_numbers = sorted(returned_set - expected_numbers)

    if duplicate_numbers:
        errors.append(f"评估模型重复返回结论编号：{duplicate_numbers}。")
    if missing_numbers:
        errors.append(f"评估模型遗漏结论编号：{missing_numbers}。")
    if unknown_numbers:
        errors.append(f"评估模型虚构结论编号：{unknown_numbers}。")

    valid_judgments = [
        item
        for item in response.judgments
        if item.claim_number in expected_numbers
    ]
    valid_judgments.sort(key=lambda item: item.claim_number)

    fully_supported_count = sum(
        item.verdict == "fully_supported"
        for item in valid_judgments
    )
    partially_supported_count = sum(
        item.verdict == "partially_supported"
        for item in valid_judgments
    )
    unsupported_count = sum(
        item.verdict == "unsupported"
        for item in valid_judgments
    )

    # 严格通过要求：verdict完全支持，并且支持分数达到阈值。
    grounded_claim_count = sum(
        item.verdict == "fully_supported"
        and item.support_score >= SUPPORT_SCORE_THRESHOLD
        for item in valid_judgments
    )
    claim_count = len(answer.claims)
    groundedness_rate = (
        grounded_claim_count / claim_count
        if claim_count
        else (1.0 if not answer.can_answer else 0.0)
    )

    if groundedness_rate < 1.0:
        errors.append(
            f"严格证据支持率为{groundedness_rate:.1%}，低于100%。"
        )

    return GroundednessReport(
        claim_count=claim_count,
        fully_supported_count=fully_supported_count,
        partially_supported_count=partially_supported_count,
        unsupported_count=unsupported_count,
        grounded_claim_count=grounded_claim_count,
        groundedness_rate=groundedness_rate,
        judgments=valid_judgments,
        errors=errors,
    )


def print_groundedness_report(report: GroundednessReport) -> None:
    print("结论总数：", report.claim_count)
    print("完全支持：", report.fully_supported_count)
    print("部分支持：", report.partially_supported_count)
    print("不支持：", report.unsupported_count)
    print("达到严格标准的结论数：", report.grounded_claim_count)
    print("严格证据支持率：", f"{report.groundedness_rate:.1%}")
    print("评估结果：", "通过" if report.passed else "失败")

    for judgment in report.judgments:
        print("\n" + "-" * 88)
        print("结论编号：", judgment.claim_number)
        print("判断：", judgment.verdict)
        print("支持分数：", judgment.support_score)
        print("理由：", judgment.explanation)

    if report.errors:
        print("\n发现的问题：")
        for error in report.errors:
            print("-", error)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    print("Day23 结论—证据支持度检查")
    print("问题：", question)
    print("严格支持分数阈值：", SUPPORT_SCORE_THRESHOLD)
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
    citation_audit: CitationAudit = audit_citations(
        answer,
        len(evidence),
        rendered_answer,
    )

    print("\n结构化回答：")
    print(rendered_answer)
    print("\n引用结构审计：", "通过" if citation_audit.passed else "失败")
    if not citation_audit.passed:
        for error in citation_audit.errors:
            print("-", error)
        print("引用结构不合法，停止语义支持度评估。")
        return

    print("\n正在逐条评估结论是否被引用正文支持……")
    response = evaluate_claim_support(answer, evidence)
    report = build_groundedness_report(answer, response)

    print("\n" + "=" * 88)
    print("Groundedness评估报告：")
    print_groundedness_report(report)

    print("\n" + "=" * 88)
    print("GroundednessResponse类型：", type(response))
    print("GroundednessReport类型：", type(report))
    print("groundedness_rate类型：", type(report.groundedness_rate))

    print("\n本步数据流：")
    print("合法StructuredRAGAnswer + 被每条claim引用的证据")
    print("→ Groundedness评估模型 → GroundednessResponse")
    print("→ 编号完整性检查 + 严格分数统计 → GroundednessReport")
    print("结论：引用存在不等于证据支持，二者必须分层检查。")


if __name__ == "__main__":
    main()
