"""Day23 第二步：解析引用并生成确定性的引用审计报告。

运行：
    python day23/python/day23_citation_audit.py "什么是 RAG？"

注意：候选正文会发送给配置的DeepSeek API进行重排和回答生成。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
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
from day23_structured_citations import (  # noqa: E402
    StructuredRAGAnswer,
    build_readable_answer,
    create_answer_model,
    format_evidence,
    select_evidence,
)


CITATION_PATTERN = re.compile(r"\[资料(\d+)\]")


@dataclass
class CitationAudit:
    """一次引用审计产生的全部确定性指标和错误。"""

    can_answer: bool
    claim_count: int
    cited_claim_count: int
    claim_coverage_rate: float
    evidence_count: int
    structured_citations: list[int]
    rendered_citations: list[int]
    used_evidence_numbers: list[int]
    unused_evidence_numbers: list[int]
    invalid_citation_numbers: list[int]
    duplicate_citations_by_claim: dict[int, list[int]]
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def parse_rendered_citations(text: str) -> list[int]:
    """使用正则表达式从最终文本中提取[资料N]编号。"""

    return [int(match) for match in CITATION_PATTERN.findall(text)]


def find_duplicates(numbers: list[int]) -> list[int]:
    """返回列表中重复出现过的编号，每个重复编号只返回一次。"""

    seen: set[int] = set()
    duplicates: set[int] = set()
    for number in numbers:
        if number in seen:
            duplicates.add(number)
        seen.add(number)
    return sorted(duplicates)


def audit_citations(
    answer: StructuredRAGAnswer,
    evidence_count: int,
    rendered_answer: str,
) -> CitationAudit:
    """审计结论覆盖率、引用范围、重复项和渲染一致性。"""

    errors: list[str] = []
    claim_count = len(answer.claims)
    cited_claim_count = sum(
        bool(claim.citation_numbers)
        for claim in answer.claims
    )
    coverage_rate = (
        cited_claim_count / claim_count
        if claim_count
        else (1.0 if not answer.can_answer else 0.0)
    )

    structured_citations = [
        number
        for claim in answer.claims
        for number in claim.citation_numbers
    ]
    rendered_citations = parse_rendered_citations(rendered_answer)

    valid_range = set(range(1, evidence_count + 1))
    used_numbers = sorted(set(structured_citations) & valid_range)
    unused_numbers = sorted(valid_range - set(structured_citations))
    invalid_numbers = sorted(set(structured_citations) - valid_range)

    duplicate_by_claim: dict[int, list[int]] = {}
    for claim_index, claim in enumerate(answer.claims, start=1):
        duplicates = find_duplicates(claim.citation_numbers)
        if duplicates:
            duplicate_by_claim[claim_index] = duplicates

    if answer.can_answer and claim_count == 0:
        errors.append("回答声称可以回答，但没有可审计结论。")
    if not answer.can_answer and claim_count > 0:
        errors.append("回答声称资料不足，但仍返回了事实结论。")
    if coverage_rate < 1.0:
        errors.append(
            f"结论引用覆盖率只有{coverage_rate:.1%}，不是100%。"
        )
    if invalid_numbers:
        errors.append(f"存在越界引用：{invalid_numbers}。")
    if duplicate_by_claim:
        errors.append(f"同一结论中存在重复引用：{duplicate_by_claim}。")

    # Counter式比较不仅核对使用了哪些编号，也核对每个编号出现次数。
    if sorted(structured_citations) != sorted(rendered_citations):
        errors.append(
            "结构化claims中的引用，与最终渲染文本中的引用不一致。"
        )

    return CitationAudit(
        can_answer=answer.can_answer,
        claim_count=claim_count,
        cited_claim_count=cited_claim_count,
        claim_coverage_rate=coverage_rate,
        evidence_count=evidence_count,
        structured_citations=structured_citations,
        rendered_citations=rendered_citations,
        used_evidence_numbers=used_numbers,
        unused_evidence_numbers=unused_numbers,
        invalid_citation_numbers=invalid_numbers,
        duplicate_citations_by_claim=duplicate_by_claim,
        errors=errors,
    )


def generate_structured_answer(
    question: str,
    evidence: list[RerankedResult],
) -> StructuredRAGAnswer:
    """根据已经通过Reranker守门的证据生成结构化回答。"""

    evidence_text = format_evidence(evidence)
    answer_model = create_answer_model()
    return answer_model.invoke(
        [
            SystemMessage(
                content=(
                    "你是严格的RAG回答器，只能使用给定资料。"
                    "把回答拆成可独立验证的结论，并为每条结论填写"
                    "真正支持它的资料编号。不能使用外部知识，不能"
                    "引用不存在的资料。如果资料不足，can_answer设为false，"
                    "claims必须为空列表。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：\n{question}\n\n"
                    f"参考资料：\n{evidence_text}"
                )
            ),
        ]
    )


def print_audit(audit: CitationAudit) -> None:
    """以便于学习的方式打印审计对象。"""

    print("can_answer：", audit.can_answer)
    print("结论总数：", audit.claim_count)
    print("带引用的结论数：", audit.cited_claim_count)
    print("结论引用覆盖率：", f"{audit.claim_coverage_rate:.1%}")
    print("证据总数：", audit.evidence_count)
    print("结构化引用序列：", audit.structured_citations)
    print("渲染文本引用序列：", audit.rendered_citations)
    print("已使用资料：", audit.used_evidence_numbers)
    print("未使用资料：", audit.unused_evidence_numbers)
    print("越界引用：", audit.invalid_citation_numbers)
    print("各结论重复引用：", audit.duplicate_citations_by_claim)
    print("审计结果：", "通过" if audit.passed else "失败")

    if audit.errors:
        print("发现的问题：")
        for error in audit.errors:
            print("-", error)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    print("Day23 引用解析与覆盖率审计")
    print("问题：", question)
    print("引用格式： [资料N]")
    print("=" * 88)

    rrf_results, _ = retrieve_rrf_candidates(question)
    reranked_results, _ = rerank_candidates(question, rrf_results)
    evidence = select_evidence(reranked_results)

    print("Reranker最高分：", reranked_results[0].relevance_score)
    print("守门阈值：", RERANK_THRESHOLD)
    print("通过阈值的资料数量：", len(evidence))

    if not evidence:
        print("没有合格证据，生成阶段不会执行。")
        return

    answer = generate_structured_answer(question, evidence)
    rendered_answer = build_readable_answer(answer)
    audit = audit_citations(answer, len(evidence), rendered_answer)

    print("\n" + "=" * 88)
    print("最终渲染文本：")
    print(rendered_answer)

    print("\n" + "=" * 88)
    print("引用审计报告：")
    print_audit(audit)

    print("\n" + "=" * 88)
    print("审计对象的数据类型：", type(audit))
    print("覆盖率的数据类型：", type(audit.claim_coverage_rate))
    print("引用序列的数据类型：", type(audit.structured_citations))
    print("passed是@property计算属性：", audit.passed)

    print("\n本步数据流：")
    print("StructuredRAGAnswer → claims引用编号展开")
    print("渲染文本 → 正则表达式 → [资料N]编号")
    print("→ 两套编号对照 + 覆盖率/越界/重复检查 → CitationAudit")
    print("下一步：检查资料正文是否真的支持每一条claim。")


if __name__ == "__main__":
    main()
