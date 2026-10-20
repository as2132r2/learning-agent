"""Day23 第一步：生成可以被程序检查的结构化RAG引用回答。

运行默认问题：
    python day23/python/day23_structured_citations.py

指定问题：
    python day23/python/day23_structured_citations.py "什么是 RAG？"

注意：候选正文会发送给配置的DeepSeek API进行重排和回答生成。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY22_PYTHON = PROJECT_DIR / "day22" / "python"
if str(DAY22_PYTHON) not in sys.path:
    sys.path.insert(0, str(DAY22_PYTHON))

from day22_final_hybrid_rag import (  # noqa: E402
    RERANK_THRESHOLD,
    retrieve_rrf_candidates,
)
from day22_llm_reranker import RerankedResult, rerank_candidates  # noqa: E402


MAX_EVIDENCE_COUNT = 4


class CitedClaim(BaseModel):
    """回答中的一条结论以及支持它的资料编号。"""

    claim: str = Field(description="只根据参考资料得到的一条明确结论")
    citation_numbers: list[int] = Field(
        min_length=1,
        description="支持该结论的资料编号，例如[1]或[1, 2]",
    )


class StructuredRAGAnswer(BaseModel):
    """模型必须返回的完整结构化回答。"""

    can_answer: bool = Field(description="参考资料是否足以回答用户问题")
    answer: str = Field(description="面向用户的简洁回答；资料不足时说明不足")
    claims: list[CitedClaim] = Field(
        description="回答中的可验证结论；资料不足时返回空列表"
    )


def create_answer_model():
    """创建返回StructuredRAGAnswer的结构化模型。"""

    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    return llm.with_structured_output(
        StructuredRAGAnswer,
        method="function_calling",
    )


def select_evidence(
    reranked_results: list[RerankedResult],
) -> list[RerankedResult]:
    """只选择达到Day22评测阈值的资料。"""

    return [
        item
        for item in reranked_results
        if item.relevance_score >= RERANK_THRESHOLD
    ][:MAX_EVIDENCE_COUNT]


def format_evidence(evidence: list[RerankedResult]) -> str:
    """把精排资料转换成带稳定编号的证据文本。"""

    blocks: list[str] = []
    for number, item in enumerate(evidence, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[资料{number}]",
                    f"chunk_id: {item.chunk_id}",
                    f"来源: {item.location}",
                    f"相关性: {item.relevance_score}",
                    f"正文:\n{item.content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def validate_citations(
    answer: StructuredRAGAnswer,
    evidence_count: int,
) -> list[str]:
    """执行第一层确定性检查，返回发现的问题。"""

    errors: list[str] = []

    if answer.can_answer and not answer.claims:
        errors.append("can_answer=True，但claims为空。")

    if not answer.can_answer and answer.claims:
        errors.append("can_answer=False，但仍然返回了claims。")

    for claim_number, claim in enumerate(answer.claims, start=1):
        if not claim.claim.strip():
            errors.append(f"第{claim_number}条结论是空字符串。")

        duplicate_numbers = len(claim.citation_numbers) != len(
            set(claim.citation_numbers)
        )
        if duplicate_numbers:
            errors.append(f"第{claim_number}条结论包含重复引用编号。")

        for citation_number in claim.citation_numbers:
            if not 1 <= citation_number <= evidence_count:
                errors.append(
                    f"第{claim_number}条结论引用了不存在的"
                    f"资料{citation_number}。"
                )

    return errors


def build_readable_answer(answer: StructuredRAGAnswer) -> str:
    """把结构化结果渲染成人类易读文本。"""

    if not answer.can_answer:
        return answer.answer

    lines = [answer.answer, "", "可验证结论："]
    for claim in answer.claims:
        citations = "".join(
            f"[资料{number}]" for number in claim.citation_numbers
        )
        lines.append(f"- {claim.claim} {citations}")
    return "\n".join(lines)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    print("Day23 结构化引用回答")
    print("问题：", question)
    print("Day22守门阈值：", RERANK_THRESHOLD)
    print("=" * 88)

    rrf_results, diagnostics = retrieve_rrf_candidates(question)
    print("RRF候选数：", len(rrf_results))
    print("关键词：", diagnostics["terms"])
    print("正在调用Reranker……")

    reranked_results, _ = rerank_candidates(question, rrf_results)
    evidence = select_evidence(reranked_results)

    print("Reranker最高分：", reranked_results[0].relevance_score)
    print("通过阈值的证据数量：", len(evidence))

    if not evidence:
        print("\n知识库中没有达到阈值的资料，程序在生成回答前拒绝。")
        return

    evidence_text = format_evidence(evidence)
    print("\n" + "=" * 88)
    print("发送给回答模型的证据：")
    print(evidence_text)

    answer_model = create_answer_model()
    answer: StructuredRAGAnswer = answer_model.invoke(
        [
            SystemMessage(
                content=(
                    "你是严格的RAG回答器。只能使用给定资料，不能使用"
                    "外部知识。把回答拆成可以独立验证的结论，并为每条"
                    "结论填写真正支持它的资料编号。不得引用不存在的资料。"
                    "如果资料不足，can_answer设为false，claims返回空列表。"
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

    citation_errors = validate_citations(answer, len(evidence))

    print("\n" + "=" * 88)
    print("结构化回答对象：")
    print(answer)
    print("\n对象类型：", type(answer))
    print("can_answer类型：", type(answer.can_answer))
    print("claims类型：", type(answer.claims))

    print("\n" + "=" * 88)
    print("引用结构检查：")
    if citation_errors:
        for error in citation_errors:
            print("- 失败：", error)
    else:
        print("通过：每条结论都有引用，所有引用编号都真实存在。")

    print("\n" + "=" * 88)
    print("面向用户渲染后的回答：")
    print(build_readable_answer(answer))

    print("\n" + "=" * 88)
    print("本步数据流：")
    print("问题 → 混合检索 → RRF → Reranker → 达标证据")
    print("→ StructuredRAGAnswer → 确定性引用编号检查 → 人类可读回答")
    print("注意：本步只验证引用编号存在，下一步检查引用解析和覆盖关系。")


if __name__ == "__main__":
    main()
