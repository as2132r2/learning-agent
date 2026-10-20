"""Day22 第四步：在RRF候选集上使用LLM进行二次重排。

运行默认问题：
    python day22/python/day22_llm_reranker.py

指定问题：
    python day22/python/day22_llm_reranker.py "什么是 RAG？"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from day22_retrieval_compare import fts_search, vector_search
from day22_rrf_fusion import RRFResult, reciprocal_rank_fusion


RRF_CANDIDATE_COUNT = 5
FINAL_RESULT_COUNT = 3


class RerankItem(BaseModel):
    """模型对一个候选文本块给出的判断。"""

    chunk_id: str = Field(description="候选文本块的原始chunk_id")
    relevance_score: int = Field(
        ge=0,
        le=100,
        description="正文回答用户问题的相关程度，0到100分",
    )
    reason: str = Field(description="一句话评分理由")


class RerankResponse(BaseModel):
    """一次批量重排的结构化模型输出。"""

    results: list[RerankItem] = Field(
        description="对所有候选逐项评分，不能遗漏或虚构chunk_id"
    )


@dataclass
class RerankedResult:
    """合并RRF信息与模型重排信息后的最终候选。"""

    chunk_id: str
    location: str
    content: str
    rrf_rank: int
    rrf_score: float
    relevance_score: int
    reason: str


def create_reranker():
    """创建要求返回RerankResponse的结构化模型。"""

    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    return llm.with_structured_output(
        RerankResponse,
        method="function_calling",
    )


def format_candidates(candidates: list[RRFResult]) -> str:
    """把候选列表转换成模型容易阅读的编号文本。"""

    sections: list[str] = []
    for rrf_rank, candidate in enumerate(candidates, start=1):
        sections.append(
            "\n".join(
                [
                    f"候选 {rrf_rank}",
                    f"chunk_id: {candidate.chunk_id}",
                    f"位置: {candidate.location}",
                    f"RRF排名: {rrf_rank}",
                    f"正文:\n{candidate.content}",
                ]
            )
        )
    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(sections)


def rerank_candidates(
    question: str,
    candidates: list[RRFResult],
) -> tuple[list[RerankedResult], RerankResponse]:
    """调用模型评分，校验chunk_id，再按相关性分数重新排序。"""

    reranker = create_reranker()
    candidate_text = format_candidates(candidates)

    response: RerankResponse = reranker.invoke(
        [
            SystemMessage(
                content=(
                    "你是RAG检索结果重排器，不负责回答用户问题。"
                    "请判断每段候选正文是否能直接帮助回答问题。"
                    "明确包含答案的资料应得到高分；只是在技术领域相近、"
                    "但没有回答问题的资料应得到低分。"
                    "必须为所有候选评分，保留原始chunk_id，不能虚构ID。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：\n{question}\n\n"
                    f"需要重排的候选资料：\n{candidate_text}"
                )
            ),
        ]
    )

    candidate_by_id = {
        candidate.chunk_id: (rrf_rank, candidate)
        for rrf_rank, candidate in enumerate(candidates, start=1)
    }

    model_result_by_id: dict[str, RerankItem] = {}
    for item in response.results:
        if item.chunk_id not in candidate_by_id:
            raise ValueError(f"模型返回了未知chunk_id：{item.chunk_id}")
        if item.chunk_id in model_result_by_id:
            raise ValueError(f"模型重复返回chunk_id：{item.chunk_id}")
        model_result_by_id[item.chunk_id] = item

    missing_ids = set(candidate_by_id) - set(model_result_by_id)
    if missing_ids:
        missing_text = ", ".join(sorted(missing_ids))
        raise ValueError(f"模型遗漏了候选：{missing_text}")

    reranked: list[RerankedResult] = []
    for chunk_id, (rrf_rank, candidate) in candidate_by_id.items():
        model_item = model_result_by_id[chunk_id]
        reranked.append(
            RerankedResult(
                chunk_id=chunk_id,
                location=candidate.location,
                content=candidate.content,
                rrf_rank=rrf_rank,
                rrf_score=candidate.rrf_score,
                relevance_score=model_item.relevance_score,
                reason=model_item.reason,
            )
        )

    reranked.sort(
        key=lambda item: (
            -item.relevance_score,
            item.rrf_rank,
        )
    )
    return reranked, response


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    vector_results = vector_search(question)
    terms, match_query, keyword_results = fts_search(question)
    rrf_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
    )[:RRF_CANDIDATE_COUNT]

    print("Day22 LLM Reranker二次重排")
    print("问题：", question)
    print("关键词：", terms)
    print("MATCH：", match_query)
    print("送入Reranker的候选数：", len(rrf_results))
    print("=" * 88)

    print("RRF初排结果：")
    for rank, item in enumerate(rrf_results, start=1):
        print(
            f"{rank}. {item.chunk_id} | "
            f"RRF={item.rrf_score:.6f} | {item.location}"
        )

    print("\n正在调用模型批量判断候选相关性……")
    reranked_results, raw_response = rerank_candidates(
        question,
        rrf_results,
    )

    print("\n" + "=" * 88)
    print("结构化模型返回类型：", type(raw_response))
    print("结构化评分数量：", len(raw_response.results))
    print("\nReranker最终排序：")

    for final_rank, item in enumerate(reranked_results, start=1):
        print("\n" + "-" * 88)
        print("最终排名：", final_rank)
        print("原RRF排名：", item.rrf_rank)
        print("chunk_id：", item.chunk_id)
        print("位置：", item.location)
        print("RRF分数：", round(item.rrf_score, 6))
        print("模型相关性分数：", item.relevance_score)
        print("评分理由：", item.reason)
        print("正文：", item.content)

    print("\n" + "=" * 88)
    print(f"最终保留前 {FINAL_RESULT_COUNT} 条：")
    for rank, item in enumerate(
        reranked_results[:FINAL_RESULT_COUNT],
        start=1,
    ):
        print(
            f"{rank}. 分数={item.relevance_score} | "
            f"{item.location} | {item.chunk_id}"
        )

    print("\n本步数据流：")
    print("问题 → 向量检索 + 关键词检索 → RRF候选集")
    print("→ 问题和候选正文批量发送给LLM Reranker")
    print("→ RerankResponse结构化评分 → 按relevance_score降序排列")
    print("说明：RRF负责广泛召回，Reranker负责更精细地判断相关性。")


if __name__ == "__main__":
    main()
