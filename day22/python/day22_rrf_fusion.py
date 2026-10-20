"""Day22 第三步：用 RRF 融合向量检索与关键词检索排名。

运行默认问题：
    python day22/python/day22_rrf_fusion.py

指定问题：
    python day22/python/day22_rrf_fusion.py "什么是 RAG？"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from day22_retrieval_compare import fts_search, vector_search


TOP_K = 5
RRF_K = 60


@dataclass
class RRFResult:
    """保存一个候选文本块在两路检索和融合后的完整信息。"""

    chunk_id: str
    location: str
    content: str
    vector_rank: int | None = None
    vector_distance: float | None = None
    keyword_rank: int | None = None
    bm25_score: float | None = None
    vector_rrf: float = 0.0
    keyword_rrf: float = 0.0
    rrf_score: float = 0.0


def rank_contribution(rank: int | None, rrf_k: int = RRF_K) -> float:
    """计算某一路排名对RRF总分的贡献；未命中时贡献为0。"""

    if rank is None:
        return 0.0
    return 1.0 / (rrf_k + rank)


def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    rrf_k: int = RRF_K,
) -> list[RRFResult]:
    """按chunk_id合并两路结果，计算RRF分数并从高到低排序。"""

    candidates: dict[str, RRFResult] = {}

    # Day21向量检索结果本身带有rank和distance。
    for item in vector_results:
        chunk_id = str(item["chunk_id"])
        candidates[chunk_id] = RRFResult(
            chunk_id=chunk_id,
            location=str(item["location"]),
            content=str(item["content"]),
            vector_rank=int(item["rank"]),
            vector_distance=float(item["distance"]),
        )

    # FTS结果按照当前列表顺序生成关键词排名。
    for keyword_rank, item in enumerate(keyword_results, start=1):
        chunk_id = str(item["chunk_id"])
        candidate = candidates.get(chunk_id)

        if candidate is None:
            candidate = RRFResult(
                chunk_id=chunk_id,
                location=str(item["location"]),
                content=str(item["content"]),
            )
            candidates[chunk_id] = candidate

        candidate.keyword_rank = keyword_rank
        candidate.bm25_score = float(item["bm25_score"])

    # 分别计算两路贡献，再求和得到统一的RRF分数。
    for candidate in candidates.values():
        candidate.vector_rrf = rank_contribution(
            candidate.vector_rank,
            rrf_k,
        )
        candidate.keyword_rrf = rank_contribution(
            candidate.keyword_rank,
            rrf_k,
        )
        candidate.rrf_score = (
            candidate.vector_rrf + candidate.keyword_rrf
        )

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.vector_rank or 999,
            candidate.keyword_rank or 999,
        ),
    )


def rank_text(rank: int | None) -> str:
    return str(rank) if rank is not None else "—"


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    vector_results = vector_search(question)[:TOP_K]
    terms, match_query, keyword_results = fts_search(question)
    keyword_results = keyword_results[:TOP_K]

    fused_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        rrf_k=RRF_K,
    )

    print("Day22 RRF混合检索")
    print("问题：", question)
    print("关键词：", terms)
    print("MATCH：", match_query)
    print("向量候选数：", len(vector_results))
    print("关键词候选数：", len(keyword_results))
    print("RRF常数 k：", RRF_K)
    print("公式：score = Σ 1 / (k + rank)")
    print("=" * 100)

    print(
        f"{'融合排名':>8} {'chunk_id':<27} "
        f"{'向量排名':>8} {'向量贡献':>10} "
        f"{'关键词排名':>10} {'关键词贡献':>11} {'RRF总分':>11}"
    )
    print("-" * 100)

    for fused_rank, item in enumerate(fused_results, start=1):
        print(
            f"{fused_rank:>8} {item.chunk_id:<27} "
            f"{rank_text(item.vector_rank):>8} "
            f"{item.vector_rrf:>10.6f} "
            f"{rank_text(item.keyword_rank):>10} "
            f"{item.keyword_rrf:>11.6f} "
            f"{item.rrf_score:>11.6f}"
        )

    print("\n逐条显示融合结果：")
    for fused_rank, item in enumerate(fused_results, start=1):
        print("\n" + "-" * 100)
        print("融合排名：", fused_rank)
        print("chunk_id：", item.chunk_id)
        print("位置：", item.location)
        print("向量排名/距离：", item.vector_rank, item.vector_distance)
        print("关键词排名/BM25：", item.keyword_rank, item.bm25_score)
        print(
            "RRF计算：",
            f"{item.vector_rrf:.6f} + "
            f"{item.keyword_rrf:.6f} = "
            f"{item.rrf_score:.6f}",
        )
        print("正文：", item.content)

    print("\n" + "=" * 100)
    print("融合前：向量排名和关键词排名各自独立。")
    print("融合后：两路共同命中的候选会累加两份排名贡献。")
    print("注意：RRF只使用rank，不直接使用distance或BM25原始分数。")
    print("下一步：在融合候选集上加入Reranker进行更精细的二次排序。")


if __name__ == "__main__":
    main()
