"""Day22 第二步：并排比较向量检索与FTS5/BM25关键词检索。

运行：
    python day22/python/day22_retrieval_compare.py "什么是 RAG？"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY21_PYTHON = PROJECT_DIR / "day21" / "python"
if str(DAY21_PYTHON) not in sys.path:
    sys.path.insert(0, str(DAY21_PYTHON))

from day21_browser_retrieval_server import search_knowledge  # noqa: E402
from day22_fts_basic import (  # noqa: E402
    DATABASE_PATH as FTS_DATABASE_PATH,
    keyword_search,
    open_database as open_fts_database,
    rebuild_index,
)


TOP_K = 5


@dataclass
class ComparedChunk:
    chunk_id: str
    location: str
    content: str
    vector_rank: int | None = None
    vector_distance: float | None = None
    keyword_rank: int | None = None
    bm25_score: float | None = None


def vector_search(question: str) -> list[dict]:
    """复用Day21向量检索，并只取前TOP_K条。"""

    response = search_knowledge(question)
    return response["results"][:TOP_K]


def fts_search(question: str) -> tuple[list[str], str, list[dict]]:
    """查询FTS数据库；缺少索引时自动重建。"""

    connection = open_fts_database()
    try:
        table_exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'chunks_fts'
            """
        ).fetchone()
        if table_exists is None:
            rebuild_index(connection)

        terms, match_query, rows = keyword_search(
            connection,
            question,
            limit=TOP_K,
        )
        return terms, match_query, [dict(row) for row in rows]
    finally:
        connection.close()


def compare_results(
    vector_results: list[dict],
    keyword_results: list[dict],
) -> list[ComparedChunk]:
    """按chunk_id合并两套排名，但不直接混合两种分数。"""

    compared: dict[str, ComparedChunk] = {}

    for item in vector_results:
        chunk_id = str(item["chunk_id"])
        compared[chunk_id] = ComparedChunk(
            chunk_id=chunk_id,
            location=str(item["location"]),
            content=str(item["content"]),
            vector_rank=int(item["rank"]),
            vector_distance=float(item["distance"]),
        )

    for rank, item in enumerate(keyword_results, start=1):
        chunk_id = str(item["chunk_id"])
        existing = compared.get(chunk_id)
        if existing is None:
            existing = ComparedChunk(
                chunk_id=chunk_id,
                location=str(item["location"]),
                content=str(item["content"]),
            )
            compared[chunk_id] = existing

        existing.keyword_rank = rank
        existing.bm25_score = float(item["bm25_score"])

    return sorted(
        compared.values(),
        key=lambda item: (
            item.vector_rank if item.vector_rank is not None else 999,
            item.keyword_rank if item.keyword_rank is not None else 999,
        ),
    )


def rank_text(rank: int | None) -> str:
    return str(rank) if rank is not None else "—"


def number_text(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    vector_results = vector_search(question)
    terms, match_query, keyword_results = fts_search(question)
    compared = compare_results(vector_results, keyword_results)

    vector_ids = {str(item["chunk_id"]) for item in vector_results}
    keyword_ids = {str(item["chunk_id"]) for item in keyword_results}
    overlap_ids = vector_ids & keyword_ids

    print("Day22 向量检索 vs 关键词检索")
    print("问题：", question)
    print("FTS数据库：", FTS_DATABASE_PATH)
    print("关键词：", terms)
    print("MATCH：", match_query)
    print("=" * 92)

    print(
        f"{'chunk_id':<27} {'向量排名':>8} {'距离':>9} "
        f"{'关键词排名':>10} {'BM25':>11}"
    )
    print("-" * 92)

    for item in compared:
        print(
            f"{item.chunk_id:<27} "
            f"{rank_text(item.vector_rank):>8} "
            f"{number_text(item.vector_distance):>9} "
            f"{rank_text(item.keyword_rank):>10} "
            f"{number_text(item.bm25_score):>11}"
        )

    print("\n" + "=" * 92)
    print("向量检索结果数：", len(vector_ids))
    print("关键词检索结果数：", len(keyword_ids))
    print("两路共同命中数：", len(overlap_ids))
    print("仅向量命中数：", len(vector_ids - keyword_ids))
    print("仅关键词命中数：", len(keyword_ids - vector_ids))

    print("\n逐块对比：")
    for item in compared:
        if item.vector_rank is not None and item.keyword_rank is not None:
            origin = "两路共同命中"
        elif item.vector_rank is not None:
            origin = "仅向量检索命中"
        else:
            origin = "仅关键词检索命中"

        print("\n" + "-" * 92)
        print("候选类型：", origin)
        print("chunk_id：", item.chunk_id)
        print("位置：", item.location)
        print("向量排名/距离：", item.vector_rank, item.vector_distance)
        print("关键词排名/BM25：", item.keyword_rank, item.bm25_score)
        print("正文：", item.content)

    print("\n" + "=" * 92)
    print("结论：距离和BM25量纲不同，不能直接相加。")
    print("下一步使用两路排名而不是原始分数进行RRF融合。")


if __name__ == "__main__":
    main()
