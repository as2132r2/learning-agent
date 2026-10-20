"""Day22 第一步：使用 SQLite FTS5 + BM25 做关键词全文检索。

运行默认问题：
    python day22/python/day22_fts_basic.py

指定问题：
    python day22/python/day22_fts_basic.py "什么是 RAG？"
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

from langchain_core.documents import Document


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY21_DIR = PROJECT_DIR / "day21"
DAY22_DIR = PROJECT_DIR / "day22"
DATABASE_PATH = DAY22_DIR / "database" / "day22_fts.db"
SOURCE_DIRS = [
    DAY21_DIR / "documents",
    DAY21_DIR / "uploads_validated",
]
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

# day21/python 不在当前脚本的默认导入目录中，显式加入模块搜索路径。
DAY21_PYTHON = DAY21_DIR / "python"
if str(DAY21_PYTHON) not in sys.path:
    sys.path.insert(0, str(DAY21_PYTHON))

from day21_document_pipeline import load_documents, split_documents  # noqa: E402


def scan_source_files() -> list[Path]:
    """扫描Day21内置资料和通过校验的上传资料。"""

    files: list[Path] = []
    for directory in SOURCE_DIRS:
        if not directory.exists():
            continue
        files.extend(
            file_path
            for file_path in directory.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    return sorted(files)


def format_location(document: Document) -> str:
    source = str(document.metadata.get("source", "未知来源"))
    page = document.metadata.get("page")
    return f"{source}，第 {page} 页" if page is not None else source


def open_database() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def rebuild_index(connection: sqlite3.Connection) -> int:
    """重新加载当前资料，并建立trigram全文索引。"""

    file_paths = scan_source_files()
    documents = load_documents(file_paths)
    chunks = split_documents(documents)

    connection.execute("DROP TABLE IF EXISTS chunks_fts")
    connection.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            content,
            source UNINDEXED,
            location UNINDEXED,
            tokenize = 'trigram'
        )
        """
    )

    rows = [
        (
            chunk.id,
            chunk.page_content,
            str(chunk.metadata.get("source", "")),
            format_location(chunk),
        )
        for chunk in chunks
    ]

    connection.executemany(
        """
        INSERT INTO chunks_fts (
            chunk_id, content, source, location
        )
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()
    return len(rows)


def extract_search_terms(question: str) -> list[str]:
    """从自然语言问题中提取适合FTS MATCH的中英文片段。"""

    raw_terms = re.findall(
        r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_.+\-]{2,}",
        question,
    )

    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = term.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


def build_match_query(terms: list[str]) -> str:
    """用OR连接安全引用后的检索词。"""

    if not terms:
        raise ValueError("问题中没有可用于全文检索的关键词。")

    quoted_terms = [
        '"' + term.replace('"', '""') + '"'
        for term in terms
    ]
    return " OR ".join(quoted_terms)


def keyword_search(
    connection: sqlite3.Connection,
    question: str,
    limit: int = 5,
) -> tuple[list[str], str, list[sqlite3.Row]]:
    """执行FTS5检索；BM25分数越小代表排名越靠前。"""

    terms = extract_search_terms(question)
    match_query = build_match_query(terms)

    rows = connection.execute(
        """
        SELECT
            chunk_id,
            content,
            source,
            location,
            bm25(chunks_fts) AS bm25_score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_score ASC
        LIMIT ?
        """,
        (match_query, limit),
    ).fetchall()

    return terms, match_query, rows


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    print("Day22 SQLite FTS5/BM25关键词检索")
    print("数据库：", DATABASE_PATH)
    print("扫描目录：")
    for directory in SOURCE_DIRS:
        print("-", directory)

    connection = open_database()
    try:
        chunk_count = rebuild_index(connection)
        terms, match_query, rows = keyword_search(
            connection,
            question,
            limit=5,
        )

        print("\n" + "=" * 76)
        print("建立FTS索引的文本块数量：", chunk_count)
        print("用户问题：", question)
        print("提取关键词：", terms)
        print("MATCH表达式：", match_query)
        print("检索结果数量：", len(rows))

        for rank, row in enumerate(rows, start=1):
            print("\n" + "-" * 76)
            print("排名：", rank)
            print("chunk_id：", row["chunk_id"])
            print("位置：", row["location"])
            print("BM25分数：", round(float(row["bm25_score"]), 6))
            print("正文：", row["content"])

        print("\n" + "=" * 76)
        print("数据流：")
        print("Document文本块 → FTS5 trigram倒排索引")
        print("自然语言问题 → 关键词提取 → MATCH表达式")
        print("→ BM25排序 → list[sqlite3.Row]")
        print("说明：BM25分数只用于本次关键词排名，不等于向量距离。")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
