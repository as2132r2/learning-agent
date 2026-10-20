"""Day21 第四步：真实中文Embedding与增量持久化向量索引。

命令：
    python day21/python/day21_vector_index.py sync
    python day21/python/day21_vector_index.py status
    python day21/python/day21_vector_index.py search "什么是RAG？"
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from langchain_community.vectorstores.sqlitevec import SQLiteVec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from day21_document_pipeline import load_documents, split_documents


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY21_DIR = PROJECT_DIR / "day21"
DATABASE_PATH = DAY21_DIR / "database" / "day21_vector.db"
CACHE_DIR = PROJECT_DIR / ".fastembed_cache"
SOURCE_DIRS = [
    DAY21_DIR / "documents",
    DAY21_DIR / "uploads_validated",
]
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

VECTOR_TABLE = "knowledge_chunks"
FILE_MANIFEST = "knowledge_files"
CHUNK_MANIFEST = "knowledge_chunk_manifest"


class LocalChineseEmbeddings(Embeddings):
    """将FastEmbed本地中文模型适配为LangChain Embeddings。"""

    def __init__(self) -> None:
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-zh-v1.5",
            cache_dir=str(CACHE_DIR),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


def open_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def create_manifest_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {FILE_MANIFEST} (
            source TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {CHUNK_MANIFEST} (
            chunk_id TEXT PRIMARY KEY,
            vector_rowid INTEGER NOT NULL UNIQUE,
            source TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_day21_chunk_source
        ON {CHUNK_MANIFEST}(source);
        """
    )
    connection.commit()


def create_vector_store(
    connection: sqlite3.Connection,
    embedding: Embeddings,
) -> SQLiteVec:
    return SQLiteVec(
        table=VECTOR_TABLE,
        connection=connection,
        embedding=embedding,
    )


def source_name(file_path: Path) -> str:
    return str(file_path.resolve().relative_to(PROJECT_DIR))


def file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as file:
        while chunk := file.read(64 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_files() -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}

    for directory in SOURCE_DIRS:
        if not directory.exists():
            continue
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
                source = source_name(file_path)
                paths[source] = file_path
                hashes[source] = file_sha256(file_path)

    return paths, hashes


def remove_source(
    connection: sqlite3.Connection,
    source: str,
) -> int:
    """删除一个来源对应的向量、正文和清单记录。"""

    rows = connection.execute(
        f"SELECT chunk_id, vector_rowid FROM {CHUNK_MANIFEST} WHERE source = ?",
        (source,),
    ).fetchall()

    for row in rows:
        vector_rowid = row["vector_rowid"]
        connection.execute(
            f"DELETE FROM {VECTOR_TABLE}_vec WHERE rowid = ?",
            (vector_rowid,),
        )
        connection.execute(
            f"DELETE FROM {VECTOR_TABLE} WHERE rowid = ?",
            (vector_rowid,),
        )

    connection.execute(
        f"DELETE FROM {CHUNK_MANIFEST} WHERE source = ?",
        (source,),
    )
    connection.execute(
        f"DELETE FROM {FILE_MANIFEST} WHERE source = ?",
        (source,),
    )
    return len(rows)


def sync_command() -> None:
    file_paths, current_hashes = scan_files()
    current_sources = set(current_hashes)

    embedding = LocalChineseEmbeddings()
    connection = open_connection()
    vector_store = create_vector_store(connection, embedding)
    create_manifest_tables(connection)

    stored_rows = connection.execute(
        f"SELECT source, file_hash FROM {FILE_MANIFEST}"
    ).fetchall()
    stored_hashes = {row["source"]: row["file_hash"] for row in stored_rows}
    stored_sources = set(stored_hashes)

    new_sources = current_sources - stored_sources
    deleted_sources = stored_sources - current_sources
    changed_sources = {
        source
        for source in current_sources & stored_sources
        if current_hashes[source] != stored_hashes[source]
    }
    unchanged_sources = current_sources - new_sources - changed_sources

    print("新增文件：", sorted(new_sources))
    print("修改文件：", sorted(changed_sources))
    print("未变化文件：", sorted(unchanged_sources))
    print("删除文件：", sorted(deleted_sources))

    for source in sorted(changed_sources | deleted_sources):
        removed = remove_source(connection, source)
        print(f"[清理旧索引] {source}，文本块={removed}")
    connection.commit()

    sources_to_index = sorted(new_sources | changed_sources)
    chunks: list[Document] = []
    if sources_to_index:
        documents = load_documents([file_paths[source] for source in sources_to_index])
        chunks = split_documents(documents)
        for chunk in chunks:
            chunk.metadata["chunk_id"] = chunk.id

    if chunks:
        vector_rowids = vector_store.add_documents(chunks)
        for chunk, vector_rowid in zip(chunks, vector_rowids):
            source = str(chunk.metadata["source"])
            connection.execute(
                f"""INSERT INTO {CHUNK_MANIFEST}
                    (chunk_id, vector_rowid, source) VALUES (?, ?, ?)""",
                (chunk.id, int(vector_rowid), source),
            )
            print(f"[写入向量] {chunk.id} | {source}")
    else:
        print("没有文本块需要生成Embedding。")

    indexed_at = datetime.now(timezone.utc).isoformat()
    for source in sources_to_index:
        connection.execute(
            f"""INSERT INTO {FILE_MANIFEST}
                (source, file_hash, indexed_at) VALUES (?, ?, ?)""",
            (source, current_hashes[source], indexed_at),
        )

    connection.commit()
    connection.close()
    print("同步完成。")


def status_command() -> None:
    file_paths, current_hashes = scan_files()
    print("扫描目录：")
    for directory in SOURCE_DIRS:
        print("-", directory)
    print("当前磁盘文档：", len(file_paths))

    if not DATABASE_PATH.exists():
        print("向量数据库尚未创建，请先运行 sync。")
        return

    connection = open_connection()
    create_manifest_tables(connection)
    indexed_files = connection.execute(
        f"SELECT COUNT(*) FROM {FILE_MANIFEST}"
    ).fetchone()[0]
    indexed_chunks = connection.execute(
        f"SELECT COUNT(*) FROM {CHUNK_MANIFEST}"
    ).fetchone()[0]
    connection.close()

    print("数据库已索引文档：", indexed_files)
    print("数据库文本块：", indexed_chunks)
    print("数据库路径：", DATABASE_PATH)
    print("当前哈希数量：", len(current_hashes))


def search_command(question: str) -> None:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError("向量数据库不存在，请先运行 sync。")

    embedding = LocalChineseEmbeddings()
    connection = open_connection()
    vector_store = create_vector_store(connection, embedding)
    results = vector_store.similarity_search_with_score(question, k=3)

    print("问题：", question)
    print("检索结果（距离越小越相关）：")
    for rank, (document, distance) in enumerate(results, start=1):
        source = document.metadata.get("source")
        page = document.metadata.get("page")
        location = f"{source}，第 {page} 页" if page is not None else source
        print("\n" + "-" * 72)
        print("排名：", rank)
        print("位置：", location)
        print("chunk_id：", document.metadata.get("chunk_id", document.id))
        print("距离：", round(float(distance), 4))
        print("正文：", document.page_content)

    connection.close()


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"

    if command == "sync":
        sync_command()
    elif command == "status":
        status_command()
    elif command == "search":
        if len(sys.argv) < 3:
            raise SystemExit("用法：python day21_vector_index.py search 问题")
        search_command(" ".join(sys.argv[2:]))
    else:
        raise SystemExit("命令只能是 sync、status 或 search。")


if __name__ == "__main__":
    main()
