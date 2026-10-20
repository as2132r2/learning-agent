import hashlib
import json
import os
import sqlite3
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ============================================================
# 1. 配置
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DAY_DIRECTORY = SCRIPT_DIRECTORY.parent
PROJECT_DIRECTORY = DAY_DIRECTORY.parent
KNOWLEDGE_DIRECTORY = DAY_DIRECTORY / "documents"
CACHE_DIRECTORY = PROJECT_DIRECTORY / ".fastembed_cache"
DATABASE_PATH = DAY_DIRECTORY / "database" / "day6_knowledge_cli.db"

VECTOR_TABLE = "knowledge_chunks"
FILE_MANIFEST = "file_manifest"
CHUNK_MANIFEST = "chunk_manifest"
RELEVANCE_THRESHOLD = 0.95

load_dotenv()


# ============================================================
# 2. Embedding 模型
# ============================================================

class LocalChineseEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str):
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


def create_embedding_model() -> LocalChineseEmbeddings:
    """仅在 sync 或 search 时创建模型，status 不需要模型。"""

    return LocalChineseEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_dir=str(CACHE_DIRECTORY)
    )


# ============================================================
# 3. 数据库连接
# ============================================================

def open_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {FILE_MANIFEST} (
            source TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        )
    """)

    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {CHUNK_MANIFEST} (
            chunk_id TEXT PRIMARY KEY,
            vector_rowid INTEGER NOT NULL UNIQUE,
            source TEXT NOT NULL
        )
    """)

    connection.commit()


def create_vector_store(
    connection: sqlite3.Connection,
    embedding_model: Embeddings
) -> SQLiteVec:
    return SQLiteVec(
        table=VECTOR_TABLE,
        connection=connection,
        embedding=embedding_model
    )


# ============================================================
# 4. 文件扫描与指纹
# ============================================================

def calculate_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while data := file.read(64 * 1024):
            digest.update(data)

    return digest.hexdigest()


def scan_files() -> tuple[dict[str, Path], dict[str, str]]:
    file_paths: dict[str, Path] = {}

    for pattern in ("**/*.txt", "**/*.md", "**/*.pdf"):
        for file_path in KNOWLEDGE_DIRECTORY.glob(pattern):
            if file_path.is_file():
                source = file_path.relative_to(
                    KNOWLEDGE_DIRECTORY
                ).as_posix()
                file_paths[source] = file_path

    file_hashes = {
        source: calculate_file_hash(file_path)
        for source, file_path in file_paths.items()
    }

    return file_paths, file_hashes


# ============================================================
# 5. 文档加载、切分与稳定 ID
# ============================================================

def create_chunk_id(chunk: Document) -> str:
    identity_text = (
        f"source={chunk.metadata['source']}\n"
        f"start_index={chunk.metadata['start_index']}\n"
        f"content={chunk.page_content}"
    )

    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


def load_and_split_files(
    sources: set[str],
    file_paths: dict[str, Path]
) -> list[Document]:
    documents: list[Document] = []

    for source in sorted(sources):
        file_path = file_paths[source]
        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            continue

        loaded_documents = loader.load()

        # TXT、MD 通常只有一个 Document；
        # PDF 通常每页对应一个 Document。
        for document_number, document in enumerate(
            loaded_documents,
            start=1
        ):
            original_page = document.metadata.get("page")
            page_number = (
                original_page + 1
                if original_page is not None
                else None
            )

            document.metadata = {
                "source": source,
                "file_name": file_path.name,
                "file_type": suffix,
                "title": file_path.stem,
                "page": page_number,
                "document_number": document_number
            }

            documents.append(document)

        print(
            f"[加载] {source} → "
            f"{len(loaded_documents)} 个 Document"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "；", "，", ""]
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.id = create_chunk_id(chunk)
        chunk.metadata["chunk_id"] = chunk.id

    return chunks


# ============================================================
# 6. sync：同步文件和索引
# ============================================================

def sync_command() -> None:
    file_paths, current_hashes = scan_files()
    current_sources = set(current_hashes)

    embedding_model = create_embedding_model()
    connection = open_connection()
    vector_store = create_vector_store(connection, embedding_model)
    create_tables(connection)

    rows = connection.execute(f"""
        SELECT source, file_hash
        FROM {FILE_MANIFEST}
    """).fetchall()

    stored_hashes = {
        row["source"]: row["file_hash"]
        for row in rows
    }
    stored_sources = set(stored_hashes)

    new_sources = current_sources - stored_sources
    deleted_sources = stored_sources - current_sources
    changed_sources = {
        source
        for source in current_sources & stored_sources
        if current_hashes[source] != stored_hashes[source]
    }
    unchanged_sources = (
        current_sources & stored_sources
    ) - changed_sources

    print("新增文件：", sorted(new_sources))
    print("修改文件：", sorted(changed_sources))
    print("未变化文件：", sorted(unchanged_sources))
    print("删除文件：", sorted(deleted_sources))

    # 修改文件需要先删除旧 chunks；删除文件也要清理旧 chunks。
    for source in sorted(changed_sources | deleted_sources):
        chunk_rows = connection.execute(f"""
            SELECT chunk_id, vector_rowid
            FROM {CHUNK_MANIFEST}
            WHERE source = ?
        """, (source,)).fetchall()

        for row in chunk_rows:
            vector_rowid = row["vector_rowid"]
            connection.execute(
                f"DELETE FROM {VECTOR_TABLE}_vec WHERE rowid = ?",
                (vector_rowid,)
            )
            connection.execute(
                f"DELETE FROM {VECTOR_TABLE} WHERE rowid = ?",
                (vector_rowid,)
            )
            connection.execute(
                f"DELETE FROM {CHUNK_MANIFEST} WHERE chunk_id = ?",
                (row["chunk_id"],)
            )
            print("[删除旧 chunk]", row["chunk_id"])

        connection.execute(
            f"DELETE FROM {FILE_MANIFEST} WHERE source = ?",
            (source,)
        )

    connection.commit()

    sources_to_index = new_sources | changed_sources
    chunks = load_and_split_files(sources_to_index, file_paths)

    if chunks:
        vector_rowids = vector_store.add_documents(chunks)

        for chunk, vector_rowid in zip(chunks, vector_rowids):
            connection.execute(f"""
                INSERT INTO {CHUNK_MANIFEST}
                    (chunk_id, vector_rowid, source)
                VALUES (?, ?, ?)
            """, (
                chunk.id,
                int(vector_rowid),
                chunk.metadata["source"]
            ))
            print("[写入新 chunk]", chunk.id)
    else:
        print("没有文件需要重新加载和生成向量。")

    indexed_at = datetime.now(timezone.utc).isoformat()

    for source in sorted(sources_to_index):
        connection.execute(f"""
            INSERT INTO {FILE_MANIFEST}
                (source, file_hash, indexed_at)
            VALUES (?, ?, ?)
        """, (
            source,
            current_hashes[source],
            indexed_at
        ))

    connection.commit()
    connection.close()
    print("同步完成。")


# ============================================================
# 7. search：只查询已有数据库
# ============================================================

def search_command(question: str) -> None:
    if not DATABASE_PATH.exists():
        print("数据库还不存在，请先运行：")
        print("python day06/python/day6_knowledge_cli.py sync")
        return

    embedding_model = create_embedding_model()
    connection = open_connection()
    vector_store = create_vector_store(connection, embedding_model)

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    print("问题：", question)
    print("\n检索结果（距离越小越相关）：")

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\n排名：{rank}")
        print("来源：", document.metadata["source"])
        print("距离：", round(distance, 4))
        print("正文：", document.page_content)

    connection.close()


# ============================================================
# 8. ask：守门、扩展上下文并调用聊天模型
# ============================================================

def format_location(document: Document) -> str:
    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")

    if page is None:
        return source

    return f"{source}，第 {page} 页"


def ask_command(question: str) -> None:
    if not DATABASE_PATH.exists():
        print("数据库还不存在，请先运行：")
        print("python day06/python/day6_knowledge_cli.py sync")
        return

    embedding_model = create_embedding_model()
    connection = open_connection()
    vector_store = create_vector_store(connection, embedding_model)

    hits = vector_store.similarity_search_with_score(question, k=1)

    if not hits:
        print("知识库中没有可以检索的内容。")
        connection.close()
        return

    best_document, best_distance = hits[0]

    print("问题：", question)
    print("最佳命中：", format_location(best_document))
    print("距离：", round(best_distance, 4))
    print("守门阈值：", RELEVANCE_THRESHOLD)

    # SQLiteVec 返回的是距离，所以越小越相关。
    if best_distance > RELEVANCE_THRESHOLD:
        print("\n根据当前知识库资料无法回答这个问题。")
        print("相关性守门未通过，没有调用聊天模型。")
        connection.close()
        return

    # 守门通过后，读取已有正文，不重新生成向量。
    rows = connection.execute(f"""
        SELECT text, metadata
        FROM {VECTOR_TABLE}
    """).fetchall()

    chunks_by_location: dict[
        tuple[str, int | None],
        list[Document]
    ] = defaultdict(list)

    for row in rows:
        metadata = json.loads(row["metadata"]) or {}
        document = Document(
            page_content=row["text"],
            metadata=metadata,
            id=metadata.get("chunk_id")
        )

        key = (
            metadata.get("source", "未知来源"),
            metadata.get("page")
        )
        chunks_by_location[key].append(document)

    for chunks in chunks_by_location.values():
        chunks.sort(
            key=lambda chunk: chunk.metadata.get("start_index", 0)
        )

    hit_key = (
        best_document.metadata.get("source", "未知来源"),
        best_document.metadata.get("page")
    )
    location_chunks = chunks_by_location[hit_key]
    hit_chunk_id = best_document.metadata.get("chunk_id")

    hit_index = next(
        index
        for index, chunk in enumerate(location_chunks)
        if chunk.metadata.get("chunk_id") == hit_chunk_id
    )

    # window_size=1：最多取前一个、命中块、后一个。
    start = max(0, hit_index - 1)
    end = min(len(location_chunks), hit_index + 2)
    expanded_chunks = location_chunks[start:end]

    context_parts = [
        f"[资料 {index} | {format_location(document)}]\n"
        f"{document.page_content}"
        for index, document in enumerate(expanded_chunks, start=1)
    ]
    context = "\n\n".join(context_parts)

    print("\n扩展后的参考资料：")
    print(context)

    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是一名严谨、简洁的编程学习助手。"
                "只能根据参考资料回答问题。"
                "关键结论后必须标注资料编号，例如：[资料 1]。"
                "不要编造资料、文件名或页码。"
            )
        ),
        HumanMessage(
            content=(
                f"参考资料：\n{context}\n\n"
                f"用户问题：{question}"
            )
        )
    ])

    print("\n助手回答：")
    print(response.content)

    print("\n程序核验的参考位置：")
    for location in dict.fromkeys(
        format_location(document)
        for document in expanded_chunks
    ):
        print("-", location)

    connection.close()


# ============================================================
# 9. status：只查看数据库，不加载 Embedding 模型
# ============================================================

def status_command() -> None:
    if not DATABASE_PATH.exists():
        print("数据库还不存在，请先执行 sync。")
        return

    connection = open_connection()

    required_tables = {FILE_MANIFEST, CHUNK_MANIFEST, VECTOR_TABLE}
    table_rows = connection.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
    """).fetchall()
    existing_tables = {row["name"] for row in table_rows}

    if not required_tables.issubset(existing_tables):
        print("数据库结构不完整，请执行 sync。")
        connection.close()
        return

    files = connection.execute(f"""
        SELECT source, indexed_at
        FROM {FILE_MANIFEST}
        ORDER BY source
    """).fetchall()

    chunk_count = connection.execute(
        f"SELECT COUNT(*) FROM {CHUNK_MANIFEST}"
    ).fetchone()[0]

    print("数据库：", DATABASE_PATH)
    print("文件数量：", len(files))
    print("chunk 数量：", chunk_count)
    print("\n已建立索引的文件：")

    for row in files:
        print(f"- {row['source']} | {row['indexed_at']}")

    connection.close()


# ============================================================
# 10. 命令行入口
# ============================================================

def print_help() -> None:
    print("使用方法：")
    print("  python day06/python/day6_knowledge_cli.py sync")
    print('  python day06/python/day6_knowledge_cli.py search "什么是 RAG？"')
    print('  python day06/python/day6_knowledge_cli.py ask "为什么要保留来源信息？"')
    print("  python day06/python/day6_knowledge_cli.py status")


if len(sys.argv) < 2:
    print_help()
    raise SystemExit(1)

command = sys.argv[1].lower()

if command == "sync":
    sync_command()
elif command == "search":
    if len(sys.argv) < 3:
        print("search 后面必须提供问题。")
        print_help()
        raise SystemExit(1)

    search_command(" ".join(sys.argv[2:]))
elif command == "ask":
    if len(sys.argv) < 3:
        print("ask 后面必须提供问题。")
        print_help()
        raise SystemExit(1)

    ask_command(" ".join(sys.argv[2:]))
elif command == "status":
    status_command()
else:
    print("未知命令：", command)
    print_help()
    raise SystemExit(1)
