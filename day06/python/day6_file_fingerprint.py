import hashlib
import os
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ============================================================
# 1. 基础配置
# ============================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
knowledge_directory = day_directory / "documents"
cache_directory = project_directory / ".fastembed_cache"

# 可以通过环境变量指定测试数据库，默认使用正式练习数据库。
database_path = Path(
    os.getenv(
        "DAY6_FILE_HASH_DB",
        str(day_directory / "database" / "day6_file_fingerprint.db")
    )
)


# ============================================================
# 2. Embedding 适配器
# ============================================================

class LocalChineseEmbeddings(Embeddings):
    """让 FastEmbed 模型符合 LangChain 的 Embeddings 接口。"""

    def __init__(self, model_name: str, cache_dir: str):
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


embedding_model = LocalChineseEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir=str(cache_directory)
)


# ============================================================
# 3. 扫描文件，但暂时不创建 Document
# ============================================================

def calculate_file_hash(file_path: Path) -> str:
    """读取文件原始字节，计算 SHA-256 文件指纹。"""

    digest = hashlib.sha256()

    # 分块读取，避免大文件一次全部进入内存。
    with file_path.open("rb") as file:
        while data := file.read(64 * 1024):
            digest.update(data)

    return digest.hexdigest()


def scan_knowledge_files(
    directory: Path
) -> tuple[dict[str, Path], dict[str, str]]:
    """返回 来源名→文件路径，以及 来源名→文件指纹。"""

    file_paths: dict[str, Path] = {}

    for pattern in ("**/*.txt", "**/*.md"):
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                source = file_path.relative_to(directory).as_posix()
                file_paths[source] = file_path

    file_hashes = {
        source: calculate_file_hash(file_path)
        for source, file_path in file_paths.items()
    }

    return file_paths, file_hashes


current_file_paths, current_file_hashes = scan_knowledge_files(
    knowledge_directory
)


# ============================================================
# 4. 创建数据库、向量表和两个清单表
# ============================================================

connection = sqlite3.connect(database_path)
connection.row_factory = sqlite3.Row
connection.enable_load_extension(True)
sqlite_vec.load(connection)
connection.enable_load_extension(False)

vector_store = SQLiteVec(
    table="knowledge_chunks",
    connection=connection,
    embedding=embedding_model
)

# file_manifest：记录每个文件上一次建立索引时的指纹。
connection.execute("""
    CREATE TABLE IF NOT EXISTS file_manifest (
        source TEXT PRIMARY KEY,
        file_hash TEXT NOT NULL,
        indexed_at TEXT NOT NULL
    )
""")

# chunk_manifest：记录业务 chunk_id 与 SQLite rowid 的对应关系。
connection.execute("""
    CREATE TABLE IF NOT EXISTS chunk_manifest (
        chunk_id TEXT PRIMARY KEY,
        vector_rowid INTEGER NOT NULL UNIQUE,
        source TEXT NOT NULL
    )
""")

connection.commit()


# ============================================================
# 5. 比较当前文件指纹和数据库旧指纹
# ============================================================

stored_file_rows = connection.execute("""
    SELECT source, file_hash
    FROM file_manifest
""").fetchall()

stored_file_hashes = {
    row["source"]: row["file_hash"]
    for row in stored_file_rows
}

current_sources = set(current_file_hashes)
stored_sources = set(stored_file_hashes)

new_sources = current_sources - stored_sources
deleted_sources = stored_sources - current_sources

changed_sources = {
    source
    for source in current_sources & stored_sources
    if current_file_hashes[source] != stored_file_hashes[source]
}

unchanged_sources = (
    current_sources & stored_sources
) - changed_sources


print("知识库目录：", knowledge_directory)
print("数据库：", database_path)

print("\n扫描到的文件数量：", len(current_sources))
print("新增文件：", len(new_sources), sorted(new_sources))
print("修改文件：", len(changed_sources), sorted(changed_sources))
print("未变化文件：", len(unchanged_sources), sorted(unchanged_sources))
print("已删除文件：", len(deleted_sources), sorted(deleted_sources))


# ============================================================
# 6. 删除“已修改文件”和“已删除文件”的旧 chunks
# ============================================================

sources_to_clear = changed_sources | deleted_sources

for source in sorted(sources_to_clear):
    old_chunk_rows = connection.execute("""
        SELECT chunk_id, vector_rowid
        FROM chunk_manifest
        WHERE source = ?
    """, (source,)).fetchall()

    for row in old_chunk_rows:
        vector_rowid = row["vector_rowid"]

        connection.execute(
            "DELETE FROM knowledge_chunks_vec WHERE rowid = ?",
            (vector_rowid,)
        )
        connection.execute(
            "DELETE FROM knowledge_chunks WHERE rowid = ?",
            (vector_rowid,)
        )
        connection.execute(
            "DELETE FROM chunk_manifest WHERE chunk_id = ?",
            (row["chunk_id"],)
        )

        print(f"[删除旧 chunk] {source} | {row['chunk_id']}")

    connection.execute(
        "DELETE FROM file_manifest WHERE source = ?",
        (source,)
    )

connection.commit()


# ============================================================
# 7. 只加载新增或修改过的文件
# ============================================================

sources_to_index = new_sources | changed_sources
documents_to_split: list[Document] = []

for source in sorted(sources_to_index):
    file_path = current_file_paths[source]

    loaded_documents = TextLoader(
        str(file_path),
        encoding="utf-8"
    ).load()

    document = loaded_documents[0]
    document.metadata = {
        "source": source,
        "file_name": file_path.name,
        "title": file_path.stem
    }

    documents_to_split.append(document)
    print("[加载文件]", source)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    add_start_index=True,
    separators=["\n\n", "\n", "。", "；", "，", ""]
)

new_chunks = text_splitter.split_documents(documents_to_split)


def create_chunk_id(chunk: Document) -> str:
    identity_text = (
        f"source={chunk.metadata['source']}\n"
        f"start_index={chunk.metadata['start_index']}\n"
        f"content={chunk.page_content}"
    )

    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


for chunk in new_chunks:
    chunk.id = create_chunk_id(chunk)
    chunk.metadata["chunk_id"] = chunk.id


# ============================================================
# 8. 为新 chunks 生成向量并更新清单
# ============================================================

if new_chunks:
    vector_rowids = vector_store.add_documents(new_chunks)

    for chunk, vector_rowid in zip(new_chunks, vector_rowids):
        connection.execute("""
            INSERT INTO chunk_manifest (
                chunk_id, vector_rowid, source
            ) VALUES (?, ?, ?)
        """, (
            chunk.id,
            int(vector_rowid),
            chunk.metadata["source"]
        ))

        print(
            f"[写入新 chunk] {chunk.metadata['source']} | {chunk.id}"
        )
else:
    print("\n没有需要加载、切分和向量化的文件。")


indexed_at = datetime.now(timezone.utc).isoformat()

for source in sorted(sources_to_index):
    connection.execute("""
        INSERT INTO file_manifest (source, file_hash, indexed_at)
        VALUES (?, ?, ?)
        ON CONFLICT(source) DO UPDATE SET
            file_hash = excluded.file_hash,
            indexed_at = excluded.indexed_at
    """, (
        source,
        current_file_hashes[source],
        indexed_at
    ))

connection.commit()


# ============================================================
# 9. 检查结果并执行一次检索
# ============================================================

file_count = connection.execute(
    "SELECT COUNT(*) FROM file_manifest"
).fetchone()[0]

chunk_count = connection.execute(
    "SELECT COUNT(*) FROM chunk_manifest"
).fetchone()[0]

main_count = connection.execute(
    "SELECT COUNT(*) FROM knowledge_chunks"
).fetchone()[0]

vector_count = connection.execute(
    "SELECT COUNT(*) FROM knowledge_chunks_vec"
).fetchone()[0]

print("\n" + "=" * 60)
print("同步结果：")
print("文件清单数量：", file_count)
print("chunk 清单数量：", chunk_count)
print("正文表数量：", main_count)
print("向量表数量：", vector_count)
print("chunk 三处数量一致：", chunk_count == main_count == vector_count)

question = input("\n请输入检索问题：").strip()

if question:
    results = vector_store.similarity_search_with_score(question, k=3)

    print("\n检索结果（距离越小越相关）：")

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\n排名：{rank}")
        print("来源：", document.metadata["source"])
        print("距离：", round(distance, 4))
        print("正文：", document.page_content)
else:
    print("\n没有输入问题，本次只同步文件索引。")

connection.close()

print("\n" + "=" * 60)
print("文件级增量索引完成。")
print("未变化文件：只计算文件指纹，然后跳过加载、切分和 Embedding。")
print("新增文件：加载、切分并建立向量。")
print("修改文件：删除该文件的旧 chunks，再建立新 chunks。")
print("删除文件：删除该文件对应的所有旧 chunks。")
