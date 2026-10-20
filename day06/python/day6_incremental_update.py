import hashlib
import shutil
import sqlite3
import tempfile
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

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ============================================================
# 1. Embedding 适配器
# ============================================================

class LocalChineseEmbeddings(Embeddings):
    """把 FastEmbed 模型包装成 LangChain 需要的 Embeddings。"""

    def __init__(self, model_name: str, cache_dir: str):
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


# ============================================================
# 2. 加载、切分文件，并生成稳定 chunk_id
# ============================================================

def create_chunk_id(chunk: Document) -> str:
    """来源、位置或正文改变时，chunk_id 就会改变。"""

    identity_text = (
        f"source={chunk.metadata['source']}\n"
        f"start_index={chunk.metadata['start_index']}\n"
        f"content={chunk.page_content}"
    )

    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


def load_chunks(knowledge_directory: Path) -> list[Document]:
    """读取指定目录中的 txt、md 文件，并返回切分后的 Document。"""

    loader = DirectoryLoader(
        path=str(knowledge_directory),
        glob=["**/*.txt", "**/*.md"],
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
        use_multithreading=False
    )

    documents = loader.load()
    documents.sort(key=lambda document: document.metadata["source"])

    for document in documents:
        absolute_source = Path(document.metadata["source"]).resolve()
        relative_source = absolute_source.relative_to(
            knowledge_directory.resolve()
        )

        document.metadata = {
            "source": relative_source.as_posix(),
            "file_name": absolute_source.name,
            "title": absolute_source.stem
        }

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
# 3. 打开数据库并创建索引清单
# ============================================================

def open_database(
    database_path: Path,
    embedding_model: Embeddings
) -> tuple[sqlite3.Connection, SQLiteVec]:
    """返回 SQLite 连接和 LangChain SQLiteVec 对象。"""

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

    connection.execute("""
        CREATE TABLE IF NOT EXISTS index_manifest (
            chunk_id TEXT PRIMARY KEY,
            vector_rowid INTEGER NOT NULL UNIQUE,
            source TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        )
    """)
    connection.commit()

    return connection, vector_store


# ============================================================
# 4. 执行一次增量同步
# ============================================================

def synchronize_index(
    chunks: list[Document],
    connection: sqlite3.Connection,
    vector_store: SQLiteVec
) -> dict[str, set[str]]:
    """比较当前 chunks 与数据库清单，只处理差异。"""

    current_chunks_by_id = {chunk.id: chunk for chunk in chunks}
    current_ids = set(current_chunks_by_id)

    stored_rows = connection.execute("""
        SELECT chunk_id, vector_rowid
        FROM index_manifest
    """).fetchall()

    stored_rowid_by_id = {
        row["chunk_id"]: row["vector_rowid"]
        for row in stored_rows
    }
    stored_ids = set(stored_rowid_by_id)

    new_ids = current_ids - stored_ids
    unchanged_ids = current_ids & stored_ids
    deleted_ids = stored_ids - current_ids

    # 先删除已经不存在的旧数据。
    for chunk_id in sorted(deleted_ids):
        vector_rowid = stored_rowid_by_id[chunk_id]

        connection.execute(
            "DELETE FROM knowledge_chunks_vec WHERE rowid = ?",
            (vector_rowid,)
        )
        connection.execute(
            "DELETE FROM knowledge_chunks WHERE rowid = ?",
            (vector_rowid,)
        )
        connection.execute(
            "DELETE FROM index_manifest WHERE chunk_id = ?",
            (chunk_id,)
        )

    connection.commit()

    # 只为新增数据计算 Embedding。
    new_chunks = [current_chunks_by_id[chunk_id] for chunk_id in sorted(new_ids)]

    if new_chunks:
        vector_rowids = vector_store.add_documents(new_chunks)
        indexed_at = datetime.now(timezone.utc).isoformat()

        for chunk, vector_rowid in zip(new_chunks, vector_rowids):
            connection.execute("""
                INSERT INTO index_manifest (
                    chunk_id, vector_rowid, source, indexed_at
                ) VALUES (?, ?, ?, ?)
            """, (
                chunk.id,
                int(vector_rowid),
                chunk.metadata["source"],
                indexed_at
            ))

        connection.commit()

    return {
        "new": new_ids,
        "unchanged": unchanged_ids,
        "deleted": deleted_ids
    }


def print_changes(title: str, changes: dict[str, set[str]]) -> None:
    """显示一次同步产生的三类差异。"""

    print("\n" + "=" * 60)
    print(title)
    print("新增：", len(changes["new"]))
    print("未变化：", len(changes["unchanged"]))
    print("删除：", len(changes["deleted"]))

    for chunk_id in sorted(changes["new"]):
        print("  [新增]", chunk_id)

    for chunk_id in sorted(changes["deleted"]):
        print("  [删除]", chunk_id)


# ============================================================
# 5. 主程序：在临时副本中演示一次文件更新
# ============================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
original_knowledge_directory = day_directory / "documents"
cache_directory = project_directory / ".fastembed_cache"

embedding_model = LocalChineseEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir=str(cache_directory)
)

# TemporaryDirectory 退出时自动清理，因此不会改动正式资料和正式数据库。
with tempfile.TemporaryDirectory(prefix="day6-update-demo-") as temp_name:
    demo_directory = Path(temp_name)
    demo_knowledge_directory = demo_directory / "knowledge"
    demo_database_path = demo_directory / "demo.db"

    shutil.copytree(
        original_knowledge_directory,
        demo_knowledge_directory
    )

    connection, vector_store = open_database(
        demo_database_path,
        embedding_model
    )

    print("本次使用临时知识库：", demo_knowledge_directory)
    print("不会修改正式 knowledge 目录。")

    # 第一次：数据库为空，所有 chunks 都属于新增。
    first_chunks = load_chunks(demo_knowledge_directory)
    first_changes = synchronize_index(
        first_chunks,
        connection,
        vector_store
    )
    print_changes("第一次同步：建立完整索引", first_changes)

    # 模拟用户更新一份真实文件。
    changed_file = demo_knowledge_directory / "txt" / "day6_langchain.txt"
    old_text = changed_file.read_text(encoding="utf-8")
    added_text = (
        "\n\n增量索引只处理新增和变化的文本块，"
        "未变化的文本块不需要重复生成向量。\n"
    )
    changed_file.write_text(old_text + added_text, encoding="utf-8")

    print("\n[修改测试文件]", changed_file.name)
    print("[追加内容]", added_text.strip())

    # 第二次：重新生成当前 chunks，再与数据库旧清单比较。
    second_chunks = load_chunks(demo_knowledge_directory)
    second_changes = synchronize_index(
        second_chunks,
        connection,
        vector_store
    )
    print_changes("第二次同步：文件内容发生变化", second_changes)

    main_count = connection.execute(
        "SELECT COUNT(*) FROM knowledge_chunks"
    ).fetchone()[0]
    vector_count = connection.execute(
        "SELECT COUNT(*) FROM knowledge_chunks_vec"
    ).fetchone()[0]
    manifest_count = connection.execute(
        "SELECT COUNT(*) FROM index_manifest"
    ).fetchone()[0]

    print("\n" + "=" * 60)
    print("同步后一致性检查：")
    print("当前切分结果：", len(second_chunks))
    print("正文表：", main_count)
    print("向量表：", vector_count)
    print("清单表：", manifest_count)
    print(
        "四处数量一致：",
        len(second_chunks) == main_count == vector_count == manifest_count
    )

    question = "增量索引有什么作用？"
    results = vector_store.similarity_search_with_score(question, k=2)

    print("\n检索问题：", question)
    print("检索结果：")

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"{rank}. 距离={round(distance, 4)}")
        print("   来源：", document.metadata["source"])
        print("   正文：", document.page_content)

    connection.close()

print("\n" + "=" * 60)
print("演示结束，临时知识库和临时数据库已自动清理。")
print("正式 knowledge 目录和 day6_knowledge.db 均未改变。")
