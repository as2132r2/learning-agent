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

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ==================================================
# 1. 路径与表名配置
# ==================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
knowledge_directory = day_directory / "documents"
cache_directory = project_directory / ".fastembed_cache"

database_path = Path(
    os.getenv(
        "DAY6_VECTOR_DB",
        str(day_directory / "database" / "day6_knowledge.db")
    )
)

vector_table_name = "knowledge_chunks"
manifest_table_name = "index_manifest"


# ==================================================
# 2. FastEmbed → LangChain 适配器
# ==================================================

class LocalChineseEmbeddings(Embeddings):
    """把 FastEmbed 包装成 LangChain Embeddings。"""

    def __init__(self, model_name: str, cache_dir: str):
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir
        )

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        return [
            vector.tolist()
            for vector in self.model.embed(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        vector = next(self.model.embed([text]))
        return vector.tolist()


embedding_model = LocalChineseEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir=str(cache_directory)
)


# ==================================================
# 3. 从真实目录加载原始文件
# ==================================================

loader = DirectoryLoader(
    path=str(knowledge_directory),
    glob=["**/*.txt", "**/*.md"],
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    recursive=True,
    silent_errors=False,
    use_multithreading=False
)

source_documents = loader.load()

source_documents.sort(
    key=lambda document: document.metadata["source"]
)


for document in source_documents:
    absolute_source = Path(
        document.metadata["source"]
    ).resolve()

    relative_source = absolute_source.relative_to(
        knowledge_directory.resolve()
    )

    document.metadata = {
        "source": relative_source.as_posix(),
        "file_name": absolute_source.name,
        "file_type": absolute_source.suffix.lower(),
        "title": absolute_source.stem
    }


# ==================================================
# 4. 把所有原始文档切成 chunks
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    add_start_index=True,
    separators=["\n\n", "\n", "。", "；", "，", ""]
)

current_chunks = text_splitter.split_documents(
    source_documents
)


# ==================================================
# 5. 为 chunk 生成稳定 ID
# ==================================================

def create_chunk_id(chunk: Document) -> str:
    identity_text = (
        f"source={chunk.metadata['source']}\n"
        f"start_index={chunk.metadata['start_index']}\n"
        f"content={chunk.page_content}"
    )

    full_hash = hashlib.sha256(
        identity_text.encode("utf-8")
    ).hexdigest()

    return f"chunk-{full_hash[:16]}"


for chunk in current_chunks:
    chunk_id = create_chunk_id(chunk)
    chunk.id = chunk_id
    chunk.metadata["chunk_id"] = chunk_id


# 使用字典建立：chunk_id → Document
current_chunks_by_id = {
    chunk.id: chunk
    for chunk in current_chunks
}

current_ids = set(
    current_chunks_by_id.keys()
)


# ==================================================
# 6. 创建 SQLiteVec 数据库连接
# ==================================================

connection = sqlite3.connect(
    database_path
)

connection.row_factory = sqlite3.Row
connection.enable_load_extension(True)
sqlite_vec.load(connection)
connection.enable_load_extension(False)

vector_store = SQLiteVec(
    table=vector_table_name,
    connection=connection,
    embedding=embedding_model
)


# ==================================================
# 7. 创建索引清单表
# ==================================================

# SQLiteVec 自己使用 rowid 标识数据。
# 我们使用稳定 chunk_id 标识业务文本块。
# index_manifest 负责保存两者的对应关系。
connection.execute(f"""
    CREATE TABLE IF NOT EXISTS {manifest_table_name} (
        chunk_id TEXT PRIMARY KEY,
        vector_rowid INTEGER NOT NULL UNIQUE,
        source TEXT NOT NULL,
        content_preview TEXT NOT NULL,
        indexed_at TEXT NOT NULL
    )
""")

connection.commit()


# ==================================================
# 8. 读取数据库中旧的 chunk ID
# ==================================================

stored_rows = connection.execute(f"""
    SELECT chunk_id, vector_rowid
    FROM {manifest_table_name}
""").fetchall()

stored_rowid_by_id = {
    row["chunk_id"]: row["vector_rowid"]
    for row in stored_rows
}

stored_ids = set(
    stored_rowid_by_id.keys()
)


# ==================================================
# 9. 计算新增、未变化和已删除集合
# ==================================================

new_ids = current_ids - stored_ids
unchanged_ids = current_ids & stored_ids
deleted_ids = stored_ids - current_ids


print("知识库目录：", knowledge_directory)
print("索引数据库：", database_path)

print("\n当前文件数量：", len(source_documents))
print("当前 chunk 数量：", len(current_ids))
print("数据库旧 chunk 数量：", len(stored_ids))

print("\n新增 chunk 数量：", len(new_ids))
print("未变化 chunk 数量：", len(unchanged_ids))
print("已删除 chunk 数量：", len(deleted_ids))


# ==================================================
# 10. 删除已经不存在的旧 chunks
# ==================================================

for chunk_id in sorted(deleted_ids):
    vector_rowid = stored_rowid_by_id[chunk_id]

    # SQLiteVec 当前没有实现 delete(ids=...)，
    # 所以按照清单中的 rowid 手动同步删除三处数据。
    connection.execute(
        f"DELETE FROM {vector_table_name}_vec WHERE rowid = ?",
        (vector_rowid,)
    )

    connection.execute(
        f"DELETE FROM {vector_table_name} WHERE rowid = ?",
        (vector_rowid,)
    )

    connection.execute(
        f"DELETE FROM {manifest_table_name} WHERE chunk_id = ?",
        (chunk_id,)
    )

    print("[删除]", chunk_id)


connection.commit()


# ==================================================
# 11. 只为新增 chunks 生成向量并写入
# ==================================================

new_chunks = [
    current_chunks_by_id[chunk_id]
    for chunk_id in sorted(new_ids)
]


if new_chunks:
    # add_documents() 返回 SQLite 主表中的 rowid 列表。
    new_rowids = vector_store.add_documents(
        documents=new_chunks
    )

    indexed_at = datetime.now(
        timezone.utc
    ).isoformat()

    for chunk, vector_rowid in zip(
        new_chunks,
        new_rowids
    ):
        connection.execute(
            f"""
            INSERT INTO {manifest_table_name} (
                chunk_id,
                vector_rowid,
                source,
                content_preview,
                indexed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                chunk.id,
                vector_rowid,
                chunk.metadata["source"],
                chunk.page_content[:80],
                indexed_at
            )
        )

        print(
            "[新增]",
            chunk.id,
            "→ rowid",
            vector_rowid
        )

    connection.commit()

else:
    print("\n没有新增 chunk，不需要生成新的文档向量。")


# ==================================================
# 12. 查看同步后的索引清单
# ==================================================

manifest_rows = connection.execute(f"""
    SELECT
        chunk_id,
        vector_rowid,
        source,
        content_preview,
        indexed_at
    FROM {manifest_table_name}
    ORDER BY source, chunk_id
""").fetchall()


print("\n" + "=" * 60)
print("同步后的索引清单：")

for row in manifest_rows:
    print("\nchunk_id：", row["chunk_id"])
    print("vector rowid：", row["vector_rowid"])
    print("来源：", row["source"])
    print("正文预览：", row["content_preview"])
    print("索引时间 UTC：", row["indexed_at"])


# ==================================================
# 13. 检查三处数据数量是否一致
# ==================================================

main_count = connection.execute(
    f"SELECT COUNT(*) AS total FROM {vector_table_name}"
).fetchone()["total"]

vector_count = connection.execute(
    f"SELECT COUNT(*) AS total FROM {vector_table_name}_vec"
).fetchone()["total"]

manifest_count = connection.execute(
    f"SELECT COUNT(*) AS total FROM {manifest_table_name}"
).fetchone()["total"]


print("\n" + "=" * 60)
print("一致性检查：")
print("主表数量：", main_count)
print("向量索引数量：", vector_count)
print("清单数量：", manifest_count)
print(
    "三处数量是否一致：",
    main_count == vector_count == manifest_count
)


# ==================================================
# 14. 使用同步后的数据库检索
# ==================================================

question = input("\n请输入检索问题：").strip()

if question:
    results = vector_store.similarity_search_with_score(
        query=question,
        k=3
    )

    print("\n检索结果（距离越小越相关）：")

    for rank, (document, distance) in enumerate(
        results,
        start=1
    ):
        print("\n排名：", rank)
        print("chunk_id：", document.metadata["chunk_id"])
        print("来源：", document.metadata["source"])
        print("距离：", round(distance, 4))
        print("正文：", document.page_content)

else:
    print("\n没有输入问题，本次只执行索引同步。")


# ==================================================
# 15. 关闭连接
# ==================================================

connection.close()


print("\n" + "=" * 60)
print("增量索引完成。")
print(
    "当前 chunks - 数据库 chunks "
    "→ new_ids"
)
print(
    "当前 chunks ∩ 数据库 chunks "
    "→ unchanged_ids"
)
print(
    "数据库 chunks - 当前 chunks "
    "→ deleted_ids"
)
