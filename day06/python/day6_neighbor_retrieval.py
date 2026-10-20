import json
import sqlite3
import warnings
from collections import defaultdict
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ============================================================
# 1. 配置与 Embedding 模型
# ============================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
database_path = day_directory / "database" / "day6_knowledge_cli.db"
cache_directory = project_directory / ".fastembed_cache"


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


embedding_model = LocalChineseEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir=str(cache_directory)
)


# ============================================================
# 2. 打开上一小步建立的数据库
# ============================================================

if not database_path.exists():
    print("数据库不存在，请先运行：python day06/python/day6_knowledge_cli.py sync")
    raise SystemExit(1)

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


# ============================================================
# 3. 从正文表读取全部 chunk 的位置数据
# ============================================================

def load_all_chunks() -> list[Document]:
    """读取正文与 metadata，不重新计算任何向量。"""

    rows = connection.execute("""
        SELECT text, metadata
        FROM knowledge_chunks
    """).fetchall()

    documents: list[Document] = []

    for row in rows:
        metadata = json.loads(row["metadata"]) or {}
        documents.append(Document(
            page_content=row["text"],
            metadata=metadata,
            id=metadata.get("chunk_id")
        ))

    return documents


all_chunks = load_all_chunks()


# ============================================================
# 4. 按“来源文件 + 页码”建立有序分组
# ============================================================

# key 的类型：tuple[str, int | None]
# value 的类型：list[Document]
chunks_by_location: dict[
    tuple[str, int | None],
    list[Document]
] = defaultdict(list)

for chunk in all_chunks:
    location_key = (
        chunk.metadata.get("source", "未知来源"),
        chunk.metadata.get("page")
    )
    chunks_by_location[location_key].append(chunk)

# 同一文件、同一页内，按照 start_index 从前到后排序。
for chunks in chunks_by_location.values():
    chunks.sort(
        key=lambda chunk: chunk.metadata.get("start_index", 0)
    )


# ============================================================
# 5. 找到命中 chunk 的前一个和后一个 chunk
# ============================================================

def get_neighbor_chunks(
    hit: Document,
    window_size: int = 1
) -> list[Document]:
    """
    返回命中文本块及其邻居。

    window_size=1：前一个 + 自己 + 后一个。
    window_size=2：前两个 + 自己 + 后两个。
    """

    location_key = (
        hit.metadata.get("source", "未知来源"),
        hit.metadata.get("page")
    )
    location_chunks = chunks_by_location[location_key]
    hit_chunk_id = hit.metadata.get("chunk_id")

    hit_index = next(
        index
        for index, chunk in enumerate(location_chunks)
        if chunk.metadata.get("chunk_id") == hit_chunk_id
    )

    start = max(0, hit_index - window_size)
    end = min(len(location_chunks), hit_index + window_size + 1)

    return location_chunks[start:end]


def format_location(document: Document) -> str:
    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")

    if page is None:
        return source

    return f"{source}，第 {page} 页"


# ============================================================
# 6. 先进行普通向量检索
# ============================================================

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    connection.close()
    raise SystemExit(1)

# 这一步先取一个最相似的命中块，重点观察它的邻居扩展。
hits: list[tuple[Document, float]] = (
    vector_store.similarity_search_with_score(question, k=1)
)

print("\n" + "=" * 60)
print("第一阶段：向量检索直接命中的 chunks")

for rank, (document, distance) in enumerate(hits, start=1):
    print(f"\n命中 {rank}")
    print("位置：", format_location(document))
    print("start_index：", document.metadata.get("start_index"))
    print("距离：", round(distance, 4))
    print("正文：", document.page_content)


# ============================================================
# 7. 为每个命中块扩展相邻上下文，并且去重
# ============================================================

expanded_chunks_by_id: dict[str, Document] = {}

for hit, _distance in hits:
    neighbors = get_neighbor_chunks(hit, window_size=1)

    for neighbor in neighbors:
        chunk_id = neighbor.metadata["chunk_id"]
        expanded_chunks_by_id[chunk_id] = neighbor


# 按来源、页码、原文位置排序，方便模型顺序阅读。
expanded_chunks = sorted(
    expanded_chunks_by_id.values(),
    key=lambda chunk: (
        chunk.metadata.get("source", ""),
        chunk.metadata.get("page") or 0,
        chunk.metadata.get("start_index", 0)
    )
)


print("\n" + "=" * 60)
print("第二阶段：扩展后的上下文")
print("直接命中数量：", len(hits))
print("扩展并去重后的数量：", len(expanded_chunks))

for index, document in enumerate(expanded_chunks, start=1):
    print(f"\n上下文 {index}")
    print("位置：", format_location(document))
    print("start_index：", document.metadata.get("start_index"))
    print("chunk_id：", document.metadata.get("chunk_id"))
    print("正文：", document.page_content)


# ============================================================
# 8. 拼成下一步可以交给大模型的完整 context
# ============================================================

context_parts: list[str] = []

for index, document in enumerate(expanded_chunks, start=1):
    context_parts.append(
        f"[上下文 {index} | {format_location(document)}]\n"
        f"{document.page_content}"
    )

context = "\n\n".join(context_parts)

print("\n" + "=" * 60)
print("最终 context：")
print(context)

connection.close()

print("\n" + "=" * 60)
print("数据流：")
print("问题 → 向量检索命中 chunk")
print("→ 根据 source + page 找到所在分组")
print("→ 根据 start_index 找到前后邻居")
print("→ 使用 chunk_id 去重")
print("→ 按原文顺序排列")
print("→ 拼接成更完整的 context")
