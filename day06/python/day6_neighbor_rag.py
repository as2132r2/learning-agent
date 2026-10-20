import json
import os
import sqlite3
import warnings
from collections import defaultdict
from pathlib import Path

import sqlite_vec
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ============================================================
# 1. 配置模型和数据库
# ============================================================

load_dotenv()

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

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 打开已有向量数据库
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
# 3. 读取全部 chunk，并按来源和页码分组
# ============================================================

rows = connection.execute("""
    SELECT text, metadata
    FROM knowledge_chunks
""").fetchall()

all_chunks: list[Document] = []

for row in rows:
    metadata = json.loads(row["metadata"]) or {}

    all_chunks.append(Document(
        page_content=row["text"],
        metadata=metadata,
        id=metadata.get("chunk_id")
    ))


chunks_by_location: dict[
    tuple[str, int | None],
    list[Document]
] = defaultdict(list)

for chunk in all_chunks:
    key = (
        chunk.metadata.get("source", "未知来源"),
        chunk.metadata.get("page")
    )
    chunks_by_location[key].append(chunk)

for chunks in chunks_by_location.values():
    chunks.sort(
        key=lambda chunk: chunk.metadata.get("start_index", 0)
    )


# ============================================================
# 4. 邻居扩展函数
# ============================================================

def format_location(document: Document) -> str:
    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")

    if page is None:
        return source

    return f"{source}，第 {page} 页"


def get_neighbor_chunks(
    hit: Document,
    window_size: int = 1
) -> list[Document]:
    """返回命中 chunk 以及同一文件、同一页的前后邻居。"""

    key = (
        hit.metadata.get("source", "未知来源"),
        hit.metadata.get("page")
    )
    location_chunks = chunks_by_location[key]
    hit_chunk_id = hit.metadata.get("chunk_id")

    hit_index = next(
        index
        for index, chunk in enumerate(location_chunks)
        if chunk.metadata.get("chunk_id") == hit_chunk_id
    )

    start = max(0, hit_index - window_size)
    end = min(len(location_chunks), hit_index + window_size + 1)

    return location_chunks[start:end]


# ============================================================
# 5. 第一次检索：找到核心命中 chunk
# ============================================================

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    connection.close()
    raise SystemExit(1)

hits: list[tuple[Document, float]] = (
    vector_store.similarity_search_with_score(
        question,
        k=1
    )
)

print("\n" + "=" * 60)
print("向量检索命中：")

for hit, distance in hits:
    print("位置：", format_location(hit))
    print("start_index：", hit.metadata.get("start_index"))
    print("距离：", round(distance, 4))
    print("正文：", hit.page_content)


# ============================================================
# 6. 扩展相邻文本块，并去重排序
# ============================================================

expanded_by_id: dict[str, Document] = {}

for hit, _distance in hits:
    for chunk in get_neighbor_chunks(hit, window_size=1):
        expanded_by_id[chunk.metadata["chunk_id"]] = chunk

expanded_chunks = sorted(
    expanded_by_id.values(),
    key=lambda chunk: (
        chunk.metadata.get("source", ""),
        chunk.metadata.get("page") or 0,
        chunk.metadata.get("start_index", 0)
    )
)

print("\n" + "=" * 60)
print("扩展后的文本块数量：", len(expanded_chunks))

for index, document in enumerate(expanded_chunks, start=1):
    print(f"\n上下文 {index}")
    print("位置：", format_location(document))
    print("start_index：", document.metadata.get("start_index"))
    print("正文：", document.page_content)


# ============================================================
# 7. 组织完整 context
# ============================================================

context_parts: list[str] = []

for index, document in enumerate(expanded_chunks, start=1):
    context_parts.append(
        f"[资料 {index} | {format_location(document)}]\n"
        f"{document.page_content}"
    )

context = "\n\n".join(context_parts)

print("\n" + "=" * 60)
print("发送给模型的完整 context：")
print(context)


# ============================================================
# 8. 把扩展后的 context 交给聊天模型
# ============================================================

system_message = SystemMessage(
    content=(
        "你是一名严谨、简洁的编程学习助手。"
        "请只根据提供的参考资料回答。"
        "如果资料不足，就明确回答：根据当前知识库资料无法回答。"
        "回答中的关键结论后面必须标注资料编号，例如：[资料 1]。"
        "不要编造资料编号、文件名或页码。"
    )
)

human_message = HumanMessage(
    content=(
        f"参考资料：\n{context}\n\n"
        f"用户问题：{question}"
    )
)

response = llm.invoke([
    system_message,
    human_message
])


# ============================================================
# 9. 输出回答和程序核验的来源
# ============================================================

print("\n" + "=" * 60)
print("助手回答：")
print(response.content)

source_locations = list(dict.fromkeys(
    format_location(document)
    for document in expanded_chunks
))

print("\n程序核验的参考位置：")

for location in source_locations:
    print("-", location)

connection.close()


# ============================================================
# 完整数据流
# ============================================================
#
# 用户问题
#   ↓ Embedding
# 向量检索命中核心 chunk
#   ↓ source + page + start_index
# 找到前后邻居
#   ↓ chunk_id 去重 + 原文顺序排序
# 扩展后的 context
#   ↓ SystemMessage + HumanMessage
# llm.invoke()
#   ↓
# 带资料编号的回答 + 程序核验的来源位置
