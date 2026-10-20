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
# 1. 配置
# ============================================================

load_dotenv()

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
database_path = day_directory / "database" / "day6_knowledge_cli.db"
cache_directory = project_directory / ".fastembed_cache"

# SQLiteVec 返回的是距离：数值越小，语义越接近。
# 这个值来自 day6_retrieval_evaluation.py 的评测结果。
RELEVANCE_THRESHOLD = 0.95


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
# 2. 打开向量数据库
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
# 3. 检索最相关的核心 chunk
# ============================================================

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    connection.close()
    raise SystemExit(1)

hits: list[tuple[Document, float]] = (
    vector_store.similarity_search_with_score(question, k=1)
)

if not hits:
    print("知识库中没有可以检索的内容。")
    connection.close()
    raise SystemExit(0)

best_document, best_distance = hits[0]

print("\n最相关文本块：")
print("来源：", best_document.metadata.get("source"))
print("页码：", best_document.metadata.get("page"))
print("距离：", round(best_distance, 4))
print("相关性阈值：", RELEVANCE_THRESHOLD)
print("正文：", best_document.page_content)


# ============================================================
# 4. 相关性守门
# ============================================================

# 注意：这里比较的是“距离”，不是“相似度”。
# 距离大于阈值，说明最好的检索结果仍然不够相关。
if best_distance > RELEVANCE_THRESHOLD:
    print("\n" + "=" * 60)
    print("相关性判断：未通过")
    print("根据当前知识库资料无法回答这个问题。")
    print("本次不会扩展邻居，也不会调用聊天模型。")

    connection.close()
    raise SystemExit(0)

print("\n相关性判断：通过")
print("继续扩展邻居并调用聊天模型。")


# ============================================================
# 5. 只有通过守门后，才读取全部 chunk 并建立位置分组
# ============================================================

rows = connection.execute("""
    SELECT text, metadata
    FROM knowledge_chunks
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


def format_location(document: Document) -> str:
    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")
    return source if page is None else f"{source}，第 {page} 页"


def get_neighbors(hit: Document, window_size: int = 1) -> list[Document]:
    key = (
        hit.metadata.get("source", "未知来源"),
        hit.metadata.get("page")
    )
    chunks = chunks_by_location[key]
    hit_id = hit.metadata.get("chunk_id")

    hit_index = next(
        index
        for index, chunk in enumerate(chunks)
        if chunk.metadata.get("chunk_id") == hit_id
    )

    start = max(0, hit_index - window_size)
    end = min(len(chunks), hit_index + window_size + 1)
    return chunks[start:end]


# ============================================================
# 6. 扩展上下文
# ============================================================

expanded_chunks = get_neighbors(best_document, window_size=1)

context_parts = [
    f"[资料 {index} | {format_location(document)}]\n"
    f"{document.page_content}"
    for index, document in enumerate(expanded_chunks, start=1)
]

context = "\n\n".join(context_parts)

print("\n" + "=" * 60)
print("扩展后的 context：")
print(context)


# ============================================================
# 7. 通过守门后才创建并调用聊天模型
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

system_message = SystemMessage(
    content=(
        "你是一名严谨、简洁的编程学习助手。"
        "只能根据参考资料回答。"
        "关键结论后必须标注资料编号，例如：[资料 1]。"
        "不要编造资料、文件名或页码。"
    )
)

human_message = HumanMessage(
    content=(
        f"参考资料：\n{context}\n\n"
        f"用户问题：{question}"
    )
)

response = llm.invoke([system_message, human_message])

print("\n" + "=" * 60)
print("助手回答：")
print(response.content)

print("\n程序核验的参考位置：")

for location in dict.fromkeys(
    format_location(document)
    for document in expanded_chunks
):
    print("-", location)

connection.close()


# ============================================================
# 数据流
# ============================================================
#
# 问题 → 最佳向量检索结果 → 距离阈值判断
#
# 距离 > 0.95 → 拒绝回答，不调用 LLM
#
# 距离 <= 0.95 → 邻居扩展 → context → LLM → 引用回答
