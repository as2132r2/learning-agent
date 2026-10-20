import os
import sqlite3
import warnings
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
# 1. 加载配置
# ============================================================

load_dotenv()

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
database_path = day_directory / "database" / "day6_knowledge_cli.db"
cache_directory = project_directory / ".fastembed_cache"


# ============================================================
# 2. 创建 Embedding 模型
# ============================================================

class LocalChineseEmbeddings(Embeddings):
    """把 FastEmbed 包装成 LangChain Embeddings。"""

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
# 3. 创建聊天模型
# ============================================================

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 4. 打开已有向量数据库
# ============================================================

if not database_path.exists():
    print("没有找到向量数据库：", database_path)
    print("请先运行：python day06/python/day6_knowledge_cli.py sync")
    raise SystemExit(1)

connection = sqlite3.connect(database_path)

# SQLite 默认把查询结果返回为 tuple，例如：
# ("正文", "{...metadata...}", 0.91)
#
# SQLiteVec 内部需要按照列名读取结果，例如：
# row["metadata"]
#
# 因此必须把 row_factory 设置为 sqlite3.Row。
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
# 5. 检索相关资料
# ============================================================

def format_location(document: Document) -> str:
    """把 Document.metadata 转换成人类容易阅读的来源位置。"""

    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")

    if page is None:
        return source

    return f"{source}，第 {page} 页"

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    connection.close()
    raise SystemExit(1)

# 返回值类型：list[tuple[Document, float]]
# 每个元素包含一份 Document 和它的向量距离。
search_results: list[tuple[Document, float]] = (
    vector_store.similarity_search_with_score(
        question,
        k=3
    )
)

print("\n" + "=" * 60)
print("检索结果（距离越小越相关）：")

for rank, (document, distance) in enumerate(
    search_results,
    start=1
):
    print(f"\n排名：{rank}")
    print("位置：", format_location(document))
    print("距离：", round(distance, 4))
    print("正文：", document.page_content)


# ============================================================
# 6. 把检索结果整理成模型上下文
# ============================================================

context_parts: list[str] = []

for rank, (document, distance) in enumerate(
    search_results,
    start=1
):
    location = format_location(document)

    context_part = (
        f"[资料 {rank} | {location}]\n"
        f"内容：{document.page_content}"
    )
    context_parts.append(context_part)

# 用两个换行把不同资料隔开，得到一段完整上下文。
context = "\n\n".join(context_parts)

print("\n" + "=" * 60)
print("发送给模型的参考资料：")
print(context)


# ============================================================
# 7. 组合提示词并调用模型
# ============================================================

system_message = SystemMessage(
    content=(
        "你是一名严谨的编程学习助手。"
        "请只根据用户消息中提供的参考资料回答问题。"
        "如果参考资料不足以回答，就明确说："
        "根据当前知识库资料无法回答。"
        "不要使用参考资料之外的知识进行猜测。"
        "回答中的关键结论后面请标注对应的资料编号，"
        "例如：[资料 1]。"
        "回答应当简洁、清楚。"
    )
)

human_message = HumanMessage(
    content=(
        f"参考资料：\n{context}\n\n"
        f"用户问题：{question}"
    )
)

# messages 的类型是 list[BaseMessage]。
# llm.invoke() 返回 AIMessage。
response = llm.invoke([
    system_message,
    human_message
])


# ============================================================
# 8. 显示回答和确定性来源
# ============================================================

print("\n" + "=" * 60)
print("助手回答：")
print(response.content)

# 来源位置不让模型自由生成，而是从 metadata 中确定性提取。
# dict.fromkeys() 在去重的同时保留原来的检索排名顺序。
source_locations = list(dict.fromkeys(
    format_location(document)
    for document, _distance in search_results
))

print("\n程序核验的参考位置：")

for location in source_locations:
    print("-", location)

connection.close()


# ============================================================
# 数据流
# ============================================================
#
# 用户问题 str
#     ↓ embed_query()
# 问题向量 list[float]
#     ↓ SQLiteVec 相似度检索
# list[tuple[Document, distance]]
#     ↓ 拼接正文和 metadata
# context str
#     ↓ SystemMessage + HumanMessage
# llm.invoke(messages)
#     ↓
# AIMessage
#     ↓
# 回答正文 + 程序提取的参考来源
