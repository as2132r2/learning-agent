import os
import sqlite3
import warnings
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# SQLiteVec 当前位于 langchain-community。
# 这个包会显示整体迁移提示，与本次代码运行无关，学习时暂时隐藏。
warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ==================================================
# 1. 基础路径配置
# ==================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
cache_directory = project_directory / ".fastembed_cache"

# 默认把数据库保存在当前代码文件旁边。
# DAY5_VECTOR_DB 主要用于测试时临时替换数据库路径。
database_path = Path(
    os.getenv(
        "DAY5_VECTOR_DB",
        str(day_directory / "database" / "day5_vector_store.db")
    )
)

table_name = "knowledge_chunks"


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
# 3. 创建支持 sqlite-vec 的数据库连接
# ==================================================

connection = sqlite3.connect(
    database_path
)

# SQLiteVec 按列名读取查询结果，因此设置 sqlite3.Row。
connection.row_factory = sqlite3.Row

# sqlite-vec 是 SQLite 扩展，负责向量保存和距离查询。
connection.enable_load_extension(True)
sqlite_vec.load(connection)
connection.enable_load_extension(False)


# ==================================================
# 4. 创建持久化向量数据库
# ==================================================

# 初始化时，如果数据表不存在，SQLiteVec 会自动创建：
#
# knowledge_chunks
#     保存正文、metadata 和向量
#
# knowledge_chunks_vec
#     sqlite-vec 使用的向量索引虚拟表
vector_store = SQLiteVec(
    table=table_name,
    connection=connection,
    embedding=embedding_model
)


# ==================================================
# 5. 准备首次写入的知识库
# ==================================================

documents = [
    Document(
        page_content=(
            "LangChain 是一个用于开发大语言模型应用的框架，"
            "可以连接模型、提示词、工具、记忆和检索系统。"
        ),
        metadata={
            "source": "langchain.txt",
            "topic": "LangChain",
            "chunk_id": "chunk-001"
        }
    ),
    Document(
        page_content=(
            "LangGraph 使用图结构组织应用流程。"
            "节点负责处理数据，边决定流程走向，"
            "状态负责在节点之间传递数据。"
        ),
        metadata={
            "source": "langgraph.txt",
            "topic": "LangGraph",
            "chunk_id": "chunk-002"
        }
    ),
    Document(
        page_content=(
            "RAG 的中文名称是检索增强生成。"
            "它会先从外部知识库中检索与问题相关的资料，"
            "再把资料和用户问题一起交给大语言模型生成回答。"
        ),
        metadata={
            "source": "rag.txt",
            "topic": "RAG",
            "chunk_id": "chunk-003"
        }
    ),
    Document(
        page_content=(
            "Python 是一种语法简洁的编程语言，"
            "常用于自动化、数据分析、人工智能和后端开发。"
        ),
        metadata={
            "source": "python.txt",
            "topic": "Python",
            "chunk_id": "chunk-004"
        }
    ),
    Document(
        page_content=(
            "C 语言接近计算机底层，"
            "常用于操作系统、驱动程序和嵌入式开发。"
        ),
        metadata={
            "source": "c-language.txt",
            "topic": "C语言",
            "chunk_id": "chunk-005"
        }
    )
]


# ==================================================
# 6. 判断数据库中是否已经存在资料
# ==================================================

row = connection.execute(
    f"SELECT COUNT(*) AS total FROM {table_name}"
).fetchone()

stored_document_count = row["total"]


print("数据库路径：", database_path)
print("启动时已有文本块数量：", stored_document_count)


# ==================================================
# 7. 只在空数据库中写入一次
# ==================================================

if stored_document_count == 0:
    print("\n数据库为空，开始生成并保存文档向量……")

    stored_row_ids = vector_store.add_documents(
        documents=documents
    )

    print("首次建库完成。")
    print("SQLite 行编号：", stored_row_ids)

else:
    print("\n数据库已有资料，跳过重复写入。")
    print("直接使用上一次保存的正文、metadata 和向量。")


# 再次查询，确认当前数据库内的最终数量。
row = connection.execute(
    f"SELECT COUNT(*) AS total FROM {table_name}"
).fetchone()

print("当前文本块总数：", row["total"])


# ==================================================
# 8. 查看 SQLite 中实际创建的数据表
# ==================================================

table_rows = connection.execute(
    """
    SELECT name
    FROM sqlite_master
    WHERE type IN ('table', 'trigger')
    ORDER BY name
    """
).fetchall()


print("\n" + "=" * 60)
print("数据库中的表和触发器：")

for table_row in table_rows:
    print("-", table_row["name"])


# ==================================================
# 9. 从持久化向量库检索
# ==================================================

question = input("\n请输入问题：").strip()

if not question:
    print("问题不能为空。")
    connection.close()
    raise SystemExit(1)


# SQLiteVec 返回的是距离 distance：
# - 距离越小，表示越相关
# - 它与上一版“分数越大越相关”方向相反
search_results = (
    vector_store.similarity_search_with_score(
        query=question,
        k=3
    )
)


print("\n" + "=" * 60)
print("检索结果（距离越小越相关）：")

for rank, (document, distance) in enumerate(
    search_results,
    start=1
):
    print("\n排名：", rank)
    print("文本块编号：", document.metadata["chunk_id"])
    print("主题：", document.metadata["topic"])
    print("来源：", document.metadata["source"])
    print("距离：", round(distance, 4))
    print("正文：", document.page_content)


# ==================================================
# 10. 关闭数据库连接
# ==================================================

connection.close()


print("\n" + "=" * 60)
print("数据库连接已关闭，数据仍保存在磁盘中。")

print("\n本次数据流：")
print(
    "第一次运行：Document → Embedding → SQLite 文件"
)
print(
    "后续运行：SQLite 文件 → 跳过写入 → 直接检索"
)
print(
    "问题 → 问题向量 → sqlite-vec 距离查询 "
    "→ list[tuple[Document, distance]]"
)
