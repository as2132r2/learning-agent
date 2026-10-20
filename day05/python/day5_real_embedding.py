from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


# ==================================================
# 1. 创建 FastEmbed → LangChain 适配器
# ==================================================

class LocalChineseEmbeddings(Embeddings):
    """
    把 FastEmbed 本地模型包装成 LangChain Embeddings。

    LangChain 的向量数据库只要求对象提供两个方法：

    embed_documents(list[str]) -> list[list[float]]
    embed_query(str)          -> list[float]
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str
    ):
        # TextEmbedding 是 FastEmbed 提供的真实向量模型对象。
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir
        )

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """批量把文档字符串转换为向量。"""

        # self.model.embed(texts) 返回一个可迭代对象，
        # 其中每个向量是 NumPy 数组。
        numpy_vectors = self.model.embed(texts)

        # .tolist() 把 NumPy 数组转换成普通 list[float]，
        # 满足 LangChain Embeddings 接口要求。
        return [
            vector.tolist()
            for vector in numpy_vectors
        ]

    def embed_query(
        self,
        text: str
    ) -> list[float]:
        """把一个用户问题转换为向量。"""

        # embed() 接收的是 list[str]，所以把一个问题放入列表。
        numpy_vector = next(
            self.model.embed([text])
        )

        return numpy_vector.tolist()


# ==================================================
# 2. 配置本地中文 Embedding 模型
# ==================================================

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent
cache_directory = project_directory / ".fastembed_cache"

# BAAI/bge-small-zh-v1.5：
# - 能理解中文语义
# - 在本机 CPU 上运行
# - 第一次运行需要下载模型
# - 后续运行直接读取本地缓存
embedding_model = LocalChineseEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir=str(cache_directory)
)


print("真实 Embedding 对象类型：")
print(type(embedding_model))


# ==================================================
# 3. 创建多个知识库 Document
# ==================================================

documents = [
    Document(
        page_content=(
            "LangChain 是一个用于开发大语言模型应用的框架，"
            "可以连接模型、提示词、工具、记忆和检索系统。"
        ),
        metadata={
            "source": "langchain.txt",
            "topic": "LangChain"
        },
        id="chunk-001"
    ),
    Document(
        page_content=(
            "LangGraph 使用图结构组织应用流程。"
            "节点处理数据，边决定流程走向，状态在节点间传递。"
        ),
        metadata={
            "source": "langgraph.txt",
            "topic": "LangGraph"
        },
        id="chunk-002"
    ),
    Document(
        page_content=(
            "RAG 的中文名称是检索增强生成。"
            "它先从外部知识库检索相关资料，"
            "再把资料和问题一起交给大语言模型生成回答。"
        ),
        metadata={
            "source": "rag.txt",
            "topic": "RAG"
        },
        id="chunk-003"
    ),
    Document(
        page_content=(
            "Python 是一种语法简洁的编程语言，"
            "常用于自动化、数据分析、人工智能和后端开发。"
        ),
        metadata={
            "source": "python.txt",
            "topic": "Python"
        },
        id="chunk-004"
    ),
    Document(
        page_content=(
            "C 语言接近计算机底层，"
            "常用于操作系统、驱动程序和嵌入式开发。"
        ),
        metadata={
            "source": "c-language.txt",
            "topic": "C语言"
        },
        id="chunk-005"
    )
]


# ==================================================
# 4. 观察真实向量
# ==================================================

sample_text = documents[2].page_content

sample_vector = embedding_model.embed_query(
    sample_text
)


print("\n" + "=" * 60)
print("示例文本：")
print(sample_text)

print("\n真实向量的数据类型：")
print(type(sample_vector))

print("\n真实向量维度：")
print(len(sample_vector))

# 完整向量很长，只显示前 10 个数字。
print("\n向量的前 10 个数字：")
print(sample_vector[:10])


# ==================================================
# 5. 建立向量数据库
# ==================================================

vector_store = InMemoryVectorStore(
    embedding=embedding_model
)

stored_ids = vector_store.add_documents(
    documents=documents
)


print("\n" + "=" * 60)
print("保存到向量数据库的 ID：")
print(stored_ids)


# ==================================================
# 6. 使用真实语义进行检索
# ==================================================

questions = [
    "什么是检索增强生成？",
    "哪种语言适合开发操作系统？",
    "如何用图来组织大模型工作流？"
]


for question in questions:
    results = vector_store.similarity_search_with_score(
        query=question,
        k=3
    )

    print("\n" + "=" * 60)
    print("问题：", question)
    print("检索排名：")

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):
        print(
            f"{rank}. "
            f"主题={document.metadata['topic']}，"
            f"分数={score:.4f}"
        )
        print("   正文：", document.page_content)


# ==================================================
# 7. 和测试 Embedding 的区别
# ==================================================

print("\n" + "=" * 60)
print("DeterministicFakeEmbedding：")
print("生成稳定测试数字，但不理解文字含义。")

print("\nBAAI/bge-small-zh-v1.5：")
print("根据中文语义生成向量，适合真正的相似度检索。")


# ==================================================
# 8. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "中文 Document "
    "→ BGE 中文 Embedding "
    "→ 真实语义向量 "
    "→ InMemoryVectorStore"
)
print(
    "中文问题 "
    "→ 问题向量 "
    "→ 语义相似度排序 "
    "→ 相关 Document"
)
