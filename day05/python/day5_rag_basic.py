import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI


# ==================================================
# 1. 读取环境变量
# ==================================================

load_dotenv()


# ==================================================
# 2. FastEmbed → LangChain 适配器
# ==================================================

class LocalChineseEmbeddings(Embeddings):
    """把 FastEmbed 包装成 LangChain Embeddings。"""

    def __init__(
        self,
        model_name: str,
        cache_dir: str
    ):
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir
        )

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """把多个文档字符串转换为多个向量。"""

        return [
            vector.tolist()
            for vector in self.model.embed(texts)
        ]

    def embed_query(
        self,
        text: str
    ) -> list[float]:
        """把一个问题字符串转换为一个向量。"""

        vector = next(
            self.model.embed([text])
        )

        return vector.tolist()


# ==================================================
# 3. 创建真实中文 Embedding 模型
# ==================================================

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent
cache_directory = project_directory / ".fastembed_cache"

embedding_model = LocalChineseEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    cache_dir=str(cache_directory)
)


# ==================================================
# 4. 创建知识库 Document
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
            "节点负责处理数据，边决定流程走向，"
            "状态负责在节点之间传递数据。"
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
            "它会先从外部知识库中检索与问题相关的资料，"
            "再把资料和用户问题一起交给大语言模型生成回答。"
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
# 5. 建立向量数据库
# ==================================================

vector_store = InMemoryVectorStore(
    embedding=embedding_model
)

vector_store.add_documents(
    documents=documents
)


# ==================================================
# 6. 创建 Retriever
# ==================================================

# k=2 表示每次找出最相关的两个 Document。
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)


# ==================================================
# 7. 把 list[Document] 格式化成 context 字符串
# ==================================================

def format_documents(
    retrieved_documents: list[Document]
) -> str:
    """
    把 Retriever 返回的 Document 列表转换为字符串。

    输入：
        list[Document]

    输出：
        str
    """

    formatted_parts = []

    for number, document in enumerate(
        retrieved_documents,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "未知来源"
        )

        topic = document.metadata.get(
            "topic",
            "未知主题"
        )

        formatted_part = (
            f"[资料 {number}]\n"
            f"主题：{topic}\n"
            f"来源：{source}\n"
            f"内容：{document.page_content}"
        )

        formatted_parts.append(
            formatted_part
        )

    # 两份资料之间使用两个换行分隔。
    return "\n\n".join(
        formatted_parts
    )


# ==================================================
# 8. 创建 RAG Prompt
# ==================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名简洁、严谨的编程学习助手。\n"
        "请只根据下面提供的参考资料回答问题。\n"
        "如果参考资料不足以回答，就明确说："
        "根据现有资料无法回答。\n"
        "回答最后请列出实际使用的来源文件名。\n\n"
        "参考资料：\n{context}"
    ),
    (
        "human",
        "用户问题：{question}"
    )
])


# ==================================================
# 9. 创建 DeepSeek 聊天模型
# ==================================================

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ==================================================
# 10. 构建完整 RAG 链
# ==================================================

# RunnablePassthrough()：
# 把用户问题原样传给 question。
#
# retriever | format_documents：
# 问题 str
# → retriever.invoke()
# → list[Document]
# → format_documents()
# → context str
#
# 第一个字典 Runnable 的输出：
# {
#     "context": str,
#     "question": str
# }
rag_chain = (
    {
        "context": (
            retriever
            | format_documents
        ),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ==================================================
# 11. 接收用户问题
# ==================================================

question = input(
    "请输入问题："
).strip()

if not question:
    print("问题不能为空。")
    raise SystemExit(1)


# ==================================================
# 12. 单独查看 Retriever 找到的资料
# ==================================================

# 这里单独调用一次 Retriever，只是为了学习时观察中间数据。
# 正式 RAG 链在 invoke() 时还会自动检索一次。
retrieved_documents = retriever.invoke(
    question
)


print("\n" + "=" * 60)
print("Retriever 找到的参考资料：")

for number, document in enumerate(
    retrieved_documents,
    start=1
):
    print("\n" + "-" * 60)
    print("资料编号：", number)
    print("Document ID：", document.id)
    print("主题：", document.metadata["topic"])
    print("来源：", document.metadata["source"])
    print("正文：", document.page_content)


# 查看格式化以后、准备放入 Prompt 的 context。
context = format_documents(
    retrieved_documents
)


print("\n" + "=" * 60)
print("传给 Prompt 的 context：")
print(context)


# ==================================================
# 13. 调用完整 RAG 链
# ==================================================

answer = rag_chain.invoke(
    question
)


print("\n" + "=" * 60)
print("RAG 最终回答：")
print(answer)


# ==================================================
# 14. 显示最终数据类型和数据流
# ==================================================

print("\n" + "=" * 60)
print("最终回答的数据类型：")
print(type(answer))

print("\n是否可以当作普通 str 使用：")
print(isinstance(answer, str))

print("\n完整数据流：")
print(
    "问题 str "
    "→ Retriever "
    "→ list[Document] "
    "→ format_documents() "
    "→ context str"
)
print(
    "context + question "
    "→ ChatPromptTemplate "
    "→ DeepSeek "
    "→ AIMessage "
    "→ StrOutputParser "
    "→ answer str"
)
