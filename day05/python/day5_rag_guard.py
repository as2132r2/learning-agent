import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI


# ==================================================
# 1. 基础配置
# ==================================================

load_dotenv()

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent
cache_directory = project_directory / ".fastembed_cache"

# 检索分数低于这个值，就认为资料相关性不足。
# 这是当前小型示例知识库使用的教学阈值。
# 真实项目需要通过测试集评估后再确定。
RELEVANCE_THRESHOLD = 0.58

# 最多取回几个候选文本块。
TOP_K = 3


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
# 3. 创建知识库
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
# 4. 建立向量数据库
# ==================================================

vector_store = InMemoryVectorStore(
    embedding=embedding_model
)

vector_store.add_documents(
    documents=documents
)


# ==================================================
# 5. 创建生成模型和 Prompt
# ==================================================

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名简洁、严谨的编程学习助手。\n"
        "只能根据参考资料回答，不得补充资料之外的事实。\n"
        "不要自己生成来源列表，Python 程序会统一添加来源。\n\n"
        "参考资料：\n{context}"
    ),
    (
        "human",
        "用户问题：{question}"
    )
])

generation_chain = (
    prompt
    | llm
    | StrOutputParser()
)


# ==================================================
# 6. 格式化通过门槛的 Document
# ==================================================

def format_documents(
    relevant_results: list[tuple[Document, float]]
) -> str:
    """把相关 Document 和分数转换为 Prompt context。"""

    parts = []

    for number, (document, score) in enumerate(
        relevant_results,
        start=1
    ):
        parts.append(
            f"[资料 {number}]\n"
            f"来源：{document.metadata['source']}\n"
            f"相关性：{score:.4f}\n"
            f"内容：{document.page_content}"
        )

    return "\n\n".join(parts)


def collect_sources(
    relevant_results: list[tuple[Document, float]]
) -> list[str]:
    """从实际使用的 Document 中提取并去重来源。"""

    sources = []

    for document, _score in relevant_results:
        source = document.metadata.get(
            "source",
            "未知来源"
        )

        if source not in sources:
            sources.append(source)

    return sources


# ==================================================
# 7. 接收问题并取回候选资料
# ==================================================

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    raise SystemExit(1)

candidate_results = (
    vector_store.similarity_search_with_score(
        query=question,
        k=TOP_K
    )
)


print("\n" + "=" * 60)
print("向量数据库返回的候选资料：")

for rank, (document, score) in enumerate(
    candidate_results,
    start=1
):
    print(
        f"{rank}. "
        f"{document.metadata['topic']} "
        f"| 分数={score:.4f} "
        f"| 来源={document.metadata['source']}"
    )


# ==================================================
# 8. 使用阈值过滤候选资料
# ==================================================

relevant_results = [
    (document, score)
    for document, score in candidate_results
    if score >= RELEVANCE_THRESHOLD
]


print("\n相关性门槛：", RELEVANCE_THRESHOLD)
print("通过门槛的资料数量：", len(relevant_results))


# ==================================================
# 9. 资料不足时直接停止，不调用 LLM
# ==================================================

if not relevant_results:
    print("\n" + "=" * 60)
    print("RAG 最终回答：")
    print("根据现有资料无法回答。")

    print("\n来源：无")

    print("\n本次是否调用 DeepSeek：False")

    raise SystemExit(0)


# ==================================================
# 10. 资料足够时生成回答
# ==================================================

context = format_documents(
    relevant_results
)

sources = collect_sources(
    relevant_results
)


print("\n" + "=" * 60)
print("通过门槛并传给 Prompt 的 context：")
print(context)


answer_body = generation_chain.invoke({
    "context": context,
    "question": question
})


# 来源由 Python 根据实际使用的 Document 生成，
# 而不是让模型凭记忆编写。
source_text = "、".join(sources)

final_answer = (
    f"{answer_body}\n\n"
    f"来源：{source_text}"
)


print("\n" + "=" * 60)
print("RAG 最终回答：")
print(final_answer)

print("\n本次是否调用 DeepSeek：True")


# ==================================================
# 11. 完整数据流
# ==================================================

print("\n" + "=" * 60)
print("完整数据流：")
print(
    "问题 → 向量检索 → 候选结果和分数 "
    "→ threshold 过滤"
)
print(
    "无相关资料 → Python 固定回答 → 不调用 LLM"
)
print(
    "有相关资料 → context → DeepSeek → 回答正文 "
    "→ Python 添加真实来源"
)
