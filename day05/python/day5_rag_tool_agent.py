import argparse
import os
import re
import sqlite3
import warnings
from pathlib import Path

import sqlite_vec
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ==================================================
# 1. 基础配置
# ==================================================

load_dotenv()

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
project_directory = day_directory.parent
cache_directory = project_directory / ".fastembed_cache"

vector_database_path = (
    day_directory / "database" / "day5_vector_store.db"
)

conversation_database_path = (
    day_directory / "database" / "day5_rag_tool_agent.db"
)

vector_table_name = "knowledge_chunks"

# SQLiteVec 返回距离，超过该距离就视为资料不足。
MAX_RELEVANT_DISTANCE = 0.95


# ==================================================
# 2. 会话编号
# ==================================================

parser = argparse.ArgumentParser(
    description="具有 RAG 和计算工具的 Agent"
)

parser.add_argument(
    "thread_id",
    nargs="?",
    default="xiaolin-day5-agent"
)

args = parser.parse_args()

config = {
    "configurable": {
        "thread_id": args.thread_id
    },
    "recursion_limit": 12
}


# ==================================================
# 3. FastEmbed → LangChain 适配器
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
# 4. 打开持久化向量数据库
# ==================================================

vector_connection = sqlite3.connect(
    vector_database_path,
    check_same_thread=False
)

vector_connection.row_factory = sqlite3.Row
vector_connection.enable_load_extension(True)
sqlite_vec.load(vector_connection)
vector_connection.enable_load_extension(False)

vector_store = SQLiteVec(
    table=vector_table_name,
    connection=vector_connection,
    embedding=embedding_model
)


# ==================================================
# 5. 空数据库时创建知识库
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

stored_count = vector_connection.execute(
    f"SELECT COUNT(*) AS total FROM {vector_table_name}"
).fetchone()["total"]

if stored_count == 0:
    vector_store.add_documents(
        documents=documents
    )


# ==================================================
# 6. 定义 RAG 检索工具
# ==================================================

def normalize_retrieval_question(question: str) -> str:
    """扩展常见缩写，提高本地中文模型的检索稳定性。"""

    return re.sub(
        r"\bRAG\b",
        "RAG（检索增强生成）",
        question,
        flags=re.IGNORECASE
    )


@tool
def search_knowledge(question: str) -> str:
    """
    在本地知识库中检索事实资料。

    除数学计算和普通寒暄外，用户提出任何需要事实知识
    才能回答的问题时都必须使用。工具返回资料正文和真实来源；
    没有相关资料时也必须如实返回资料不足。
    """

    normalized_question = normalize_retrieval_question(
        question
    )

    print("\n[执行知识检索]")
    print("原始检索问题：", question)
    print("标准化问题：", normalized_question)

    candidate_results = (
        vector_store.similarity_search_with_score(
            query=normalized_question,
            k=3
        )
    )

    relevant_results = [
        (document, distance)
        for document, distance in candidate_results
        if distance <= MAX_RELEVANT_DISTANCE
    ]

    if not relevant_results:
        return (
            "知识库中没有找到足够相关的资料。"
            "请明确告诉用户根据现有资料无法回答。"
        )

    result_parts = []

    for number, (document, distance) in enumerate(
        relevant_results,
        start=1
    ):
        result_parts.append(
            f"[资料 {number}]\n"
            f"主题：{document.metadata['topic']}\n"
            f"来源：{document.metadata['source']}\n"
            f"距离：{distance:.4f}\n"
            f"内容：{document.page_content}"
        )

    return "\n\n".join(result_parts)


# ==================================================
# 7. 定义计算工具
# ==================================================

@tool
def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    print(f"\n[执行 add] {a} + {b}")
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(f"\n[执行 multiply] {a} × {b}")
    return a * b


tools = [
    search_knowledge,
    add,
    multiply
]


# ==================================================
# 8. 创建绑定工具的模型
# ==================================================

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

llm_with_tools = llm.bind_tools(
    tools
)


# ==================================================
# 9. 模型节点
# ==================================================

def assistant_node(state: MessagesState):
    """让模型回答，或提出一个或多个工具调用。"""

    messages = [
        SystemMessage(
            content=(
                "你是一名简洁、严谨的编程学习助手。"
                "除了纯数学计算、对话历史回忆和普通寒暄以外，"
                "任何需要外部事实知识才能回答的问题，都必须调用 "
                "search_knowledge。"
                "即使你凭自身知识知道答案，也不允许直接回答。"
                "数学加法使用 add，乘法使用 multiply。"
                "复杂计算可以分多轮调用工具。"
                "工具调用只是请求，必须等待 ToolMessage。"
                "知识检索工具没有返回相关资料时，明确说"
                "根据现有资料无法回答。"
                "绝对不能用模型自身记忆补充知识库中没有的事实。"
                "回答知识问题时，只引用工具结果中真实存在的来源。"
            )
        ),
        *state["messages"]
    ]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ==================================================
# 10. 模型后的条件路由
# ==================================================

def route_after_assistant(state: MessagesState):
    """有工具调用就去 tools，否则结束当前一轮。"""

    latest_message = state["messages"][-1]

    if latest_message.tool_calls:
        return "tools"

    return END


# ==================================================
# 11. 构建 Agent 图
# ==================================================

builder = StateGraph(
    MessagesState
)

builder.add_node(
    "assistant",
    assistant_node
)

# ToolNode 自动完成：
# AIMessage.tool_calls → 找到工具 → invoke → ToolMessage
builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(
    START,
    "assistant"
)

builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "tools": "tools",
        END: END
    }
)

# 工具执行后返回模型，让模型读取 ToolMessage。
# 模型也可以继续请求下一个工具。
builder.add_edge(
    "tools",
    "assistant"
)


# ==================================================
# 12. SQLite 保存 Agent 对话状态
# ==================================================

conversation_connection = sqlite3.connect(
    conversation_database_path,
    check_same_thread=False
)

checkpointer = SqliteSaver(
    conversation_connection
)

graph = builder.compile(
    checkpointer=checkpointer
)


# ==================================================
# 13. 聊天循环
# ==================================================

print("RAG 工具 Agent 已启动，输入 exit 结束。")
print("当前会话：", args.thread_id)
print("可用工具：search_knowledge、add、multiply")


try:
    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            print("请输入内容，不要只按回车。")
            continue

        if user_input.lower() == "exit":
            break

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            config=config
        )

        latest_message = result["messages"][-1]

        print("\n助手：")
        print(latest_message.content)


finally:
    snapshot = graph.get_state(config)

    print("\n聊天结束")
    print(
        "保存的消息数量：",
        len(snapshot.values.get("messages", []))
    )

    vector_connection.close()
    conversation_connection.close()
