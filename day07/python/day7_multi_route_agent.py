import os
import sqlite3
import warnings
from pathlib import Path
from typing import Literal

import sqlite_vec
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.vectorstores.sqlitevec import SQLiteVec


# ============================================================
# 1. 项目路径和模型
# ============================================================

load_dotenv()

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent

vector_database_path = (
    project_directory
    / "day06"
    / "database"
    / "day6_knowledge_cli.db"
)

cache_directory = project_directory / ".fastembed_cache"

# 来自 Day6 检索评测的距离阈值。
# SQLiteVec 距离越小越相关。
RELEVANCE_THRESHOLD = 0.95

# 最多允许模型改写并重新检索一次，防止图无限循环。
MAX_REWRITE_ATTEMPTS = 1

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 路由器的结构化输出
# ============================================================

class RouteDecision(BaseModel):
    route: Literal[
        "knowledge",
        "calculate",
        "chat",
        "unsupported"
    ] = Field(description="用户问题应该进入的处理路线")

    reason: str = Field(description="选择这条路线的简短原因")


router_llm = llm.with_structured_output(
    RouteDecision,
    method="function_calling"
)


# ============================================================
# 3. 计算工具
# ============================================================

@tool
def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    print(f"[执行 add] {a} + {b}")
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(f"[执行 multiply] {a} × {b}")
    return a * b


tools = [add, multiply]
calculator_llm = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# ============================================================
# 4. 本地中文 Embedding
# ============================================================

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


def open_vector_store() -> tuple[sqlite3.Connection, SQLiteVec]:
    """
    只在进入 knowledge 分支后打开数据库和 Embedding 模型。

    calculate、chat、unsupported 分支不会执行这个函数。
    """

    if not vector_database_path.exists():
        raise FileNotFoundError(
            "没有找到 Day6 向量数据库，请先运行：\n"
            "python day06/python/day6_knowledge_cli.py sync"
        )

    embedding_model = LocalChineseEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_dir=str(cache_directory)
    )

    connection = sqlite3.connect(vector_database_path)
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)

    vector_store = SQLiteVec(
        table="knowledge_chunks",
        connection=connection,
        embedding=embedding_model
    )

    return connection, vector_store


# ============================================================
# 5. 图状态
# ============================================================

class AgentState(MessagesState):
    # original_question 始终保存用户最初的问题。
    original_question: str

    # question 是当前用于检索的问题，改写后会被覆盖。
    question: str
    route: str
    reason: str
    result: str

    # knowledge 分支写入的原始检索结果。
    retrieved_documents: list[Document]

    # grade_retrieval 节点写入：relevant 或 irrelevant。
    retrieval_status: str

    # generate_answer 节点从 metadata 中确定性提取的来源位置。
    answer_sources: list[str]

    # 已经执行过的问题改写次数。
    rewrite_count: int

    # 保存每一次实际用于检索的问题。
    rewrite_history: list[str]


# ============================================================
# 6. 主路由节点
# ============================================================

def router_node(state: AgentState) -> dict:
    decision: RouteDecision = router_llm.invoke([
        SystemMessage(
            content=(
                "你是问题路由器，只分类，不回答。\n"
                "LangChain、LangGraph、RAG、Python、C语言等"
                "课程资料问题 → knowledge。\n"
                "数学运算或数学表达式 → calculate。\n"
                "问候和闲聊 → chat。\n"
                "不适合当前助手处理的请求 → unsupported。"
            )
        ),
        HumanMessage(content=state["question"])
    ])

    print("\n[router 节点]")
    print("路线：", decision.route)
    print("原因：", decision.reason)

    return {
        "route": decision.route,
        "reason": decision.reason
    }


def choose_main_route(
    state: AgentState
) -> Literal[
    "knowledge",
    "calculate",
    "chat",
    "unsupported"
]:
    return state["route"]


# ============================================================
# 7. knowledge：执行真实向量检索
# ============================================================

def format_location(document: Document) -> str:
    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")

    if page is None:
        return source

    return f"{source}，第 {page} 页"


def knowledge_node(state: AgentState) -> dict:
    """查询 Day6 SQLiteVec，返回最相关的三个 Document。"""

    print("\n[knowledge 节点]")
    print("准备查询：", vector_database_path)

    connection, vector_store = open_vector_store()

    search_results: list[tuple[Document, float]] = (
        vector_store.similarity_search_with_score(
            state["question"],
            k=3
        )
    )

    documents: list[Document] = []
    result_parts: list[str] = []

    for rank, (document, distance) in enumerate(
        search_results,
        start=1
    ):
        # 把距离补充到 metadata，方便后续节点继续判断质量。
        document.metadata["distance"] = distance
        documents.append(document)

        location = format_location(document)

        print(f"\n检索排名 {rank}")
        print("位置：", location)
        print("距离：", round(distance, 4))
        print("正文：", document.page_content)

        result_parts.append(
            f"{rank}. {location} | 距离={distance:.4f}\n"
            f"{document.page_content}"
        )

    connection.close()

    return {
        "retrieved_documents": documents,
        "result": "\n\n".join(result_parts)
    }


# ============================================================
# 8. 检索质量判断节点
# ============================================================

def grade_retrieval_node(state: AgentState) -> dict:
    """
    使用最佳检索距离进行确定性判断。

    这里不调用大模型，因为 Day6 已经通过评测集得到了阈值。
    """

    print("\n[grade_retrieval 节点]")

    documents = state["retrieved_documents"]

    if not documents:
        print("没有检索到任何 Document。")
        return {"retrieval_status": "irrelevant"}

    best_document = documents[0]
    best_distance = best_document.metadata["distance"]

    print("最佳来源：", format_location(best_document))
    print("最佳距离：", round(best_distance, 4))
    print("判断阈值：", RELEVANCE_THRESHOLD)

    if best_distance <= RELEVANCE_THRESHOLD:
        print("判断结果：资料相关")
        return {"retrieval_status": "relevant"}

    print("判断结果：资料不足")
    return {"retrieval_status": "irrelevant"}


def choose_retrieval_route(
    state: AgentState
) -> Literal["generate", "rewrite", "reject"]:
    """决定生成答案、改写重试，还是最终拒绝。"""

    if state["retrieval_status"] == "relevant":
        return "generate"

    if state["rewrite_count"] < MAX_REWRITE_ATTEMPTS:
        return "rewrite"

    return "reject"


def rewrite_question_node(state: AgentState) -> dict:
    """让模型把原问题改写成更适合向量检索的问题。"""

    print("\n[rewrite_question 节点]")
    print("改写前：", state["question"])

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是检索问题改写器。"
                "请把用户问题改写成一个语义清楚、关键词明确、"
                "适合知识库向量检索的独立问题。"
                "必须保留原意，不要回答问题，不要添加原问题没有的事实。"
                "只输出改写后的问题，不要解释。"
            )
        ),
        HumanMessage(content=state["question"])
    ])

    rewritten_question = response.content.strip()

    print("改写后：", rewritten_question)
    print(
        "重试次数：",
        state["rewrite_count"] + 1,
        "/",
        MAX_REWRITE_ATTEMPTS
    )

    return {
        "question": rewritten_question,
        "rewrite_count": state["rewrite_count"] + 1,
        "rewrite_history": [
            *state["rewrite_history"],
            rewritten_question
        ],
        # 清空上一轮检索结果，下一轮由 knowledge 重新写入。
        "retrieved_documents": [],
        "retrieval_status": ""
    }


def generate_answer_node(state: AgentState) -> dict:
    """把检索到的 Document 组织成 context，再调用模型回答。"""

    print("\n[generate_answer 节点]")

    context_parts: list[str] = []

    for rank, document in enumerate(
        state["retrieved_documents"],
        start=1
    ):
        location = format_location(document)

        context_parts.append(
            f"[资料 {rank} | {location}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    print("发送给模型的 context：")
    print(context)

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是一名严谨、简洁的编程学习助手。"
                "只能根据提供的参考资料回答问题。"
                "关键结论后必须标注资料编号，例如：[资料 1]。"
                "不要编造参考资料、文件名或页码。"
            )
        ),
        HumanMessage(
            content=(
                f"参考资料：\n{context}\n\n"
                f"用户原始问题：{state['original_question']}"
            )
        )
    ])

    # 来源由程序从 metadata 提取，不让模型自由编造。
    answer_sources = list(dict.fromkeys(
        format_location(document)
        for document in state["retrieved_documents"]
    ))

    print("模型回答：", response.content)

    return {
        "messages": [response],
        "result": response.content,
        "answer_sources": answer_sources
    }


def irrelevant_node(state: AgentState) -> dict:
    """检索质量未通过时安全结束。"""

    return {
        "result": "根据当前知识库资料无法回答这个问题。"
    }


# ============================================================
# 9. calculate：模型与工具循环
# ============================================================

def calculator_node(state: AgentState) -> dict:
    response = calculator_llm.invoke([
        SystemMessage(
            content=(
                "你是数学计算助手。"
                "必须使用 add、multiply 工具完成计算。"
                "复杂表达式可以分多轮调用工具。"
            )
        ),
        *state["messages"]
    ])

    print("\n[calculator 节点]")

    if response.tool_calls:
        print("模型请求工具：", response.tool_calls)
        return {"messages": [response]}

    print("模型生成最终答案：", response.content)
    return {
        "messages": [response],
        "result": response.content
    }


def choose_calculator_route(
    state: AgentState
) -> Literal["tools", "done"]:
    latest_message = state["messages"][-1]
    return "tools" if latest_message.tool_calls else "done"


# ============================================================
# 10. 其他分支
# ============================================================

def chat_node(state: AgentState) -> dict:
    """处理不需要知识库和工具的普通对话。"""

    print("\n[chat 节点]")

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是一名简洁、友好的编程学习助手。"
                "当前问题属于普通聊天，不需要调用工具，"
                "也不需要查询知识库。"
                "请直接自然地回复用户。"
            )
        ),
        *state["messages"]
    ])

    print("模型回答：", response.content)

    return {
        "messages": [response],
        "result": response.content
    }


def unsupported_node(state: AgentState) -> dict:
    return {"result": "当前助手不处理这类请求。"}


# ============================================================
# 11. 创建 LangGraph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("knowledge", knowledge_node)
builder.add_node("grade_retrieval", grade_retrieval_node)
builder.add_node("rewrite_question", rewrite_question_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("irrelevant", irrelevant_node)
builder.add_node("calculator", calculator_node)
builder.add_node("tools", tool_node)
builder.add_node("chat", chat_node)
builder.add_node("unsupported", unsupported_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    choose_main_route,
    {
        "knowledge": "knowledge",
        "calculate": "calculator",
        "chat": "chat",
        "unsupported": "unsupported"
    }
)

builder.add_conditional_edges(
    "calculator",
    choose_calculator_route,
    {
        "tools": "tools",
        "done": END
    }
)

builder.add_edge("tools", "calculator")

# knowledge 检索完成后，必须先经过质量判断。
builder.add_edge("knowledge", "grade_retrieval")

builder.add_conditional_edges(
    "grade_retrieval",
    choose_retrieval_route,
    {
        "generate": "generate_answer",
        "rewrite": "rewrite_question",
        "reject": "irrelevant"
    }
)

# 改写完成后回到 knowledge，再执行一轮检索。
builder.add_edge("rewrite_question", "knowledge")

builder.add_edge("generate_answer", END)
builder.add_edge("irrelevant", END)
builder.add_edge("chat", END)
builder.add_edge("unsupported", END)

graph = builder.compile()


# ============================================================
# 12. 运行程序
# ============================================================

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    raise SystemExit(1)

initial_state: AgentState = {
    "original_question": question,
    "question": question,
    "route": "",
    "reason": "",
    "result": "",
    "retrieved_documents": [],
    "retrieval_status": "",
    "answer_sources": [],
    "rewrite_count": 0,
    "rewrite_history": [question],
    "messages": [HumanMessage(content=question)]
}

final_state = graph.invoke(initial_state)


# ============================================================
# 13. 显示最终状态
# ============================================================

print("\n" + "=" * 60)
print("最终路线：", final_state["route"])
print("路由原因：", final_state["reason"])
print("检索质量：", final_state["retrieval_status"] or "未执行检索")
print("问题改写次数：", final_state["rewrite_count"])
print("检索问题历史：", final_state["rewrite_history"])
print("最终结果：")
print(final_state["result"])

if final_state["answer_sources"]:
    print("\n程序核验的参考位置：")

    for location in final_state["answer_sources"]:
        print("-", location)

print("\n检索到的 Document 数量：", len(
    final_state["retrieved_documents"]
))


# ============================================================
# knowledge 分支的数据流
# ============================================================
#
# router
#   ↓ route=knowledge
# knowledge_node
#   ↓ open_vector_store()
# Day6 SQLiteVec
#   ↓ similarity_search_with_score(question, k=3)
# list[tuple[Document, distance]]
#   ↓
# retrieved_documents
#   ↓
# grade_retrieval_node
#   ├── distance <= 0.95
#   │       ↓
#   │   generate_answer → LLM → END
#   │
#   └── distance > 0.95
#           ↓ rewrite_count < 1
#       rewrite_question
#           ↓
#       knowledge（重新检索）
#           ↓
#       grade_retrieval
#           ├── relevant   → generate_answer → END
#           └── irrelevant → 最终拒绝 → END
#
# chat 分支：
# router → chat → 普通模型回答 → END
#
# calculate 分支：
# router → calculator ↔ tools → END
#
# unsupported 分支：
# router → unsupported → 安全拒绝 → END
