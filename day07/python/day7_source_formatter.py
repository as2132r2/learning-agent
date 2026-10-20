import json
import os
import re
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Literal

import sqlite_vec
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
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

conversation_database_path = (
    project_directory
    / "day07"
    / "database"
    / "day7_source_formatter.db"
)

cache_directory = project_directory / ".fastembed_cache"

# 来自 Day6 检索评测的距离阈值。
# SQLiteVec 距离越小越相关。
RELEVANCE_THRESHOLD = 0.95

# 最多允许模型改写并重新检索一次，防止图无限循环。
MAX_REWRITE_ATTEMPTS = 1

# 第一次答案不合格时，最多再生成一次。
# 总生成次数上限是 2，避免 generate_answer 和 grade_answer 无限循环。
MAX_GENERATION_ATTEMPTS = 2

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


class AnswerGrade(BaseModel):
    """答案质量检查器必须返回的结构化数据。"""

    grounded: bool = Field(
        description="答案中的事实是否完全受参考资料支持"
    )

    answers_question: bool = Field(
        description="答案是否直接、完整地回答了用户原始问题"
    )

    reason: str = Field(description="质量判断的简短理由")


router_llm = llm.with_structured_output(
    RouteDecision,
    method="function_calling"
)

answer_grader_llm = llm.with_structured_output(
    AnswerGrade,
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

    # 本轮已经生成答案的次数。
    generation_count: int

    # grade_answer 写入：accepted、retry 或 rejected。
    answer_status: str

    # 保存答案质量检查器给出的原因。
    answer_grade_reason: str

    # 确定性引用检查的结果：valid、retry 或 rejected。
    citation_status: str

    # 保存引用检查的具体说明。
    citation_reason: str


# ============================================================
# 6. 主路由节点
# ============================================================

def router_node(state: AgentState) -> dict:
    # 把持久化 messages 转换成路由器容易阅读的对话文本。
    # 当前用户消息已经由 add_messages 追加在列表末尾。
    conversation_text = get_buffer_string(state["messages"])

    decision: RouteDecision = router_llm.invoke([
        SystemMessage(
            content=(
                "你是问题路由器，只分类，不回答。\n"
                "LangChain、LangGraph、RAG、Python、C语言等"
                "课程资料问题 → knowledge。\n"
                "数学运算或数学表达式 → calculate。\n"
                "问候和闲聊 → chat。\n"
                "询问姓名、刚才的结果、前面说过什么等"
                "依赖对话历史的追问 → chat。\n"
                "不适合当前助手处理的请求 → unsupported。"
            )
        ),
        HumanMessage(
            content=(
                f"完整对话历史：\n{conversation_text}\n\n"
                f"当前需要分类的问题：{state['question']}"
            )
        )
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
        # 第一名通过后，再对每一份 Document 单独过滤。
        # 这样不会因为第一名相关，就把后面无关的结果也交给模型。
        relevant_documents = [
            document
            for document in documents
            if document.metadata["distance"] <= RELEVANCE_THRESHOLD
        ]

        print("判断结果：资料相关")
        print("原始检索数量：", len(documents))
        print("过滤后资料数量：", len(relevant_documents))

        for document in documents:
            distance = document.metadata["distance"]
            decision = (
                "保留"
                if distance <= RELEVANCE_THRESHOLD
                else "丢弃"
            )

            print(
                f"[{decision}] "
                f"{format_location(document)} | "
                f"距离={distance:.4f}"
            )

        return {
            "retrieval_status": "relevant",
            "retrieved_documents": relevant_documents
        }

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


def expand_context_node(state: AgentState) -> dict:
    """
    根据最佳命中块的位置，补充同一文件、同一页的前后邻居。

    质量判断已经完成，所以这里不再生成问题向量，
    只读取数据库中现有的正文和 metadata。
    """

    print("\n[expand_context 节点]")

    best_hit = state["retrieved_documents"][0]
    hit_source = best_hit.metadata.get("source")
    hit_page = best_hit.metadata.get("page")
    hit_chunk_id = best_hit.metadata.get("chunk_id")

    connection = sqlite3.connect(vector_database_path)
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)

    rows = connection.execute("""
        SELECT text, metadata
        FROM knowledge_chunks
    """).fetchall()

    location_chunks: list[Document] = []

    for row in rows:
        metadata = json.loads(row["metadata"]) or {}

        if (
            metadata.get("source") == hit_source
            and metadata.get("page") == hit_page
        ):
            location_chunks.append(Document(
                page_content=row["text"],
                metadata=metadata,
                id=metadata.get("chunk_id")
            ))

    connection.close()

    location_chunks.sort(
        key=lambda document:
            document.metadata.get("start_index", 0)
    )

    hit_index = next(
        index
        for index, document in enumerate(location_chunks)
        if document.metadata.get("chunk_id") == hit_chunk_id
    )

    # window_size=1：最多取前一个、命中块、后一个。
    start = max(0, hit_index - 1)
    end = min(len(location_chunks), hit_index + 2)
    expanded_documents = location_chunks[start:end]

    print("核心命中：", format_location(best_hit))
    print("核心 start_index：", best_hit.metadata.get("start_index"))
    print("扩展后数量：", len(expanded_documents))

    for index, document in enumerate(expanded_documents, start=1):
        print(f"\n邻近资料 {index}")
        print("start_index：", document.metadata.get("start_index"))
        print("正文：", document.page_content)

    # 后面的 generate_answer 继续读取同一个字段，
    # 但此时内容已经从“向量命中”变成了“按原文扩展后的上下文”。
    return {
        "retrieved_documents": expanded_documents
    }


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

    generation_count = state["generation_count"] + 1

    retry_instruction = ""
    if state["generation_count"] > 0:
        retry_instruction = (
            "\n这是重新生成。上一次答案没有通过质量检查，原因是："
            f"{state['answer_grade_reason']}。"
            "请修正该问题，但仍然只能使用参考资料。"
        )

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是一名严谨、简洁的编程学习助手。"
                "只能根据提供的参考资料回答问题。"
                "关键结论后必须标注资料编号，例如：[资料 1]。"
                "不要编造参考资料、文件名或页码。"
                f"{retry_instruction}"
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

    print(
        "答案生成次数：",
        generation_count,
        "/",
        MAX_GENERATION_ATTEMPTS
    )
    print("模型回答：", response.content)

    return {
        "result": response.content,
        "answer_sources": answer_sources,
        "generation_count": generation_count
    }


def validate_citations_node(state: AgentState) -> dict:
    """
    使用普通 Python 代码检查答案里的引用编号。

    这里不需要调用大模型，因为规则是确定的：
    1. 答案至少引用一份资料；
    2. 每个引用编号必须在实际资料数量范围内。
    """

    print("\n[validate_citations 节点]")

    # 例如：[资料 1]、[资料2]，提取其中的数字。
    citation_numbers = [
        int(number)
        for number in re.findall(
            r"\[资料\s*(\d+)\]",
            state["result"]
        )
    ]

    document_count = len(state["retrieved_documents"])
    invalid_numbers = sorted({
        number
        for number in citation_numbers
        if number < 1 or number > document_count
    })

    print("实际资料数量：", document_count)
    print("答案中的引用编号：", citation_numbers)

    if not citation_numbers:
        valid = False
        reason = "答案没有包含任何 [资料 N] 引用。"
    elif invalid_numbers:
        valid = False
        reason = (
            f"答案引用了不存在的资料编号：{invalid_numbers}。"
        )
    else:
        valid = True
        reason = "答案引用的资料编号全部真实存在。"

    if valid:
        status = "valid"
    elif state["generation_count"] < MAX_GENERATION_ATTEMPTS:
        status = "retry"
    else:
        status = "rejected"

    print("引用检查结果：", status)
    print("检查说明：", reason)

    return {
        "citation_status": status,
        "citation_reason": reason,
        # 如果需要重新生成，把确定性检查原因交给生成节点。
        "answer_grade_reason": reason
    }


def choose_citation_route(
    state: AgentState
) -> Literal["valid", "retry", "rejected"]:
    return state["citation_status"]


def grade_answer_node(state: AgentState) -> dict:
    """
    检查生成结果是否受资料支持，并且真正回答了原问题。

    与 grade_retrieval 不同：
    grade_retrieval 检查输入给模型的资料；
    grade_answer 检查模型已经生成的答案。
    """

    print("\n[grade_answer 节点]")

    context_parts: list[str] = []

    for rank, document in enumerate(
        state["retrieved_documents"],
        start=1
    ):
        context_parts.append(
            f"[资料 {rank} | {format_location(document)}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    grade: AnswerGrade = answer_grader_llm.invoke([
        SystemMessage(
            content=(
                "你是严格的 RAG 答案质量检查器，只做判断，不回答问题。"
                "grounded 只有在答案中的事实都能从参考资料找到支持时"
                "才是 true。answers_question 只有在答案直接且充分回答"
                "原始问题时才是 true。引用编号必须与资料对应。"
            )
        ),
        HumanMessage(
            content=(
                f"参考资料：\n{context}\n\n"
                f"用户原始问题：{state['original_question']}\n\n"
                f"待检查答案：{state['result']}"
            )
        )
    ])

    print("是否有资料支撑：", grade.grounded)
    print("是否回答了问题：", grade.answers_question)
    print("判断原因：", grade.reason)

    if grade.grounded and grade.answers_question:
        status = "accepted"
    elif state["generation_count"] < MAX_GENERATION_ATTEMPTS:
        status = "retry"
    else:
        status = "rejected"

    print("后续处理：", status)

    return {
        "answer_status": status,
        "answer_grade_reason": grade.reason
    }


def choose_answer_route(
    state: AgentState
) -> Literal["accepted", "retry", "rejected"]:
    """根据答案检查结果决定结束、重新生成或拒绝。"""

    return state["answer_status"]


def format_sources_node(state: AgentState) -> dict:
    """
    把答案中的 [资料 N] 转换成可追溯的真实来源清单。

    该节点只处理已经通过引用校验和语义质检的答案。
    来源位置直接读取 Document.metadata，不让模型自由生成。
    """

    print("\n[format_sources 节点]")

    citation_numbers = [
        int(number)
        for number in re.findall(
            r"\[资料\s*(\d+)\]",
            state["result"]
        )
    ]

    # 保留引用第一次出现的顺序，同时去除重复编号。
    unique_numbers = list(dict.fromkeys(citation_numbers))
    documents = state["retrieved_documents"]

    source_lines: list[str] = []
    used_sources: list[str] = []

    for number in unique_numbers:
        document = documents[number - 1]
        location = format_location(document)

        source_lines.append(f"- [资料 {number}] {location}")
        used_sources.append(location)

    source_section = "\n".join(source_lines)
    formatted_answer = (
        f"{state['result']}\n\n"
        f"参考来源：\n{source_section}"
    )

    print("答案实际引用的编号：", unique_numbers)
    print("生成的来源清单：")
    print(source_section)

    return {
        "result": formatted_answer,
        # 这里只保留答案真正引用过的来源，不再显示未使用资料。
        "answer_sources": used_sources
    }


def accept_answer_node(state: AgentState) -> dict:
    """只有通过质量检查的候选答案才写入持久化消息历史。"""

    print("\n[accept_answer 节点]")
    print("答案已通过检查，写入对话历史。")

    return {
        "messages": [AIMessage(content=state["result"])]
    }


def reject_answer_node(state: AgentState) -> dict:
    """连续两次答案都未通过质量检查时安全结束。"""

    refusal = (
        "检索到了相关资料，但生成的答案未通过质量检查，"
        "因此本次不提供可能不可靠的回答。"
    )

    print("\n[reject_answer 节点]")
    print("拒绝原因：", state["answer_grade_reason"])

    return {
        "messages": [AIMessage(content=refusal)],
        "result": refusal,
        "answer_sources": []
    }


def irrelevant_node(state: AgentState) -> dict:
    """检索质量未通过时安全结束。"""

    refusal = "根据当前知识库资料无法回答这个问题。"

    return {
        "messages": [AIMessage(content=refusal)],
        "result": refusal
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
    refusal = "当前助手不处理这类请求。"

    return {
        "messages": [AIMessage(content=refusal)],
        "result": refusal
    }


# ============================================================
# 11. 创建 LangGraph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("knowledge", knowledge_node)
builder.add_node("grade_retrieval", grade_retrieval_node)
builder.add_node("rewrite_question", rewrite_question_node)
builder.add_node("expand_context", expand_context_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("validate_citations", validate_citations_node)
builder.add_node("grade_answer", grade_answer_node)
builder.add_node("format_sources", format_sources_node)
builder.add_node("accept_answer", accept_answer_node)
builder.add_node("reject_answer", reject_answer_node)
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
        "generate": "expand_context",
        "rewrite": "rewrite_question",
        "reject": "irrelevant"
    }
)

# 改写完成后回到 knowledge，再执行一轮检索。
builder.add_edge("rewrite_question", "knowledge")

# 邻居扩展完成后，才把完整上下文交给生成节点。
builder.add_edge("expand_context", "generate_answer")

builder.add_edge("generate_answer", "validate_citations")

builder.add_conditional_edges(
    "validate_citations",
    choose_citation_route,
    {
        "valid": "grade_answer",
        "retry": "generate_answer",
        "rejected": "reject_answer"
    }
)

builder.add_conditional_edges(
    "grade_answer",
    choose_answer_route,
    {
        "accepted": "format_sources",
        "retry": "generate_answer",
        "rejected": "reject_answer"
    }
)

builder.add_edge("format_sources", "accept_answer")
builder.add_edge("accept_answer", END)
builder.add_edge("reject_answer", END)
builder.add_edge("irrelevant", END)
builder.add_edge("chat", END)
builder.add_edge("unsupported", END)

# SqliteSaver 会按 thread_id 保存每一轮执行后的图状态。
conversation_database_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_connection = sqlite3.connect(
    conversation_database_path,
    check_same_thread=False
)

checkpointer = SqliteSaver(checkpoint_connection)

graph = builder.compile(checkpointer=checkpointer)


# ============================================================
# 12. 持续聊天
# ============================================================

thread_id = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "xiaolin-day7"
)

config = {
    "configurable": {
        "thread_id": thread_id
    }
}

print("Day7 可追溯来源 Agent 已启动，输入 exit 结束。")
print("当前会话：", thread_id)
print("状态数据库：", conversation_database_path)

while True:
    user_input = input("\n你：").strip()

    if not user_input:
        print("请输入内容，不要只按回车。")
        continue

    if user_input.lower() == "exit":
        break

    # messages 使用 add_messages reducer：
    # 新 HumanMessage 会追加到数据库中已有的历史消息后面。
    turn_input: AgentState = {
        "original_question": user_input,
        "question": user_input,
        "route": "",
        "reason": "",
        "result": "",
        "retrieved_documents": [],
        "retrieval_status": "",
        "answer_sources": [],
        "rewrite_count": 0,
        "rewrite_history": [user_input],
        "generation_count": 0,
        "answer_status": "",
        "answer_grade_reason": "",
        "citation_status": "",
        "citation_reason": "",
        "messages": [HumanMessage(content=user_input)]
    }

    final_state = graph.invoke(
        turn_input,
        config=config
    )

    print("\n助手：", final_state["result"])
    print("[本轮路线]", final_state["route"])

    if final_state["rewrite_count"]:
        print("[检索问题历史]", final_state["rewrite_history"])

    if final_state["answer_sources"]:
        print("[参考位置]")
        for location in final_state["answer_sources"]:
            print("-", location)

    if final_state["answer_status"]:
        print("[答案质量]", final_state["answer_status"])
        print("[质检原因]", final_state["answer_grade_reason"])

    if final_state["citation_status"]:
        print("[引用检查]", final_state["citation_status"])
        print("[引用说明]", final_state["citation_reason"])


# ============================================================
# 13. 读取本会话最终保存的状态
# ============================================================

snapshot = graph.get_state(config)
saved_state = snapshot.values

print("\n聊天结束")
print("当前会话：", thread_id)
print("保存的消息数量：", len(saved_state.get("messages", [])))
print("最后一轮路线：", saved_state.get("route", ""))

checkpoint_connection.close()


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
#   │   expand_context
#   │       ↓
#   │   generate_answer → LLM
#   │       ↓
#   │   validate_citations（确定性检查引用编号）
#   │       ├── retry → generate_answer
#   │       └── rejected → reject_answer → END
#   │       ↓
#   │   grade_answer
#   │       ├── accepted → format_sources
#   │       │                  ↓
#   │       │             accept_answer → 写入历史 → END
#   │       ├── retry → generate_answer（最多再生成一次）
#   │       └── rejected → reject_answer → END
#   │
#   └── distance > 0.95
#           ↓ rewrite_count < 1
#       rewrite_question
#           ↓
#       knowledge（重新检索）
#           ↓
#       grade_retrieval
#           ├── relevant   → expand_context → generate_answer
#           │                                  ↓
#           │                              grade_answer
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
#
# 每轮结束：
# AgentState → SqliteSaver → thread_id 对应的持久化状态
