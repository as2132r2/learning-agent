import argparse
import os
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

import sqlite_vec
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


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
    day_directory / "database" / "day5_conversational_rag.db"
)

vector_table_name = "knowledge_chunks"


# ==================================================
# 2. 命令行中的会话编号
# ==================================================

parser = argparse.ArgumentParser(
    description="带持久化记忆的对话式 RAG"
)

parser.add_argument(
    "thread_id",
    nargs="?",
    default="xiaolin-day5"
)

args = parser.parse_args()

config = {
    "configurable": {
        "thread_id": args.thread_id
    }
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
# 5. 空数据库时写入示例知识库
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
# 6. 创建 DeepSeek 模型
# ==================================================

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ==================================================
# 7. 定义 LangGraph 状态
# ==================================================

class ConversationalRAGState(TypedDict):
    # add_messages 表示新消息追加到旧消息后面，
    # 而不是直接覆盖整个 messages 列表。
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    # 根据历史对话改写出的独立问题。
    standalone_question: str

    # 检索后准备传给生成模型的参考资料。
    context: str

    # 实际检索并使用的来源文件名。
    sources: list[str]


# ==================================================
# 8. 把消息历史格式化为字符串
# ==================================================

def format_history(
    messages: list[BaseMessage]
) -> str:
    """把消息对象列表转换为问题改写模型可读的文本。"""

    lines = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            role = type(message).__name__

        lines.append(
            f"{role}：{message.content}"
        )

    return "\n".join(lines)


# ==================================================
# 9. 检索术语标准化
# ==================================================

def normalize_retrieval_question(
    question: str
) -> str:
    """
    扩展知识库中的关键缩写，提高中文语义检索稳定性。

    这里只修改送给 Embedding 模型的检索问题，
    不修改用户保存在 messages 中的原始消息。
    """

    normalized_question = re.sub(
        r"\bRAG\b",
        "RAG（检索增强生成）",
        question,
        flags=re.IGNORECASE
    )

    return normalized_question


# ==================================================
# 10. 节点一：把追问改写为独立问题
# ==================================================

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你的任务是根据对话历史，把用户最新问题改写成"
        "一个不依赖历史也能看懂的独立检索问题。"
        "只输出改写后的问题，不要回答问题。"
        "如果最新问题已经完整，就原样输出。"
    ),
    (
        "human",
        "对话历史：\n{history}\n\n"
        "用户最新问题：{question}"
    )
])

rewrite_chain = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)


def rewrite_question_node(
    state: ConversationalRAGState
):
    """结合历史，把当前追问改写为独立问题。"""

    messages = state["messages"]
    latest_question = str(
        messages[-1].content
    )

    # 第一次提问没有历史，不需要额外调用模型改写。
    if len(messages) == 1:
        standalone_question = latest_question

    else:
        history_messages = messages[:-1]

        standalone_question = rewrite_chain.invoke({
            "history": format_history(
                history_messages
            ),
            "question": latest_question
        })

    # 对模型改写出的独立问题再做术语标准化。
    normalized_question = normalize_retrieval_question(
        str(standalone_question)
    )

    print("\n[独立检索问题]")
    print(standalone_question)

    print("\n[标准化后的检索问题]")
    print(normalized_question)

    return {
        "standalone_question": normalized_question
    }


# ==================================================
# 11. 节点二：使用独立问题检索
# ==================================================

def retrieve_node(
    state: ConversationalRAGState
):
    """检索相关资料并生成 context。"""

    results = vector_store.similarity_search_with_score(
        query=state["standalone_question"],
        k=2
    )

    context_parts = []
    sources = []

    print("\n[检索结果，距离越小越相关]")

    for number, (document, distance) in enumerate(
        results,
        start=1
    ):
        source = document.metadata["source"]

        print(
            f"{number}. "
            f"{document.metadata['topic']} "
            f"| 距离={distance:.4f} "
            f"| 来源={source}"
        )

        context_parts.append(
            f"[资料 {number}]\n"
            f"来源：{source}\n"
            f"内容：{document.page_content}"
        )

        if source not in sources:
            sources.append(source)

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources
    }


# ==================================================
# 12. 节点三：根据资料生成回答
# ==================================================

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名简洁、严谨的编程学习助手。"
        "只能根据参考资料回答。"
        "不要自己编写来源，Python 会统一添加。\n\n"
        "参考资料：\n{context}"
    ),
    (
        "human",
        "请回答这个独立问题：{question}"
    )
])


def generate_answer_node(
    state: ConversationalRAGState
):
    """使用 context 回答，并把答案加入消息历史。"""

    response = llm.invoke(
        answer_prompt.invoke({
            "context": state["context"],
            "question": state["standalone_question"]
        })
    )

    sources_text = "、".join(
        state["sources"]
    )

    final_content = (
        f"{response.content}\n\n"
        f"来源：{sources_text}"
    )

    return {
        "messages": [
            AIMessage(
                content=final_content
            )
        ]
    }


# ==================================================
# 13. 构建 LangGraph
# ==================================================

builder = StateGraph(
    ConversationalRAGState
)

builder.add_node(
    "rewrite_question",
    rewrite_question_node
)

builder.add_node(
    "retrieve",
    retrieve_node
)

builder.add_node(
    "generate_answer",
    generate_answer_node
)

builder.add_edge(
    START,
    "rewrite_question"
)

builder.add_edge(
    "rewrite_question",
    "retrieve"
)

builder.add_edge(
    "retrieve",
    "generate_answer"
)

builder.add_edge(
    "generate_answer",
    END
)


# ==================================================
# 14. 使用 SQLite 保存对话状态
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
# 15. 多轮聊天循环
# ==================================================

print("对话式 RAG 已启动，输入 exit 结束。")
print("当前会话：", args.thread_id)
print("向量数据库：", vector_database_path)
print("对话数据库：", conversation_database_path)


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
                    HumanMessage(
                        content=user_input
                    )
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
