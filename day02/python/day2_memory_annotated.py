"""
LangChain / LangGraph Day 2：带摘要和 SQLite 持久化的多轮聊天

完整数据流：

    终端输入 str
        ↓
    HumanMessage(content=user_input)
        ↓
    graph.invoke(input=本轮消息, config=会话编号)
        ↓
    SqliteSaver 根据 thread_id 恢复旧 State
        ↓
    MessagesState 把本轮 HumanMessage 加入 messages
        ↓
    chatbot(state) 调用普通聊天模型
        ↓
    AIMessage 被加入 messages
        ↓
    should_summarize(state) 判断是否超过 6 条消息
        ↓
    如果超过：summarize_conversation(state)
        ↓
    更新 summary，并用 RemoveMessage 删除旧原文
        ↓
    SqliteSaver 保存最终 State
        ↓
    graph.invoke() 返回最新 State
"""

import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph


# ============================================================================
# 1. 读取 .env，创建模型调用对象
# ============================================================================

# load_dotenv() -> bool
#
# 作用：查找并读取项目中的 .env 文件，将其中的键值放进环境变量。
# 返回值通常表示是否成功找到并加载了文件；这里不需要保存这个返回值。
load_dotenv()

# os.getenv(name) -> str | None
#
# 如果环境变量存在，返回字符串；如果不存在，返回 None。
# 例如 model_name 可能是 "deepseek-chat"，数据类型是 str。
model_name: str | None = os.getenv("DEEPSEEK_MODEL")
api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
base_url: str | None = os.getenv("DEEPSEEK_BASE_URL")

# 在创建模型前主动检查配置，避免缺少变量时产生不容易理解的网络错误。
missing_variables: list[str] = []

if not model_name:
    missing_variables.append("DEEPSEEK_MODEL")
if not api_key:
    missing_variables.append("DEEPSEEK_API_KEY")
if not base_url:
    missing_variables.append("DEEPSEEK_BASE_URL")

if missing_variables:
    missing_text: str = ", ".join(missing_variables)
    raise ValueError(f".env 缺少必要配置：{missing_text}")

# ChatOpenAI(...) -> ChatOpenAI 对象
#
# 参数流动：
#   model_name ──→ model     决定调用哪个模型
#   api_key ─────→ api_key   用于身份验证
#   base_url ────→ base_url  决定请求发往哪个兼容接口
#
# llm 只负责根据输入消息生成回复，它自己不负责永久保存聊天历史。
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
)

# bind(max_tokens=250) 会返回一个预先绑定调用参数的 Runnable。
# 它与 llm 使用相同的模型、密钥和地址，但摘要调用最多生成约 250 Token。
# Token 不严格等于汉字数量。
summary_llm = llm.bind(max_tokens=250)


# ============================================================================
# 2. 定义图状态 State
# ============================================================================

class State(MessagesState):
    """整个对话图在节点之间传递和持久化的数据结构。

    从 MessagesState 继承得到：
        messages: list[BaseMessage]

    messages 中可包含 HumanMessage、AIMessage 等消息对象。
    MessagesState 对 messages 使用专门的 add_messages 合并规则，因此节点返回
    新消息时会追加，而 RemoveMessage 可以按照消息 id 删除旧消息。

    本类新增：
        summary: str

    summary 保存早期对话的压缩摘要。
    """

    summary: str


# ============================================================================
# 3. 普通聊天节点
# ============================================================================

def chatbot(state: State) -> dict[str, Any]:
    """根据摘要和近期消息调用模型，返回一条新的 AIMessage。

    参数：
        state: State
            LangGraph 在运行节点前自动传入的当前状态。
            它可能来自 SQLite 中的旧检查点，并已合并本轮 HumanMessage。

    返回：
        dict[str, Any]
            形如 {"messages": [response]}。
            response 是 AIMessage。LangGraph 会把它合并进 state["messages"]。
    """

    # dict.get(key, default) 在字段不存在时返回默认值。
    # 第一次对话还没有摘要，因此会得到空字符串 ""。
    summary: str = state.get("summary", "")

    # system_content: str
    # 这是每次调用模型时临时创建的系统规则，不直接保存进 MessagesState。
    system_content: str = """
你是一名简洁、友好的编程学习助手。
请根据用户的学习情况提供清晰、适合初学者的回答。
""".strip()

    # 非空字符串在 if 判断中为 True。
    # 如果已经有摘要，就通过 f-string 将它追加到系统规则中。
    if summary:
        system_content += f"""

以下是以前对话的摘要：
{summary}
"""

    # state["messages"] 是消息列表。
    # 前面的 * 表示列表展开，避免产生“列表里面套列表”。
    #
    # 假设状态消息是 [human_1, ai_1, human_2]，展开后相当于：
    # [SystemMessage(...), human_1, ai_1, human_2]
    messages = [
        SystemMessage(content=system_content),
        *state["messages"],
    ]

    # llm.invoke(messages) -> AIMessage
    # messages 参数包含：系统规则 + 旧摘要 + 最近原始对话 + 本轮问题。
    response = llm.invoke(messages)

    # 这里只返回本节点产生的“状态更新”，而不是自己复制完整 State。
    # MessagesState 会把 response 追加到原来的消息列表。
    return {
        "messages": [response],
    }


# ============================================================================
# 4. 条件路由函数
# ============================================================================

def should_summarize(state: State) -> str:
    """决定 chatbot 执行完成后应该前往哪个节点。

    返回 "summarize_conversation"：进入摘要节点。
    返回 END：结束本轮图执行。
    """

    # len(state["messages"]) -> int
    # 一轮正常对话通常有 1 条 HumanMessage + 1 条 AIMessage。
    message_count: int = len(state["messages"])

    # 三轮通常是 6 条；第四轮回答完成后通常变成 8 条，于是触发摘要。
    if message_count > 6:
        return "summarize_conversation"

    return END


# ============================================================================
# 5. 摘要节点
# ============================================================================

def summarize_conversation(state: State) -> dict[str, Any]:
    """生成或更新摘要，并删除较早的原始消息。

    输入：完整的当前 State。

    输出：
        summary  -> 新摘要字符串
        messages -> RemoveMessage 列表，用于删除旧消息

    最终保留最后 4 条原始消息，大约等于最近两轮对话。
    """

    old_summary: str = state.get("summary", "")

    # 已有摘要时，不从零开始，而是让模型结合旧摘要和新对话更新摘要。
    if old_summary:
        summary_request: str = f"""
这是以前保存的对话摘要：

{old_summary}

请结合最近的新对话更新摘要。

要求：
1. 不超过150个汉字
2. 保留用户姓名
3. 保留学习目标
4. 保留用户偏好
5. 保留已完成的学习内容
6. 保留尚未解决的问题
7. 删除寒暄和重复信息
8. 只输出更新后的摘要
"""
    else:
        summary_request = """
请总结以上对话。

要求：
1. 不超过150个汉字
2. 保留用户姓名
3. 保留学习目标
4. 保留用户偏好
5. 保留已完成的学习内容
6. 保留尚未解决的问题
7. 删除寒暄和重复信息
8. 只输出摘要
"""

    # 摘要请求只用于这一次模型调用，不会作为正常用户消息保存。
    # 数据流：当前消息 + 临时摘要指令 -> summary_llm。
    messages_for_summary = [
        *state["messages"],
        HumanMessage(content=summary_request),
    ]

    # summary_response 是 AIMessage。
    # summary_response.content 是模型生成的摘要正文。
    summary_response = summary_llm.invoke(messages_for_summary)

    # state["messages"][:-4] 表示：选择除最后 4 条以外的全部消息。
    # 对选中的每条旧消息，根据其唯一 id 创建一条删除指令。
    messages_to_delete: list[RemoveMessage] = [
        RemoveMessage(id=message.id)
        for message in state["messages"][:-4]
    ]

    # 返回两个状态更新：
    # 1. summary 字段被最新摘要替换；
    # 2. messages 合并器执行 RemoveMessage，删除旧原文并保留最后 4 条。
    return {
        "summary": summary_response.content,
        "messages": messages_to_delete,
    }


# ============================================================================
# 6. 构建 LangGraph 工作流
# ============================================================================

# StateGraph(State) 表示这个图中的所有节点都围绕 State 读取和更新数据。
builder = StateGraph(State)

# add_node(name, function)
# name 是图内部使用的节点名称；function 是进入该节点时执行的函数。
builder.add_node("chatbot", chatbot)
builder.add_node("summarize_conversation", summarize_conversation)

# 固定边：每次运行都从 START 进入 chatbot。
builder.add_edge(START, "chatbot")

# 条件边：chatbot 回答后调用 should_summarize，并根据返回值决定路线。
builder.add_conditional_edges("chatbot", should_summarize)

# 摘要节点执行完成后，本轮图执行结束。
builder.add_edge("summarize_conversation", END)

# 图结构：
#
# START -> chatbot -> should_summarize
#                         ├── END
#                         └── summarize_conversation -> END


# ============================================================================
# 7. 创建 SQLite Checkpointer
# ============================================================================

# 使用脚本所在目录构造绝对数据库路径。
# 这样即使从其他终端目录启动脚本，也会打开同一个数据库文件。
script_directory: Path = Path(__file__).resolve().parent
database_path: Path = (
    script_directory.parent
    / "database"
    / "day2_memory.db"
)

# sqlite3.connect(database) -> sqlite3.Connection
# check_same_thread=False 允许 LangGraph 的同步检查点实现安全使用该连接；
# SqliteSaver 自身使用锁协调访问。
connection: sqlite3.Connection = sqlite3.connect(
    database_path,
    check_same_thread=False,
)

# SqliteSaver(connection) 把 SQLite 连接包装成 LangGraph Checkpointer。
checkpointer = SqliteSaver(connection)

# compile(checkpointer=...) 创建真正可执行的图。
# 从此 graph.invoke() 会在每轮执行前后自动读取、保存检查点。
graph = builder.compile(checkpointer=checkpointer)


# ============================================================================
# 8. 设置会话配置
# ============================================================================

# config 是 graph.invoke() 的运行配置。
# thread_id 决定 Checkpointer 应该读取数据库中的哪一份对话。
config: dict[str, dict[str, str]] = {
    "configurable": {
        "thread_id": "xiaolin",
    }
}

# 会话恢复规则：
#   相同 database_path + 相同 thread_id -> 继续原会话
#   相同 database_path + 不同 thread_id -> 另一份独立会话
#   不同 database_path                    -> 另一份数据库


# ============================================================================
# 9. 命令行聊天循环
# ============================================================================

print("开始聊天，输入 exit 结束。")
print("当前会话：xiaolin")
print(f"数据库：{database_path}")

try:
    while True:
        # input(prompt) -> str
        # strip() -> str，删除首尾空格和换行。
        user_input: str = input("\n你：").strip()

        # 空字符串在布尔判断中为 False。
        if not user_input:
            print("请输入内容，不要只按回车。")
            continue

        # lower() 将英文转为小写，所以 exit、EXIT、Exit 都能退出。
        if user_input.lower() == "exit":
            break

        # graph.invoke(input, config=...) 的参数数据流：
        #
        # 第一个参数：本轮新增状态
        #   {"messages": [HumanMessage(content=user_input)]}
        #
        # 第二个参数：运行配置
        #   config["configurable"]["thread_id"] == "xiaolin"
        #
        # 内部执行：
        #   1. SqliteSaver 根据 thread_id 查询旧检查点；
        #   2. 找到则恢复 State，找不到则创建初始 State；
        #   3. 把本轮 HumanMessage 合并进 messages；
        #   4. 依照图的边执行 chatbot；
        #   5. 必要时执行 summarize_conversation；
        #   6. 把最终 State 写回 SQLite；
        #   7. 返回最终 State。
        result: dict[str, Any] = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input),
                ]
            },
            config=config,
        )

        # result["messages"] 是最终状态中的消息列表。
        # [-1] 取得最后一条消息，即本轮 chatbot 生成的 AIMessage。
        latest_message = result["messages"][-1]

        # latest_message.content 是回答正文。
        print("助手：", latest_message.content)

    # graph.get_state(config) 只读取指定会话的最新检查点：
    # 它不会添加消息、执行节点或调用模型。
    snapshot = graph.get_state(config)

    # snapshot.values 是最新 State 的字段字典。
    saved_messages = snapshot.values.get("messages", [])
    saved_summary = snapshot.values.get("summary", "暂无摘要")

    print("\n聊天结束")
    print("当前原始消息数量：", len(saved_messages))
    print("对话摘要：", saved_summary)

finally:
    # finally 无论正常退出还是发生异常都会执行。
    # 关闭连接，释放 SQLite 文件和相关资源。
    connection.close()
