"""Day 3 最终项目：带工具、记忆、摘要与流式输出的 LangGraph Agent。

主数据流：
终端输入 -> HumanMessage -> SQLite 恢复 State -> chatbot -> ToolNode（可循环）
-> 摘要路由 -> SQLite 保存 State -> 流式打印最终回答。
"""

import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage
)
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph
)
from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)


# ==================================================
# 1. 模型配置
# ==================================================

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


summary_llm = llm.bind(
    max_tokens=250
)


# ==================================================
# 2. 定义工具
# ==================================================

@tool
def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    print(f"\n[执行 add] {a} + {b}")

    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """计算第一个数字减去第二个数字。"""

    print(f"\n[执行 subtract] {a} - {b}")

    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(f"\n[执行 multiply] {a} × {b}")

    return a * b


@tool
def divide(a: float, b: float) -> float:
    """计算第一个数字除以第二个数字。第二个数字不能为零。"""

    print(f"\n[执行 divide] {a} ÷ {b}")

    if b == 0:
        raise ValueError("除数不能为零")

    return a / b


tools = [
    add,
    subtract,
    multiply,
    divide
]


llm_with_tools = llm.bind_tools(
    tools
)


# ==================================================
# 3. 定义对话状态
# ==================================================

class State(MessagesState):
    """保存近期消息和早期对话摘要。"""

    summary: str


# ==================================================
# 4. 聊天节点
# ==================================================

def chatbot(
    state: State
) -> dict[str, Any]:
    """根据摘要和历史消息调用模型。"""

    summary = state.get(
        "summary",
        ""
    )

    system_content = (
        "你是一名友好、简洁的学习助手。"
        "遇到加减乘除计算时必须使用工具，"
        "不要自己心算。"
        "如果工具执行失败，请向用户解释原因，"
        "不要使用相同的错误参数反复调用工具。"
        "普通知识问题不需要调用计算工具。"
        "请记住用户提供的重要信息。"
    )

    if summary:
        system_content += f"""

以前对话的摘要：
{summary}
"""

    messages = [
        SystemMessage(
            content=system_content
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
# 5. 工具错误处理
# ==================================================

def handle_tool_error(
    error: Exception
) -> str:
    """把工具异常转换成模型能够理解的信息。"""

    error_type = type(error).__name__
    error_message = str(error)

    return (
        "工具执行失败。"
        f"异常类型：{error_type}；"
        f"异常信息：{error_message}。"
        "请向用户解释原因，"
        "不要使用相同参数重复调用。"
    )


tool_node = ToolNode(
    tools,
    handle_tool_errors=handle_tool_error
)


# ==================================================
# 6. 统计用户消息
# ==================================================

def count_user_messages(
    state: State
) -> int:
    """统计当前原始状态中的用户消息数量。"""

    return sum(
        1
        for message in state["messages"]
        if isinstance(
            message,
            HumanMessage
        )
    )


# ==================================================
# 7. 路由判断
# ==================================================

def route_after_chatbot(
    state: State
):
    """选择工具节点、摘要节点或结束。"""

    tool_route = tools_condition(
        state
    )

    if tool_route == "tools":
        return "tools"

    user_message_count = (
        count_user_messages(state)
    )

    if user_message_count > 4:
        return "summarize_conversation"

    return END


# ==================================================
# 8. 摘要节点
# ==================================================

def summarize_conversation(
    state: State
) -> dict[str, Any]:
    """更新摘要，并保留最近两轮完整消息。"""

    old_summary = state.get(
        "summary",
        ""
    )

    if old_summary:
        summary_request = f"""
这是以前保存的摘要：

{old_summary}

请结合最近的新对话更新摘要。

要求：
1. 不超过150个汉字
2. 保留用户的重要信息
3. 保留学习目标和偏好
4. 保留重要计算结果
5. 保留尚未解决的问题
6. 删除寒暄和重复内容
7. 只输出更新后的摘要
"""
    else:
        summary_request = """
请总结以上对话。

要求：
1. 不超过150个汉字
2. 保留用户的重要信息
3. 保留学习目标和偏好
4. 保留重要计算结果
5. 保留尚未解决的问题
6. 删除寒暄和重复内容
7. 只输出摘要
"""

    messages_for_summary = [
        *state["messages"],
        HumanMessage(
            content=summary_request
        )
    ]

    summary_response = (
        summary_llm.invoke(
            messages_for_summary
        )
    )

    human_message_indexes = [
        index
        for index, message
        in enumerate(state["messages"])
        if isinstance(
            message,
            HumanMessage
        )
    ]

    messages_to_delete = []

    if len(human_message_indexes) >= 2:
        keep_from_index = (
            human_message_indexes[-2]
        )

        messages_to_delete = [
            RemoveMessage(
                id=message.id
            )
            for message
            in state["messages"][
                :keep_from_index
            ]
        ]

    print(
        "\n[摘要完成，删除旧消息：",
        len(messages_to_delete),
        "条]"
    )

    return {
        "summary": (
            summary_response.content
        ),
        "messages": messages_to_delete
    }


# ==================================================
# 9. 构建工作流
# ==================================================

builder = StateGraph(
    State
)


builder.add_node(
    "chatbot",
    chatbot
)


builder.add_node(
    "tools",
    tool_node
)


builder.add_node(
    "summarize_conversation",
    summarize_conversation
)


builder.add_edge(
    START,
    "chatbot"
)


builder.add_conditional_edges(
    "chatbot",
    route_after_chatbot
)


builder.add_edge(
    "tools",
    "chatbot"
)


builder.add_edge(
    "summarize_conversation",
    END
)


# ==================================================
# 10. SQLite 持久化
# ==================================================

script_directory = (
    Path(__file__).resolve().parent
)


database_path = (
    script_directory.parent
    / "database"
    / "day3_tool_agent_summary.db"
)


connection = sqlite3.connect(
    database_path,
    check_same_thread=False
)


checkpointer = SqliteSaver(
    connection
)


graph = builder.compile(
    checkpointer=checkpointer
)


# ==================================================
# 11. 会话配置
# ==================================================

config = {
    "configurable": {
        "thread_id": "xiaolin-day3-summary"
    },
    "recursion_limit": 10
}


# ==================================================
# 12. 清理终端输入
# ==================================================

def clean_terminal_input(
    raw_text: str
) -> str:
    """删除不可打印控制字符和可见的 ^?。"""

    cleaned_text = "".join(
        character
        for character in raw_text
        if character.isprintable()
    )

    cleaned_text = cleaned_text.replace(
        "^?",
        ""
    )

    return cleaned_text.strip()


# ==================================================
# 13. 连续聊天
# ==================================================

print(
    "工具记忆助手已启动，"
    "输入 exit 结束。"
)

print(
    "当前会话：xiaolin-day3-summary"
)


try:
    while True:
        raw_input = input("\n你：")

        user_input = clean_terminal_input(
            raw_input
        )

        if not user_input:
            print(
                "请输入内容，"
                "不要只按回车。"
            )

            continue

        if user_input.lower() == "exit":
            break

        print(
            "助手：",
            end="",
            flush=True
        )

        received_content = False

        try:
            # stream_mode="messages"：
            # 每当模型生成新的消息片段时，
            # 返回 message_chunk 和 metadata。
            message_stream = graph.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=config,
                stream_mode="messages"
            )

            for (
                message_chunk,
                metadata
            ) in message_stream:

                # chatbot 和 summarize_conversation
                # 都会调用模型。
                #
                # 这里只显示 chatbot 生成的正文，
                #避免把内部摘要打印给用户。
                node_name = metadata.get(
                    "langgraph_node"
                )

                if node_name != "chatbot":
                    continue

                content = message_chunk.content

                # 工具调用阶段的 content 通常为空；
                # 最终回答阶段才会产生正文。
                if isinstance(content, str):
                    if content:
                        print(
                            content,
                            end="",
                            flush=True
                        )

                        received_content = True

            # 流式回答结束后换行
            print()

            if not received_content:
                print(
                    "[本轮没有产生可显示的回答正文]"
                )

        except GraphRecursionError:
            print()

            print(
                "本轮执行步骤过多，"
                "程序已停止本轮执行。"
            )


    snapshot = graph.get_state(
        config
    )

    saved_messages = (
        snapshot.values.get(
            "messages",
            []
        )
    )

    saved_summary = (
        snapshot.values.get(
            "summary",
            "暂无摘要"
        )
    )

    print("\n聊天结束")

    print(
        "当前保留的原始消息：",
        len(saved_messages),
        "条"
    )

    print(
        "对话摘要：",
        saved_summary
    )


finally:
    connection.close()
