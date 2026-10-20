import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)
from langchain_core.tools import tool
from langgraph.graph import (
    StateGraph,
    START,
    MessagesState
)
from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)


load_dotenv()


@tool
def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    print(f"[执行 add] {a} + {b}")

    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """计算第一个数字减去第二个数字。"""

    print(f"[执行 subtract] {a} - {b}")

    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(f"[执行 multiply] {a} × {b}")

    return a * b


@tool
def divide(a: float, b: float) -> float:
    """计算第一个数字除以第二个数字。第二个数字不能为零。"""

    print(f"[执行 divide] {a} ÷ {b}")

    if b == 0:
        raise ValueError("除数不能为零")

    return a / b


tools = [
    add,
    subtract,
    multiply,
    divide
]


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


llm_with_tools = llm.bind_tools(tools)


def chatbot(state: MessagesState):
    messages = [
        SystemMessage(
            content=(
                "你是一名计算助手。"
                "进行加减乘除时必须调用工具，"
                "不要自己心算。"
            )
        ),
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ToolNode 自动完成：
# 读取 tool_calls
# 查找对应工具
# 执行工具
# 创建 ToolMessage
tool_node = ToolNode(tools)


builder = StateGraph(MessagesState)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_node(
    "tools",
    tool_node
)

builder.add_edge(
    START,
    "chatbot"
)

# 检查 chatbot 返回的 AIMessage：
#
# 有 tool_calls -> 前往 tools
# 无 tool_calls -> 结束
builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

# 工具执行完成后返回 chatbot
builder.add_edge(
    "tools",
    "chatbot"
)


graph = builder.compile()


user_input = input("请输入一道计算题：").strip()


result = graph.invoke(
    {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }
)


print("\n====================")
print("最终回答：")
print(result["messages"][-1].content)


print("\n====================")
print("完整消息流程：")

for number, message in enumerate(
    result["messages"],
    start=1
):
    print(
        f"\n第 {number} 条：",
        type(message).__name__
    )

    print("content：", repr(message.content))

    if hasattr(message, "tool_calls"):
        if message.tool_calls:
            print(
                "tool_calls：",
                message.tool_calls
            )

    if hasattr(message, "tool_call_id"):
        if message.tool_call_id:
            print(
                "tool_call_id：",
                message.tool_call_id
            )

