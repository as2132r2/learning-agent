import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# ============================================================
# 1. 创建聊天模型
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 定义路由器的结构化输出
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
# 3. 定义计算工具
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

# calculator_llm 可以提出工具调用请求。
calculator_llm = llm.bind_tools(tools)

# ToolNode 负责读取 AIMessage.tool_calls，执行对应工具，
# 再把结果包装成 ToolMessage 写回 messages。
tool_node = ToolNode(tools)


# ============================================================
# 4. 定义图状态
# ============================================================

class AgentState(MessagesState):
    # MessagesState 已经包含：
    # messages: Annotated[list[BaseMessage], add_messages]

    question: str
    route: str
    reason: str
    result: str


# ============================================================
# 5. 第一层：问题路由节点
# ============================================================

def router_node(state: AgentState) -> dict:
    decision: RouteDecision = router_llm.invoke([
        SystemMessage(
            content=(
                "你是问题路由器，只分类，不回答。\n"
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
# 6. 第二层：计算模型节点
# ============================================================

def calculator_node(state: AgentState) -> dict:
    """
    调用绑定了工具的模型。

    模型可能：
    1. 返回 tool_calls，请求执行工具；
    2. 返回普通 content，给出最终答案。
    """

    response = calculator_llm.invoke([
        SystemMessage(
            content=(
                "你是数学计算助手。"
                "计算时必须使用提供的 add、multiply 工具，"
                "不要自己猜测结果。"
                "复杂表达式可以分多轮调用工具。"
            )
        ),
        *state["messages"]
    ])

    print("\n[calculator 节点]")

    if response.tool_calls:
        print("模型请求工具：", response.tool_calls)
        return {
            "messages": [response]
        }

    print("模型生成最终答案：", response.content)
    return {
        "messages": [response],
        "result": response.content
    }


def choose_calculator_route(
    state: AgentState
) -> Literal["tools", "done"]:
    """判断计算模型是要执行工具，还是已经生成最终答案。"""

    latest_message = state["messages"][-1]

    if latest_message.tool_calls:
        return "tools"

    return "done"


# ============================================================
# 7. 其他三个分支节点
# ============================================================

def knowledge_node(state: AgentState) -> dict:
    return {
        "result": "问题已进入知识库分支，下一步再接入真实 Retriever。"
    }


def chat_node(state: AgentState) -> dict:
    return {
        "result": "问题已进入普通聊天分支。"
    }


def unsupported_node(state: AgentState) -> dict:
    return {
        "result": "当前助手不处理这类请求。"
    }


# ============================================================
# 8. 创建 LangGraph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("calculator", calculator_node)
builder.add_node("tools", tool_node)
builder.add_node("knowledge", knowledge_node)
builder.add_node("chat", chat_node)
builder.add_node("unsupported", unsupported_node)

builder.add_edge(START, "router")

# 第一层条件分支：决定问题属于哪种任务。
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

# 第二层条件分支：计算模型决定调用工具还是结束。
builder.add_conditional_edges(
    "calculator",
    choose_calculator_route,
    {
        "tools": "tools",
        "done": END
    }
)

# 工具执行完成后回到 calculator，让模型读取 ToolMessage。
builder.add_edge("tools", "calculator")

builder.add_edge("knowledge", END)
builder.add_edge("chat", END)
builder.add_edge("unsupported", END)

graph = builder.compile()


# ============================================================
# 9. 运行程序
# ============================================================

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    raise SystemExit(1)

initial_state: AgentState = {
    "question": question,
    "route": "",
    "reason": "",
    "result": "",
    "messages": [HumanMessage(content=question)]
}

final_state = graph.invoke(initial_state)


# ============================================================
# 10. 显示结果和消息流
# ============================================================

print("\n" + "=" * 60)
print("最终路线：", final_state["route"])
print("路由原因：", final_state["reason"])
print("最终结果：", final_state["result"])

print("\n完整消息流：")

for index, message in enumerate(final_state["messages"], start=1):
    print(f"\n第 {index} 条：{type(message).__name__}")
    print("content：", repr(message.content))

    if getattr(message, "tool_calls", None):
        print("tool_calls：", message.tool_calls)

    if hasattr(message, "tool_call_id"):
        print("tool_call_id：", message.tool_call_id)


# ============================================================
# 计算分支的数据流
# ============================================================
#
# router
#   ↓ route=calculate
# calculator
#   ↓ AIMessage(tool_calls=[add])
# tools
#   ↓ ToolMessage(result)
# calculator
#   ↓ 可能继续请求 multiply
# tools
#   ↓ ToolMessage(result)
# calculator
#   ↓ AIMessage(content=最终答案)
# END

