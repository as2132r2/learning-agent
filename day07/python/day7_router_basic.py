import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# ============================================================
# 1. 加载环境变量并创建聊天模型
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 定义模型必须返回的路由结构
# ============================================================

class RouteDecision(BaseModel):
    """模型对用户问题做出的路由决定。"""

    route: Literal[
        "knowledge",
        "calculate",
        "chat",
        "unsupported"
    ] = Field(
        description=(
            "knowledge=查询课程知识库；"
            "calculate=数学计算；"
            "chat=普通聊天；"
            "unsupported=当前助手不应处理的请求"
        )
    )

    reason: str = Field(
        description="选择这条路线的简短原因"
    )


# structured_llm 的输入仍然是消息列表，
# 但输出不再是 AIMessage，而是 RouteDecision 对象。
#
# method="function_calling" 很重要：
# 当前使用的 DeepSeek 兼容接口不支持 LangChain 默认采用的
# response_format/json_schema 模式，但支持工具调用模式。
structured_llm = llm.with_structured_output(
    RouteDecision,
    method="function_calling"
)


# ============================================================
# 3. 定义 LangGraph 状态
# ============================================================

class RouterState(TypedDict):
    # 用户原始问题。
    question: str

    # router 节点写入的路线。
    route: str

    # router 节点写入的判断原因。
    reason: str

    # 最终分支节点写入的演示结果。
    result: str


# ============================================================
# 4. 路由节点：调用模型进行分类
# ============================================================

def router_node(state: RouterState) -> dict:
    """
    参数 state：当前完整图状态。
    返回 dict：只更新 route 和 reason 两个字段。
    """

    question = state["question"]

    decision: RouteDecision = structured_llm.invoke([
        SystemMessage(
            content=(
                "你是一个问题路由器，只负责分类，不回答问题。\n"
                "请按照以下规则选择路线：\n"
                "1. LangChain、LangGraph、RAG、Python、C语言等"
                "学习资料问题 → knowledge。\n"
                "2. 加减乘除或明确数学表达式 → calculate。\n"
                "3. 问候、闲聊、自我介绍 → chat。\n"
                "4. 与当前学习助手能力无关，或不适合处理的请求"
                " → unsupported。"
            )
        ),
        HumanMessage(content=question)
    ])

    print("\n[router 节点]")
    print("模型返回对象：", decision)
    print("对象类型：", type(decision))
    print("选择路线：", decision.route)
    print("判断原因：", decision.reason)

    return {
        "route": decision.route,
        "reason": decision.reason
    }


# ============================================================
# 5. 四个分支节点
# ============================================================

# 这一步只演示路由，所以分支节点暂时不执行真正业务。
# 后续步骤再分别接入 Retriever、计算工具和聊天模型。

def knowledge_node(state: RouterState) -> dict:
    print("\n[knowledge 节点]")
    return {
        "result": "问题已进入知识库检索分支。"
    }


def calculate_node(state: RouterState) -> dict:
    print("\n[calculate 节点]")
    return {
        "result": "问题已进入数学计算分支。"
    }


def chat_node(state: RouterState) -> dict:
    print("\n[chat 节点]")
    return {
        "result": "问题已进入普通聊天分支。"
    }


def unsupported_node(state: RouterState) -> dict:
    print("\n[unsupported 节点]")
    return {
        "result": "当前助手不处理这类请求。"
    }


# ============================================================
# 6. 条件边函数
# ============================================================

def choose_route(
    state: RouterState
) -> Literal[
    "knowledge",
    "calculate",
    "chat",
    "unsupported"
]:
    """
    LangGraph 在 router_node 执行完后调用这个函数。

    返回值不是最终回答，而是条件边使用的路线标签。
    """

    return state["route"]


# ============================================================
# 7. 创建 LangGraph
# ============================================================

builder = StateGraph(RouterState)

builder.add_node("router", router_node)
builder.add_node("knowledge", knowledge_node)
builder.add_node("calculate", calculate_node)
builder.add_node("chat", chat_node)
builder.add_node("unsupported", unsupported_node)

# 所有问题先进入 router。
builder.add_edge(START, "router")

# router 执行后，由 choose_route 读取 state["route"]。
builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "knowledge": "knowledge",
        "calculate": "calculate",
        "chat": "chat",
        "unsupported": "unsupported"
    }
)

# 四个分支完成后都结束本次工作流。
builder.add_edge("knowledge", END)
builder.add_edge("calculate", END)
builder.add_edge("chat", END)
builder.add_edge("unsupported", END)

graph = builder.compile()


# ============================================================
# 8. 接收问题并运行工作流
# ============================================================

question = input("请输入问题：").strip()

if not question:
    print("问题不能为空。")
    raise SystemExit(1)

initial_state: RouterState = {
    "question": question,
    "route": "",
    "reason": "",
    "result": ""
}

final_state = graph.invoke(initial_state)


# ============================================================
# 9. 显示最终状态
# ============================================================

print("\n" + "=" * 60)
print("最终状态：")
print("question：", final_state["question"])
print("route：", final_state["route"])
print("reason：", final_state["reason"])
print("result：", final_state["result"])

print("\n完整状态对象：")
print(final_state)


# ============================================================
# 数据流
# ============================================================
#
# 用户问题 question
#     ↓
# router_node(state)
#     ↓ structured_llm.invoke(messages)
# RouteDecision(route, reason)
#     ↓ 写回 RouterState
# choose_route(state)
#     ├── knowledge   → knowledge_node
#     ├── calculate   → calculate_node
#     ├── chat        → chat_node
#     └── unsupported → unsupported_node
#                          ↓
#                         END
