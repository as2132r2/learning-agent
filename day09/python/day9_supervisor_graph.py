import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# ============================================================
# 1. 创建基础模型
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 定义主管必须返回的数据结构
# ============================================================

class DelegationDecision(BaseModel):
    """主管为当前请求生成的一次任务委派决定。"""

    worker: Literal[
        "researcher",
        "calculator",
        "writer"
    ] = Field(
        description="最适合执行当前任务的专业Agent"
    )

    task: str = Field(
        description="交给专业Agent的具体、独立、可执行任务"
    )

    expected_output: str = Field(
        description="专业Agent完成任务后应该返回的结果"
    )

    reason: str = Field(
        description="主管选择该专业Agent的简短原因"
    )


# 普通模型被包装成主管模型。
# 它的返回值必须是 DelegationDecision，而不是自由文本。
supervisor_llm = llm.with_structured_output(
    DelegationDecision,
    method="function_calling"
)


# ============================================================
# 3. 定义多Agent共享状态
# ============================================================

class MultiAgentState(TypedDict):
    # 用户最初输入的请求。
    user_request: str

    # 主管选择的专业Agent名称。
    selected_worker: str

    # 主管改写后的独立任务。
    delegated_task: str

    # 主管期望专业Agent返回的结果。
    expected_output: str

    # 主管的分派理由。
    delegation_reason: str

    # 专业Agent真正产生的结果。
    worker_result: str


# ============================================================
# 4. supervisor节点：选择专业Agent
# ============================================================

def supervisor_node(state: MultiAgentState) -> dict:
    """读取用户请求，选择一个专业Agent并生成独立任务。"""

    print("\n[supervisor 节点]")
    print("用户请求：", state["user_request"])

    decision: DelegationDecision = supervisor_llm.invoke([
        SystemMessage(
            content=(
                "你是多Agent系统的主管，只负责分派任务，不亲自完成任务。"
                "你可以选择三个专业Agent：\n"
                "1. researcher：负责分析、解释知识和研究问题；\n"
                "2. calculator：负责数学运算和数值验证；\n"
                "3. writer：负责整理、改写和生成结构清晰的成品文本。\n"
                "请选择最适合当前请求的一个Agent。"
                "task必须是一条脱离原始对话也能独立理解的完整指令。"
            )
        ),
        HumanMessage(content=state["user_request"])
    ])

    print("选择的Agent：", decision.worker)
    print("委派任务：", decision.task)
    print("预期输出：", decision.expected_output)
    print("分派原因：", decision.reason)

    return {
        "selected_worker": decision.worker,
        "delegated_task": decision.task,
        "expected_output": decision.expected_output,
        "delegation_reason": decision.reason
    }


def choose_worker(
    state: MultiAgentState
) -> Literal["researcher", "calculator", "writer"]:
    """把主管选择转换为LangGraph条件边的路线。"""

    return state["selected_worker"]


# ============================================================
# 5. 三个专业Agent节点
# ============================================================

def researcher_node(state: MultiAgentState) -> dict:
    """负责知识分析和概念解释。"""

    print("\n[researcher 节点]")

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是researcher专业Agent。"
                "你只负责知识分析、概念解释和逻辑研究。"
                "请完成主管分配的任务，并满足预期输出。"
                "如果缺少外部资料，要明确说明，不要伪造来源。"
            )
        ),
        HumanMessage(
            content=(
                f"主管分配的任务：{state['delegated_task']}\n\n"
                f"预期输出：{state['expected_output']}"
            )
        )
    ])

    print("researcher结果：", response.content)
    return {"worker_result": response.content}


def calculator_node(state: MultiAgentState) -> dict:
    """负责数学运算和数值推理。"""

    print("\n[calculator 节点]")

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是calculator专业Agent。"
                "你只负责数学运算、分步计算和数值验证。"
                "请清楚展示必要的计算过程，并给出最终数值。"
            )
        ),
        HumanMessage(
            content=(
                f"主管分配的任务：{state['delegated_task']}\n\n"
                f"预期输出：{state['expected_output']}"
            )
        )
    ])

    print("calculator结果：", response.content)
    return {"worker_result": response.content}


def writer_node(state: MultiAgentState) -> dict:
    """负责整理和生成成品文本。"""

    print("\n[writer 节点]")

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是writer专业Agent。"
                "你只负责组织结构、改写内容和生成清晰成品。"
                "不要添加主管任务中不存在的事实。"
            )
        ),
        HumanMessage(
            content=(
                f"主管分配的任务：{state['delegated_task']}\n\n"
                f"预期输出：{state['expected_output']}"
            )
        )
    ])

    print("writer结果：", response.content)
    return {"worker_result": response.content}


# ============================================================
# 6. 创建主管图
# ============================================================

builder = StateGraph(MultiAgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("calculator", calculator_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    choose_worker,
    {
        "researcher": "researcher",
        "calculator": "calculator",
        "writer": "writer"
    }
)

builder.add_edge("researcher", END)
builder.add_edge("calculator", END)
builder.add_edge("writer", END)

graph = builder.compile()


# ============================================================
# 7. 接收请求并执行图
# ============================================================

user_request = input("请输入一个需要交给专业Agent的任务：").strip()

if not user_request:
    raise ValueError("任务不能为空。")

initial_state: MultiAgentState = {
    "user_request": user_request,
    "selected_worker": "",
    "delegated_task": "",
    "expected_output": "",
    "delegation_reason": "",
    "worker_result": ""
}

final_state = graph.invoke(initial_state)


# ============================================================
# 8. 查看最终状态
# ============================================================

print("\n" + "=" * 60)
print("多Agent工作流执行结束。")
print("\n用户原始请求：")
print(final_state["user_request"])
print("\n主管选择：")
print(final_state["selected_worker"])
print("\n主管委派任务：")
print(final_state["delegated_task"])
print("\n分派原因：")
print(final_state["delegation_reason"])
print("\n专业Agent结果：")
print(final_state["worker_result"])


# ============================================================
# 本步骤的数据流
# ============================================================
#
# user_request
#      ↓
# supervisor_node
#      ↓ DelegationDecision
# selected_worker + delegated_task + expected_output
#      ↓ choose_worker()
#      ├── researcher_node
#      ├── calculator_node
#      └── writer_node
#              ↓
# worker_result
#      ↓
# END
