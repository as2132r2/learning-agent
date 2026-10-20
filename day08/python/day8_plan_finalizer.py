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
# 2. 规划模型的结构化输出
# ============================================================

class PlanStep(BaseModel):
    """规划模型生成的一条任务步骤。"""

    step_id: int = Field(description="从 1 开始的连续步骤编号")
    description: str = Field(description="需要执行的具体任务")
    expected_output: str = Field(description="这一步预期产生的结果")
    status: Literal["pending", "completed"] = Field(
        default="pending",
        description="新计划中的步骤必须是 pending"
    )


class TaskPlan(BaseModel):
    """规划模型返回的完整计划。"""

    goal: str = Field(description="用户最终目标")
    steps: list[PlanStep] = Field(description="按顺序执行的步骤")


planner_llm = llm.with_structured_output(
    TaskPlan,
    method="function_calling"
)


# ============================================================
# 3. LangGraph 状态
# ============================================================

class PlanState(TypedDict):
    # 用户输入的原始目标。
    user_goal: str

    # 规划器整理后的目标。
    plan_goal: str

    # model_dump() 后的步骤列表。
    # 每个元素都是 dict，而不是 PlanStep 对象。
    steps: list[dict]

    # select_step 节点选出的下一步。
    # 还没有选择时使用 None。
    current_step: dict | None

    # executor 节点执行当前步骤后产生的文本结果。
    current_result: str

    # finalizer 节点根据全部步骤结果整理出的最终回答。
    final_answer: str


# ============================================================
# 4. planner 节点：生成计划
# ============================================================

def planner_node(state: PlanState) -> dict:
    """读取 user_goal，生成结构化计划并写入图状态。"""

    print("\n[planner 节点]")
    print("收到用户目标：", state["user_goal"])

    plan: TaskPlan = planner_llm.invoke([
        SystemMessage(
            content=(
                "你是一名任务规划器，只制定计划，不执行任务。"
                "请把用户目标拆成 3 到 6 个有先后顺序、可以独立执行的步骤。"
                "后面的步骤可以使用前面步骤的输出。"
                "所有步骤的初始状态必须是 pending。"
            )
        ),
        HumanMessage(content=state["user_goal"])
    ])

    # 把 list[PlanStep] 转换成 list[dict]。
    # 普通字典更方便后续节点修改 status、添加执行结果并保存到数据库。
    step_dictionaries = [
        step.model_dump()
        for step in plan.steps
    ]

    print("规划后的目标：", plan.goal)
    print("生成步骤数量：", len(step_dictionaries))

    for step in step_dictionaries:
        print(
            f"{step['step_id']}. "
            f"{step['description']} "
            f"[{step['status']}]"
        )

    return {
        "plan_goal": plan.goal,
        "steps": step_dictionaries
    }


# ============================================================
# 5. select_step 节点：寻找下一条待执行步骤
# ============================================================

def select_step_node(state: PlanState) -> dict:
    """按照列表顺序选择第一条 pending 步骤。"""

    print("\n[select_step 节点]")

    next_step = next(
        (
            step
            for step in state["steps"]
            if step["status"] == "pending"
        ),
        None
    )

    if next_step is None:
        print("没有待执行步骤。")
    else:
        print("选中的步骤编号：", next_step["step_id"])
        print("任务说明：", next_step["description"])
        print("预期输出：", next_step["expected_output"])

    return {
        "current_step": next_step
    }


def choose_step_route(
    state: PlanState
) -> Literal["execute", "finish"]:
    """有待执行步骤就继续，否则结束工作流。"""

    if state["current_step"] is None:
        return "finish"

    return "execute"


# ============================================================
# 6. executor 节点：执行当前步骤并更新计划
# ============================================================

def executor_node(state: PlanState) -> dict:
    """执行 current_step，并把对应步骤标记为 completed。"""

    print("\n[executor 节点]")

    current_step = state["current_step"]

    if current_step is None:
        print("没有待执行步骤，executor 不需要工作。")
        return {
            "current_result": "没有待执行步骤。"
        }

    print("准备执行步骤：", current_step["step_id"])
    print("任务说明：", current_step["description"])

    # 收集之前已经完成的步骤结果。
    # 后续步骤可以使用前面步骤产生的内容，而不是彼此完全独立。
    completed_parts: list[str] = []

    for step in state["steps"]:
        if step["status"] == "completed" and "result" in step:
            completed_parts.append(
                f"步骤 {step['step_id']}：{step['description']}\n"
                f"执行结果：{step['result']}"
            )

    completed_context = (
        "\n\n".join(completed_parts)
        if completed_parts
        else "暂无已完成步骤。"
    )

    print("已完成步骤数量：", len(completed_parts))

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是一名任务执行器。"
                "请只执行当前分配给你的这一个步骤，不要执行其他步骤。"
                "输出应该具体、可直接交给后续步骤使用。"
                "如果当前步骤是制定学习内容，请给出明确内容；"
                "如果是设计方案，请给出具体方案。"
            )
        ),
        HumanMessage(
            content=(
                f"最终目标：{state['plan_goal']}\n\n"
                f"之前步骤的执行结果：\n{completed_context}\n\n"
                f"当前步骤编号：{current_step['step_id']}\n"
                f"当前任务：{current_step['description']}\n"
                f"预期输出：{current_step['expected_output']}"
            )
        )
    ])

    execution_result = response.content

    # 创建一个新的步骤列表，而不是直接修改 state["steps"]。
    # 这样每个节点返回的状态变化更加明确，也更适合检查点保存。
    updated_steps: list[dict] = []
    completed_step: dict | None = None

    for step in state["steps"]:
        updated_step = step.copy()

        if step["step_id"] == current_step["step_id"]:
            updated_step["status"] = "completed"
            updated_step["result"] = execution_result
            completed_step = updated_step

        updated_steps.append(updated_step)

    print("步骤执行结果：")
    print(execution_result)
    print("步骤状态已更新为：completed")

    return {
        "steps": updated_steps,
        "current_step": completed_step,
        "current_result": execution_result
    }


# ============================================================
# 7. finalizer 节点：汇总所有步骤
# ============================================================

def finalizer_node(state: PlanState) -> dict:
    """把全部已完成步骤整理成面向用户的最终交付结果。"""

    print("\n[finalizer 节点]")
    print("所有计划步骤已经完成，开始整理最终答案。")

    result_parts: list[str] = []

    for step in state["steps"]:
        result_parts.append(
            f"步骤 {step['step_id']}：{step['description']}\n"
            f"预期输出：{step['expected_output']}\n"
            f"实际结果：{step.get('result', '没有执行结果')}"
        )

    all_results = "\n\n".join(result_parts)

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是任务结果汇总器。"
                "请根据已经完成的所有步骤，整理一份可以直接交付给用户的最终回答。"
                "不要重新制定计划，不要声称执行了没有出现在步骤结果里的操作。"
                "保留重要细节，删除重复内容，并使用清晰的标题和列表。"
            )
        ),
        HumanMessage(
            content=(
                f"用户原始目标：{state['user_goal']}\n\n"
                f"规划后的目标：{state['plan_goal']}\n\n"
                f"全部步骤结果：\n{all_results}"
            )
        )
    ])

    print("最终答案已经生成。")

    return {
        "final_answer": response.content
    }


# ============================================================
# 8. 创建图
# ============================================================

builder = StateGraph(PlanState)

builder.add_node("planner", planner_node)
builder.add_node("select_step", select_step_node)
builder.add_node("executor", executor_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "select_step")

builder.add_conditional_edges(
    "select_step",
    choose_step_route,
    {
        "execute": "executor",
        "finish": "finalizer"
    }
)

# 每执行完一步，都回去重新选择第一条 pending 步骤。
builder.add_edge("executor", "select_step")
builder.add_edge("finalizer", END)

graph = builder.compile()


# ============================================================
# 9. 接收输入并执行图
# ============================================================

user_goal = input("请输入一个需要分步骤完成的复杂任务：").strip()

if not user_goal:
    raise ValueError("任务不能为空。")

initial_state: PlanState = {
    "user_goal": user_goal,
    "plan_goal": "",
    "steps": [],
    "current_step": None,
    "current_result": "",
    "final_answer": ""
}

final_state = graph.invoke(initial_state)


# ============================================================
# 10. 查看最终状态
# ============================================================

print("\n" + "=" * 60)
print("图执行完成后的状态：")

print("\n用户原始目标：")
print(final_state["user_goal"])

print("\n规划后的目标：")
print(final_state["plan_goal"])

print("\n完整步骤列表：")

for step in final_state["steps"]:
    print(
        f"{step['step_id']}. "
        f"{step['description']} "
        f"[{step['status']}]"
    )

print("\n当前准备执行的步骤：")
print(final_state["current_step"])

print("\ncurrent_step 的数据类型：")
print(type(final_state["current_step"]))

print("\n当前步骤的执行结果：")
print(final_state["current_result"])

print("\n执行后的步骤状态：")

for step in final_state["steps"]:
    print(
        f"步骤 {step['step_id']}："
        f"{step['status']}"
    )

    if "result" in step:
        print("保存的执行结果：", step["result"])

completed_count = sum(
    1
    for step in final_state["steps"]
    if step["status"] == "completed"
)

print("\n完成步骤数量：")
print(f"{completed_count}/{len(final_state['steps'])}")

print("\n最终 current_step：")
print(final_state["current_step"])

print("\n" + "=" * 60)
print("最终交付结果：")
print(final_state["final_answer"])


# ============================================================
# 本步骤的数据流
# ============================================================
#
# user_goal: str
#      ↓
# planner_node
#      ↓ planner_llm.invoke()
# TaskPlan
#      ↓ model_dump()
# steps: list[dict]
#      ↓
# select_step_node
#      ├── 找到 pending → current_step: dict
#      │                       ↓
# executor_node
#      │  ↓ llm.invoke() 执行当前步骤
# current_result: str
#      │  ↓ 更新对应步骤
# status: pending → completed
# result: str
#      │
#      └──────────────→ select_step_node
#
# select_step_node
#      ↓ 找不到 pending
# current_step: None
#      ↓
# finalizer_node
#      ↓ 读取所有 step["result"]
# final_answer: str
#      ↓
# END
