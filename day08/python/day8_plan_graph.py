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


# ============================================================
# 6. 创建图
# ============================================================

builder = StateGraph(PlanState)

builder.add_node("planner", planner_node)
builder.add_node("select_step", select_step_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "select_step")
builder.add_edge("select_step", END)

graph = builder.compile()


# ============================================================
# 7. 接收输入并执行图
# ============================================================

user_goal = input("请输入一个需要分步骤完成的复杂任务：").strip()

if not user_goal:
    raise ValueError("任务不能为空。")

initial_state: PlanState = {
    "user_goal": user_goal,
    "plan_goal": "",
    "steps": [],
    "current_step": None
}

final_state = graph.invoke(initial_state)


# ============================================================
# 8. 查看最终状态
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
#      ↓ 查找第一条 status == "pending"
# current_step: dict | None
#      ↓
# END

