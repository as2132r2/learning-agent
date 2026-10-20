import argparse
import os
import sqlite3
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
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


class StepEvaluation(BaseModel):
    """质量检查模型对一条步骤执行结果的判断。"""

    passed: bool = Field(
        description="执行结果是否完成了当前任务并满足预期输出"
    )

    reason: str = Field(
        description="通过或不通过的具体原因"
    )

    improvement: str = Field(
        description="如果不通过，下一次执行应该如何改进；通过时填写无需改进"
    )


step_grader_llm = llm.with_structured_output(
    StepEvaluation,
    method="function_calling"
)


class RevisedStep(BaseModel):
    """重规划器提出的新步骤，身份和顺序由程序分配。"""

    description: str = Field(description="新的具体任务")
    expected_output: str = Field(description="新步骤的预期输出")


class RevisedPlan(BaseModel):
    """失败后生成的替代执行路线。"""

    reason: str = Field(description="为什么这样调整剩余计划")
    steps: list[RevisedStep] = Field(
        min_length=1,
        max_length=4,
        description="1到4条按执行顺序排列的新步骤"
    )


replanner_llm = llm.with_structured_output(
    RevisedPlan,
    method="function_calling"
)


# 每条步骤最多执行两次，避免 executor 和 grade_step 无限循环。
MAX_STEP_ATTEMPTS = 2

# 整个任务最多重规划一次，避免计划反复膨胀。
MAX_REPLAN_ATTEMPTS = 1


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

    # grade_step 节点的路由判断：approved、retry、replan 或 failed。
    step_decision: str

    # 最近一次步骤质量检查的原因和改进建议。
    step_grade_reason: str
    step_improvement: str

    # 已经执行过的动态重规划次数，以及最近一次重规划原因。
    replan_count: int
    replan_reason: str

    # 当前生效的计划版本，以及下一个全局步骤编号。
    current_plan_version: int
    next_step_number: int

    # 人工是否批准执行规划器生成的计划。
    plan_approved: bool


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
    step_dictionaries: list[dict] = []

    for order, step in enumerate(plan.steps, start=1):
        step_dictionaries.append({
            "step_id": f"step-{order:03d}",
            "plan_version": 1,
            "order": order,
            **step.model_dump(),
            "attempts": 0,
            "created_by": "planner"
        })

    print("规划后的目标：", plan.goal)
    print("生成步骤数量：", len(step_dictionaries))

    for step in step_dictionaries:
        print(
            f"版本{step['plan_version']} / 顺序{step['order']} / "
            f"ID={step['step_id']}："
            f"{step['description']} "
            f"[{step['status']}]"
        )

    return {
        "plan_goal": plan.goal,
        "steps": step_dictionaries,
        "current_plan_version": 1,
        "next_step_number": len(step_dictionaries) + 1
    }


# ============================================================
# 5. review_plan 节点：暂停并等待人工审批
# ============================================================

def review_plan_node(state: PlanState) -> dict:
    """把计划暴露给图外用户，并暂停工作流。"""

    print("\n[review_plan 节点]")
    print("计划已经生成，准备请求人工审批。")

    decision = interrupt({
        "question": "是否批准执行这份任务计划？",
        "plan_goal": state["plan_goal"],
        "steps": [
            {
                "step_id": step["step_id"],
                "plan_version": step["plan_version"],
                "order": step["order"],
                "description": step["description"],
                "expected_output": step["expected_output"]
            }
            for step in state["steps"]
        ]
    })

    approved = str(decision).strip().lower() in {
        "yes",
        "y"
    }

    print("[review_plan 节点恢复]")
    print("收到人工决定：", decision)

    return {
        "plan_approved": approved
    }


def choose_approval_route(
    state: PlanState
) -> Literal["approved", "rejected"]:
    return "approved" if state["plan_approved"] else "rejected"


def reject_plan_node(state: PlanState) -> dict:
    """用户拒绝计划后取消待办步骤，不调用执行模型。"""

    print("\n[reject_plan 节点]")
    print("用户拒绝计划，不执行任何步骤。")

    cancelled_steps: list[dict] = []

    for step in state["steps"]:
        cancelled_step = step.copy()

        if step["status"] == "pending":
            cancelled_step["status"] = "cancelled"
            cancelled_step["cancel_reason"] = "用户拒绝执行计划。"

        cancelled_steps.append(cancelled_step)

    return {
        "steps": cancelled_steps,
        "current_step": None,
        "final_answer": "用户拒绝了任务计划，没有执行任何计划步骤。"
    }


# ============================================================
# 6. select_step 节点：寻找下一条待执行步骤
# ============================================================

def select_step_node(state: PlanState) -> dict:
    """按照列表顺序选择第一条 pending 步骤。"""

    print("\n[select_step 节点]")

    pending_steps = [
        step
        for step in state["steps"]
        if (
            step["plan_version"] == state["current_plan_version"]
            and step["status"] == "pending"
        )
    ]

    next_step = (
        min(pending_steps, key=lambda step: step["order"])
        if pending_steps
        else None
    )

    if next_step is None:
        print("没有待执行步骤。")
    else:
        print("当前计划版本：", next_step["plan_version"])
        print("执行顺序：", next_step["order"])
        print("步骤唯一ID：", next_step["step_id"])
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
    """执行 current_step，产生等待质量检查的候选结果。"""

    print("\n[executor 节点]")

    current_step = state["current_step"]

    if current_step is None:
        print("没有待执行步骤，executor 不需要工作。")
        return {
            "current_result": "没有待执行步骤。"
        }

    print("当前计划版本：", current_step["plan_version"])
    print("执行顺序：", current_step["order"])
    print("步骤唯一ID：", current_step["step_id"])
    print("任务说明：", current_step["description"])

    current_attempt = current_step.get("attempts", 0) + 1
    print(
        "本步骤执行次数：",
        current_attempt,
        "/",
        MAX_STEP_ATTEMPTS
    )

    # 收集之前已经完成的步骤结果。
    # 后续步骤可以使用前面步骤产生的内容，而不是彼此完全独立。
    completed_parts: list[str] = []

    for step in state["steps"]:
        if step["status"] == "completed" and "result" in step:
            completed_parts.append(
                f"步骤ID {step['step_id']}：{step['description']}\n"
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
                f"当前计划版本：{current_step['plan_version']}\n"
                f"当前执行顺序：{current_step['order']}\n"
                f"当前步骤ID：{current_step['step_id']}\n"
                f"当前任务：{current_step['description']}\n"
                f"预期输出：{current_step['expected_output']}"
                f"\n本次是第 {current_attempt} 次执行。"
                f"\n上一次检查意见：{state['step_improvement'] or '无'}"
            )
        )
    ])

    execution_result = response.content

    # 创建一个新的步骤列表，而不是直接修改 state["steps"]。
    # 这样每个节点返回的状态变化更加明确，也更适合检查点保存。
    updated_steps: list[dict] = []
    executed_step: dict | None = None

    for step in state["steps"]:
        updated_step = step.copy()

        if step["step_id"] == current_step["step_id"]:
            # executor 只能写入候选结果，无权宣布 completed。
            updated_step["status"] = "in_progress"
            updated_step["result"] = execution_result
            updated_step["attempts"] = current_attempt
            executed_step = updated_step

        updated_steps.append(updated_step)

    print("步骤执行结果：")
    print(execution_result)
    print("步骤状态已更新为：in_progress，等待质量检查")

    return {
        "steps": updated_steps,
        "current_step": executed_step,
        "current_result": execution_result
    }


# ============================================================
# 7. grade_step 节点：检查当前步骤结果
# ============================================================

def grade_step_node(state: PlanState) -> dict:
    """判断当前执行结果是否满足计划中的预期输出。"""

    print("\n[grade_step 节点]")

    current_step = state["current_step"]

    if current_step is None:
        raise ValueError("grade_step 没有收到 current_step。")

    evaluation: StepEvaluation = step_grader_llm.invoke([
        SystemMessage(
            content=(
                "你是严格的任务步骤质量检查器，只负责检查，不重新执行任务。"
                "如果结果具体完成了当前任务并满足预期输出，passed 才能为 true。"
                "不要因为文字较长就自动通过，也不要要求当前步骤完成后续步骤。"
            )
        ),
        HumanMessage(
            content=(
                f"最终目标：{state['plan_goal']}\n\n"
                f"当前任务：{current_step['description']}\n"
                f"预期输出：{current_step['expected_output']}\n\n"
                f"实际执行结果：\n{state['current_result']}"
            )
        )
    ])

    attempts = current_step["attempts"]

    if evaluation.passed:
        decision = "approved"
        new_status = "completed"
    elif attempts < MAX_STEP_ATTEMPTS:
        decision = "retry"
        new_status = "pending"
    elif state["replan_count"] < MAX_REPLAN_ATTEMPTS:
        decision = "replan"
        new_status = "failed"
    else:
        decision = "failed"
        new_status = "failed"

    updated_steps: list[dict] = []
    graded_step: dict | None = None

    for step in state["steps"]:
        updated_step = step.copy()

        if step["step_id"] == current_step["step_id"]:
            updated_step["status"] = new_status
            updated_step["grade_reason"] = evaluation.reason
            graded_step = updated_step

        updated_steps.append(updated_step)

    print("是否通过：", evaluation.passed)
    print("检查原因：", evaluation.reason)
    print("改进建议：", evaluation.improvement)
    print("后续决定：", decision)

    return {
        "steps": updated_steps,
        "current_step": graded_step,
        "step_decision": decision,
        "step_grade_reason": evaluation.reason,
        "step_improvement": evaluation.improvement
    }


def choose_grade_route(
    state: PlanState
) -> Literal["approved", "retry", "replan", "failed"]:
    return state["step_decision"]


# ============================================================
# 8. replanner 节点：失败后修改剩余计划
# ============================================================

def replanner_node(state: PlanState) -> dict:
    """根据已完成结果和失败原因，用新步骤替换旧的 pending 步骤。"""

    print("\n[replanner 节点]")

    history_parts: list[str] = []

    for step in state["steps"]:
        history_parts.append(
            f"步骤ID：{step['step_id']}\n"
            f"计划版本：{step['plan_version']}\n"
            f"版本内顺序：{step['order']}\n"
            f"任务：{step['description']}\n"
            f"状态：{step['status']}\n"
            f"结果：{step.get('result', '无')}\n"
            f"检查原因：{step.get('grade_reason', '无')}"
        )

    plan_history = "\n\n".join(history_parts)

    revised_plan: RevisedPlan = replanner_llm.invoke([
        SystemMessage(
            content=(
                "你是动态任务重规划器。"
                "某个步骤已经连续失败，请根据现有结果设计替代路线。"
                "只规划尚未完成的部分，不要重复已经成功完成的步骤。"
                "请生成1到4条具体、可独立执行的新步骤。"
                "新路线应该尽可能绕过失败原因，同时仍服务于最终目标。"
            )
        ),
        HumanMessage(
            content=(
                f"用户目标：{state['user_goal']}\n\n"
                f"当前计划执行历史：\n{plan_history}\n\n"
                f"最近失败原因：{state['step_grade_reason']}\n"
                f"检查器改进建议：{state['step_improvement']}"
            )
        )
    ])

    # 旧计划中的 pending 步骤已经不再适用，但保留并标记 cancelled，
    # 方便以后审计“原来计划过什么、为什么没有继续执行”。
    updated_steps: list[dict] = []

    for step in state["steps"]:
        updated_step = step.copy()

        if step["status"] == "pending":
            updated_step["status"] = "cancelled"
            updated_step["cancel_reason"] = "动态重规划后被新步骤替代。"

        updated_steps.append(updated_step)

    new_plan_version = state["current_plan_version"] + 1
    next_step_number = state["next_step_number"]
    new_steps: list[dict] = []

    for order, step in enumerate(revised_plan.steps, start=1):
        new_steps.append({
            "step_id": f"step-{next_step_number:03d}",
            "plan_version": new_plan_version,
            "order": order,
            "description": step.description,
            "expected_output": step.expected_output,
            "status": "pending",
            "attempts": 0,
            "created_by": "replanner"
        })
        next_step_number += 1

    print("重规划原因：", revised_plan.reason)
    print("取消的旧待办步骤数量：", sum(
        1
        for step in updated_steps
        if step["status"] == "cancelled"
    ))
    print("新增步骤数量：", len(new_steps))

    for step in new_steps:
        print(
            f"版本{step['plan_version']} / 顺序{step['order']} / "
            f"ID={step['step_id']}："
            f"{step['description']}"
        )

    return {
        "steps": [*updated_steps, *new_steps],
        "current_step": None,
        "replan_count": state["replan_count"] + 1,
        "replan_reason": revised_plan.reason,
        "current_plan_version": new_plan_version,
        "next_step_number": next_step_number,
        "step_decision": ""
    }


# ============================================================
# 9. finalizer 节点：汇总所有步骤
# ============================================================

def finalizer_node(state: PlanState) -> dict:
    """把全部已处理步骤整理成面向用户的最终交付结果。"""

    print("\n[finalizer 节点]")
    print("所有计划步骤已经处理，开始整理最终答案。")

    result_parts: list[str] = []

    for step in state["steps"]:
        result_parts.append(
            f"步骤ID：{step['step_id']}\n"
            f"计划版本：{step['plan_version']}\n"
            f"版本内顺序：{step['order']}\n"
            f"任务：{step['description']}\n"
            f"最终状态：{step['status']}\n"
            f"预期输出：{step['expected_output']}\n"
            f"实际结果：{step.get('result', '没有执行结果')}"
        )

    all_results = "\n\n".join(result_parts)

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是任务结果汇总器。"
                "请根据所有已处理步骤，整理一份可以直接交付给用户的最终回答。"
                "如果存在 failed 步骤，必须明确说明失败内容和限制，不能假装成功。"
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
# 10. 创建图
# ============================================================

builder = StateGraph(PlanState)

builder.add_node("planner", planner_node)
builder.add_node("review_plan", review_plan_node)
builder.add_node("reject_plan", reject_plan_node)
builder.add_node("select_step", select_step_node)
builder.add_node("executor", executor_node)
builder.add_node("grade_step", grade_step_node)
builder.add_node("replanner", replanner_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "review_plan")

builder.add_conditional_edges(
    "review_plan",
    choose_approval_route,
    {
        "approved": "select_step",
        "rejected": "reject_plan"
    }
)

builder.add_edge("reject_plan", END)

builder.add_conditional_edges(
    "select_step",
    choose_step_route,
    {
        "execute": "executor",
        "finish": "finalizer"
    }
)

# executor 只生成候选结果，必须先经过质量检查。
builder.add_edge("executor", "grade_step")

builder.add_conditional_edges(
    "grade_step",
    choose_grade_route,
    {
        "approved": "select_step",
        "retry": "executor",
        "replan": "replanner",
        # 已经用过重规划机会，再失败时继续处理其他步骤。
        "failed": "select_step"
    }
)

builder.add_edge("replanner", "select_step")
builder.add_edge("finalizer", END)

# interrupt() 必须配合 checkpointer，才能保存暂停位置并恢复。
script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent
database_path = (
    project_directory
    / "day08"
    / "database"
    / "day8_plan_approval.db"
)

database_path.parent.mkdir(parents=True, exist_ok=True)

checkpoint_connection = sqlite3.connect(
    database_path,
    check_same_thread=False
)

checkpointer = SqliteSaver(checkpoint_connection)
graph = builder.compile(checkpointer=checkpointer)


# ============================================================
# 11. 跨进程启动、恢复和结果显示
# ============================================================

def make_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 80
    }


def show_approval_request(
    approval_request: dict,
    thread_id: str
) -> None:
    print("\n" + "=" * 60)
    print("工作流已经暂停并保存。")
    print("流程编号：", thread_id)
    print("规划目标：", approval_request["plan_goal"])

    print("\n待审批计划：")
    for step in approval_request["steps"]:
        print(
            f"{step['order']}. {step['description']}\n"
            f"   预期输出：{step['expected_output']}\n"
            f"   唯一ID：{step['step_id']}"
        )

    print("\n现在可以关闭程序。")
    print("稍后批准：")
    print(
        "python day08/python/day8_plan_approval_persistent.py "
        f"resume {thread_id} yes"
    )
    print("\n稍后拒绝：")
    print(
        "python day08/python/day8_plan_approval_persistent.py "
        f"resume {thread_id} no"
    )


def show_final_state(final_state: PlanState) -> None:
    print("\n" + "=" * 60)
    print("工作流执行结束。")
    print("计划是否批准：", final_state["plan_approved"])
    print("最终计划版本：", final_state["current_plan_version"])
    print("动态重规划次数：", final_state["replan_count"])

    print("\n步骤状态：")
    for step in final_state["steps"]:
        print(
            f"版本{step['plan_version']} / "
            f"顺序{step['order']} / "
            f"ID={step['step_id']} / "
            f"状态={step['status']} / "
            f"执行次数={step.get('attempts', 0)}"
        )

    print("\n最终交付结果：")
    print(final_state["final_answer"])


def start_workflow(thread_id: str) -> None:
    config = make_config(thread_id)
    snapshot = graph.get_state(config)

    if snapshot.values:
        print("该流程编号已经存在，请更换 thread_id。")
        return

    user_goal = input(
        "请输入一个需要分步骤完成的复杂任务："
    ).strip()

    if not user_goal:
        print("任务不能为空。")
        return

    initial_state: PlanState = {
        "user_goal": user_goal,
        "plan_goal": "",
        "steps": [],
        "current_step": None,
        "current_result": "",
        "final_answer": "",
        "step_decision": "",
        "step_grade_reason": "",
        "step_improvement": "",
        "replan_count": 0,
        "replan_reason": "",
        "current_plan_version": 1,
        "next_step_number": 1,
        "plan_approved": False
    }

    result = graph.invoke(initial_state, config=config)
    interrupts = result.get("__interrupt__", [])

    if interrupts:
        show_approval_request(
            interrupts[0].value,
            thread_id
        )
    else:
        show_final_state(result)


def resume_workflow(
    thread_id: str,
    decision: str
) -> None:
    decision = decision.strip().lower()

    if decision not in {"yes", "no"}:
        print("审批决定只能是 yes 或 no。")
        return

    config = make_config(thread_id)
    snapshot = graph.get_state(config)

    if not snapshot.values:
        print("没有找到这个流程编号。")
        return

    if not snapshot.next:
        print("该流程已经结束，没有等待审批的节点。")
        return

    print("找到暂停中的工作流。")
    print("流程编号：", thread_id)
    print("审批决定：", decision)

    result = graph.invoke(
        Command(resume=decision),
        config=config
    )

    interrupts = result.get("__interrupt__", [])

    if interrupts:
        show_approval_request(
            interrupts[0].value,
            thread_id
        )
    else:
        show_final_state(result)


parser = argparse.ArgumentParser(
    description="Day8 跨进程任务计划审批"
)
parser.add_argument(
    "action",
    choices=["start", "resume"],
    help="start 创建流程；resume 恢复流程"
)
parser.add_argument("thread_id", help="流程唯一编号")
parser.add_argument(
    "decision",
    nargs="?",
    help="resume 时填写 yes 或 no"
)

args = parser.parse_args()

try:
    if args.action == "start":
        start_workflow(args.thread_id)
    elif args.decision is None:
        print("resume 命令必须提供 yes 或 no。")
    else:
        resume_workflow(
            args.thread_id,
            args.decision
        )
finally:
    checkpoint_connection.close()


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
# review_plan_node
#      ↓ interrupt() 暂停并保存检查点
# 人工 yes/no
#      ├── no  → reject_plan_node → END
#      └── yes → Command(resume=...) 恢复
#                         ↓
# select_step_node
#      ├── 找到 pending → current_step: dict
#      │                       ↓
# executor_node
#      │  ↓ llm.invoke() 执行当前步骤
# current_result: str
#      │  ↓ 写入候选结果
# status: pending → in_progress
# result: str
#      │
# grade_step_node
#      ├── approved → completed → select_step_node
#      ├── retry    → pending   → executor_node
#      ├── replan   → failed    → replanner_node
#      │                              ↓
#      │                    旧pending变成cancelled
#      │                    新步骤加入steps
#      │                              ↓
#      │                       select_step_node
#      └── failed   → failed    → select_step_node
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
