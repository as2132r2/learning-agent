import json
import os
import sqlite3
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# ============================================================
# 1. 基础模型
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 专业Agent各自拥有的工具
# ============================================================

@tool
def search_course_knowledge(keyword: str) -> str:
    """根据关键词查询本地LangChain课程知识库。"""

    project_directory = Path(__file__).resolve().parent.parent.parent
    database_path = (
        project_directory
        / "day06"
        / "database"
        / "day6_knowledge_cli.db"
    )

    print(f"[researcher工具] 查询关键词：{keyword}")

    if not database_path.exists():
        return "没有找到Day6课程知识库。"

    connection = sqlite3.connect(database_path)

    try:
        rows = connection.execute(
            """
            SELECT text, metadata
            FROM knowledge_chunks
            WHERE text LIKE ?
            LIMIT 3
            """,
            (f"%{keyword}%",)
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        return f"没有找到包含“{keyword}”的课程资料。"

    results: list[str] = []

    for rank, (text, metadata_text) in enumerate(rows, start=1):
        metadata = json.loads(metadata_text) if metadata_text else {}
        source = metadata.get("source", "未知来源")
        page = metadata.get("page")
        location = (
            f"{source}，第{page}页"
            if page is not None
            else source
        )
        results.append(f"资料{rank} | {location}\n{text}")

    return "\n\n".join(results)


@tool
def calculate(
    a: float,
    b: float,
    operation: Literal["add", "multiply"]
) -> float:
    """计算两个数字的加法或乘法。"""

    if operation == "add":
        result = a + b
        print(f"[calculator工具] {a} + {b} = {result}")
        return result

    result = a * b
    print(f"[calculator工具] {a} × {b} = {result}")
    return result


researcher_tools = [search_course_knowledge]
calculator_tools = [calculate]

researcher_llm = llm.bind_tools(researcher_tools)
calculator_llm = llm.bind_tools(calculator_tools)

researcher_tool_map = {
    current_tool.name: current_tool
    for current_tool in researcher_tools
}

calculator_tool_map = {
    current_tool.name: current_tool
    for current_tool in calculator_tools
}


# ============================================================
# 3. 主管的结构化团队计划
# ============================================================

class TeamAssignment(BaseModel):
    """主管分配给一个专业Agent的独立任务。"""

    worker: Literal[
        "researcher",
        "calculator",
        "writer"
    ] = Field(description="负责该任务的专业Agent")

    task: str = Field(description="可以独立理解和执行的完整任务")
    expected_output: str = Field(description="该任务必须产生的结果")


class TeamPlan(BaseModel):
    """主管为整个用户请求制定的团队协作计划。"""

    goal: str = Field(description="团队最终需要完成的目标")

    assignments: list[TeamAssignment] = Field(
        min_length=1,
        max_length=5,
        description="按执行顺序排列的1到5项专业任务"
    )


class ReviewDecision(BaseModel):
    """主管对当前专业任务结果的检查。"""

    approved: bool = Field(description="结果是否满足当前任务要求")
    reason: str = Field(description="检查结论的具体原因")
    feedback: str = Field(description="返工时的修改意见")


supervisor_llm = llm.with_structured_output(
    TeamPlan,
    method="function_calling"
)

reviewer_llm = llm.with_structured_output(
    ReviewDecision,
    method="function_calling"
)

MAX_WORKER_ATTEMPTS = 2


# ============================================================
# 3. 多Agent共享状态
# ============================================================

class TeamState(TypedDict):
    user_request: str
    team_goal: str

    # 所有任务，状态可能是pending/in_progress/completed/failed。
    assignments: list[dict]

    # 当前选中的一项任务。
    current_assignment: dict | None

    # 最近一次专业Agent输出。
    current_result: str

    # 最近一次主管检查决定。
    review_status: str
    review_reason: str
    review_feedback: str

    # 最终汇总回答。
    final_answer: str


# ============================================================
# 4. supervisor：拆分复合任务
# ============================================================

def supervisor_node(state: TeamState) -> dict:
    print("\n[supervisor 节点]")

    plan: TeamPlan = supervisor_llm.invoke([
        SystemMessage(
            content=(
                "你是多Agent团队的主管，只负责拆分和分派任务。"
                "可用成员：researcher负责知识研究，calculator负责数学计算，"
                "writer负责根据已有结果整理最终文本。"
                "把用户请求拆成1到5项按先后顺序执行的任务。"
                "不要为了使用所有成员而制造无关任务。"
                "如果writer需要前面结果，请在它的task中明确说明。"
            )
        ),
        HumanMessage(content=state["user_request"])
    ])

    assignments: list[dict] = []

    for order, assignment in enumerate(plan.assignments, start=1):
        assignments.append({
            "assignment_id": f"assignment-{order:03d}",
            "order": order,
            **assignment.model_dump(),
            "status": "pending",
            "attempts": 0,
            "result": "",
            "feedback": ""
        })

    print("团队目标：", plan.goal)
    print("任务数量：", len(assignments))

    for assignment in assignments:
        print(
            f"{assignment['order']}. "
            f"{assignment['worker']} → {assignment['task']}"
        )

    return {
        "team_goal": plan.goal,
        "assignments": assignments
    }


# ============================================================
# 5. select_assignment：选择下一项任务
# ============================================================

def select_assignment_node(state: TeamState) -> dict:
    print("\n[select_assignment 节点]")

    pending = [
        assignment
        for assignment in state["assignments"]
        if assignment["status"] == "pending"
    ]

    current = (
        min(pending, key=lambda item: item["order"])
        if pending
        else None
    )

    if current is None:
        print("没有待执行的团队任务。")
    else:
        print("任务ID：", current["assignment_id"])
        print("专业Agent：", current["worker"])
        print("任务：", current["task"])

    return {
        "current_assignment": current,
        "review_feedback": ""
    }


def choose_assignment_route(
    state: TeamState
) -> Literal[
    "researcher",
    "calculator",
    "writer",
    "finish"
]:
    current = state["current_assignment"]

    if current is None:
        return "finish"

    return current["worker"]


# ============================================================
# 6. 专业Agent共用的执行函数
# ============================================================

def run_worker(
    state: TeamState,
    role_name: str,
    role_instruction: str,
    bound_llm=None,
    tool_map: dict | None = None
) -> dict:
    current = state["current_assignment"]

    if current is None:
        raise ValueError(f"{role_name}没有收到current_assignment。")

    print(f"\n[{role_name} 节点]")

    # 把之前已完成任务的结果提供给后续成员。
    completed_context_parts: list[str] = []

    for assignment in state["assignments"]:
        if assignment["status"] == "completed":
            completed_context_parts.append(
                f"任务 {assignment['assignment_id']}\n"
                f"负责人：{assignment['worker']}\n"
                f"结果：{assignment['result']}"
            )

    completed_context = (
        "\n\n".join(completed_context_parts)
        if completed_context_parts
        else "暂无已完成任务。"
    )

    worker_messages = [
        SystemMessage(
            content=(
                f"你是{role_name}专业Agent。"
                f"{role_instruction}"
                "只完成当前分配的任务，不要代替其他成员完成后续任务。"
            )
        ),
        HumanMessage(
            content=(
                f"团队目标：{state['team_goal']}\n\n"
                f"之前成员的完成结果：\n{completed_context}\n\n"
                f"当前任务：{current['task']}\n"
                f"预期输出：{current['expected_output']}\n"
                f"主管返工意见：{state['review_feedback'] or '无'}"
            )
        )
    ]

    # writer没有工具，直接调用基础模型。
    if bound_llm is None or tool_map is None:
        response = llm.invoke(worker_messages)
        worker_result = response.content
    else:
        # researcher和calculator最多连续调用5轮工具。
        tool_trace: list[str] = []
        worker_result = ""

        for tool_round in range(1, 6):
            response = bound_llm.invoke(worker_messages)
            worker_messages.append(response)

            if not response.tool_calls:
                worker_result = response.content
                break

            print(f"{role_name}工具调用轮次：{tool_round}")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_arguments = tool_call["args"]
                selected_tool = tool_map.get(tool_name)

                if selected_tool is None:
                    tool_result = f"无权调用工具：{tool_name}"
                else:
                    tool_result = selected_tool.invoke(tool_arguments)

                tool_result_text = str(tool_result)

                print("工具名称：", tool_name)
                print("工具参数：", tool_arguments)
                print("工具结果：", tool_result_text)

                tool_trace.append(
                    f"{tool_name}({tool_arguments}) → {tool_result_text}"
                )

                worker_messages.append(ToolMessage(
                    content=tool_result_text,
                    tool_call_id=tool_call["id"]
                ))
        else:
            worker_result = "工具调用达到5轮上限，未形成最终结果。"

        if tool_trace:
            worker_result = (
                f"{worker_result}\n\n"
                "工具执行记录：\n- "
                + "\n- ".join(tool_trace)
            )

    updated_assignments: list[dict] = []
    updated_current: dict | None = None

    for assignment in state["assignments"]:
        updated = assignment.copy()

        if assignment["assignment_id"] == current["assignment_id"]:
            updated["status"] = "in_progress"
            updated["attempts"] = assignment["attempts"] + 1
            updated["result"] = worker_result
            updated_current = updated

        updated_assignments.append(updated)

    print("执行次数：", updated_current["attempts"])
    print("执行结果：", worker_result)

    return {
        "assignments": updated_assignments,
        "current_assignment": updated_current,
        "current_result": worker_result
    }


def researcher_node(state: TeamState) -> dict:
    return run_worker(
        state,
        "researcher",
        "必须使用search_course_knowledge查询课程资料，"
        "并基于工具结果分析知识、解释概念和标注来源。",
        researcher_llm,
        researcher_tool_map
    )


def calculator_node(state: TeamState) -> dict:
    return run_worker(
        state,
        "calculator",
        "必须使用calculate完成数学运算和数值验证。",
        calculator_llm,
        calculator_tool_map
    )


def writer_node(state: TeamState) -> dict:
    return run_worker(
        state,
        "writer",
        "负责使用已有材料整理结构清晰、可直接交付的文本。"
    )


# ============================================================
# 7. supervisor_review：逐项检查
# ============================================================

def supervisor_review_node(state: TeamState) -> dict:
    print("\n[supervisor_review 节点]")

    current = state["current_assignment"]

    if current is None:
        raise ValueError("主管检查时没有current_assignment。")

    review: ReviewDecision = reviewer_llm.invoke([
        SystemMessage(
            content=(
                "你是多Agent团队主管，负责检查当前成员结果。"
                "只根据当前任务和预期输出判断，不要求成员完成其他任务。"
                "不合格时给出具体的返工意见。"
            )
        ),
        HumanMessage(
            content=(
                f"成员：{current['worker']}\n"
                f"任务：{current['task']}\n"
                f"预期输出：{current['expected_output']}\n\n"
                f"实际结果：\n{state['current_result']}"
            )
        )
    ])

    attempts = current["attempts"]

    if review.approved:
        status = "approved"
        assignment_status = "completed"
    elif attempts < MAX_WORKER_ATTEMPTS:
        status = "retry"
        assignment_status = "pending"
    else:
        status = "failed"
        assignment_status = "failed"

    updated_assignments: list[dict] = []
    reviewed_current: dict | None = None

    for assignment in state["assignments"]:
        updated = assignment.copy()

        if assignment["assignment_id"] == current["assignment_id"]:
            updated["status"] = assignment_status
            updated["feedback"] = review.feedback
            updated["review_reason"] = review.reason
            reviewed_current = updated

        updated_assignments.append(updated)

    print("检查状态：", status)
    print("检查原因：", review.reason)
    print("返工意见：", review.feedback)

    return {
        "assignments": updated_assignments,
        "current_assignment": reviewed_current,
        "review_status": status,
        "review_reason": review.reason,
        "review_feedback": review.feedback
    }


def choose_review_route(
    state: TeamState
) -> Literal[
    "researcher",
    "calculator",
    "writer",
    "next"
]:
    if state["review_status"] == "retry":
        return state["current_assignment"]["worker"]

    return "next"


# ============================================================
# 8. finalizer：汇总团队结果
# ============================================================

def finalizer_node(state: TeamState) -> dict:
    print("\n[finalizer 节点]")

    result_parts: list[str] = []

    for assignment in state["assignments"]:
        result_parts.append(
            f"任务：{assignment['task']}\n"
            f"成员：{assignment['worker']}\n"
            f"状态：{assignment['status']}\n"
            f"结果：{assignment['result']}"
        )

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是团队成果汇总员。"
                "请根据各成员的真实结果回答用户原始请求。"
                "如果存在failed任务，必须如实说明限制。"
                "不要编造成员没有提供的事实。"
            )
        ),
        HumanMessage(
            content=(
                f"用户请求：{state['user_request']}\n\n"
                f"团队执行结果：\n{'\n\n'.join(result_parts)}"
            )
        )
    ])

    print("团队最终结果：", response.content)
    return {"final_answer": response.content}


# ============================================================
# 9. 构建多Agent团队图
# ============================================================

builder = StateGraph(TeamState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("select_assignment", select_assignment_node)
builder.add_node("researcher", researcher_node)
builder.add_node("calculator", calculator_node)
builder.add_node("writer", writer_node)
builder.add_node("supervisor_review", supervisor_review_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "select_assignment")

builder.add_conditional_edges(
    "select_assignment",
    choose_assignment_route,
    {
        "researcher": "researcher",
        "calculator": "calculator",
        "writer": "writer",
        "finish": "finalizer"
    }
)

builder.add_edge("researcher", "supervisor_review")
builder.add_edge("calculator", "supervisor_review")
builder.add_edge("writer", "supervisor_review")

builder.add_conditional_edges(
    "supervisor_review",
    choose_review_route,
    {
        "researcher": "researcher",
        "calculator": "calculator",
        "writer": "writer",
        "next": "select_assignment"
    }
)

builder.add_edge("finalizer", END)

graph = builder.compile()


# ============================================================
# 10. 运行
# ============================================================

user_request = input("请输入一个需要团队协作的复合任务：").strip()

if not user_request:
    raise ValueError("任务不能为空。")

initial_state: TeamState = {
    "user_request": user_request,
    "team_goal": "",
    "assignments": [],
    "current_assignment": None,
    "current_result": "",
    "review_status": "",
    "review_reason": "",
    "review_feedback": "",
    "final_answer": ""
}

final_state = graph.invoke(
    initial_state,
    config={"recursion_limit": 40}
)


# ============================================================
# 11. 查看最终状态
# ============================================================

print("\n" + "=" * 60)
print("团队任务执行完成。")
print("团队目标：", final_state["team_goal"])

print("\n各项任务状态：")
for assignment in final_state["assignments"]:
    print(
        f"{assignment['order']}. "
        f"{assignment['worker']} | "
        f"{assignment['status']} | "
        f"尝试{assignment['attempts']}次"
    )

print("\n最终回答：")
print(final_state["final_answer"])


# ============================================================
# 数据流
# ============================================================
#
# user_request
#   ↓
# supervisor → assignments: list[dict]
#   ↓
# select_assignment
#   ├── researcher ─┐
#   ├── calculator ─┼→ supervisor_review
#   └── writer ─────┘        ├── retry → 原成员
#                            └── next  → select_assignment
#                                           ↓ 无pending
#                                       finalizer
#                                           ↓
#                                          END
