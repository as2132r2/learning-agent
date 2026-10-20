import json
import operator
import os
import sqlite3
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field


# ============================================================
# 1. 模型和专业工具
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


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

    print(f"[researcher工具] 查询：{keyword}")

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
        return f"没有找到包含“{keyword}”的资料。"

    parts: list[str] = []

    for rank, (text, metadata_text) in enumerate(rows, start=1):
        metadata = json.loads(metadata_text) if metadata_text else {}
        source = metadata.get("source", "未知来源")
        page = metadata.get("page")
        location = (
            f"{source}，第{page}页"
            if page is not None
            else source
        )
        parts.append(f"资料{rank} | {location}\n{text}")

    return "\n\n".join(parts)


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


researcher_llm = llm.bind_tools([search_course_knowledge])
calculator_llm = llm.bind_tools([calculate])

specialist_tool_maps = {
    "researcher": {
        search_course_knowledge.name: search_course_knowledge
    },
    "calculator": {
        calculate.name: calculate
    }
}


# ============================================================
# 2. 主管的并行任务计划
# ============================================================

class ParallelAssignment(BaseModel):
    """一项可以独立并行执行的研究或计算任务。"""

    worker: Literal["researcher", "calculator"]
    task: str = Field(description="不依赖其他任务结果的独立任务")
    expected_output: str = Field(description="该任务预期返回的结果")


class ParallelPlan(BaseModel):
    goal: str = Field(description="团队最终目标")
    assignments: list[ParallelAssignment] = Field(
        min_length=1,
        max_length=4,
        description="可以同时执行的1到4项独立任务"
    )


class WorkerReview(BaseModel):
    """主管对一项并行专业结果的检查。"""

    approved: bool = Field(description="结果是否满足该任务的预期输出")
    reason: str = Field(description="检查结论的具体原因")
    feedback: str = Field(description="返工时必须修改的内容")


supervisor_llm = llm.with_structured_output(
    ParallelPlan,
    method="function_calling"
)

reviewer_llm = llm.with_structured_output(
    WorkerReview,
    method="function_calling"
)

MAX_WORKER_ATTEMPTS = 2


# ============================================================
# 3. 主图状态和并行工作单元输入
# ============================================================

class ParallelTeamState(TypedDict):
    user_request: str
    team_goal: str
    assignments: list[dict]

    # 多个并行节点都会返回list[dict]。
    # operator.add会把这些列表合并，而不是互相覆盖。
    worker_results: Annotated[list[dict], operator.add]

    # batch_review写入的最新检查结果和待返工任务。
    result_reviews: dict[str, dict]
    retry_assignments: list[dict]

    final_answer: str


class SpecialistInput(TypedDict):
    assignment: dict


# ============================================================
# 4. supervisor：生成可并行任务
# ============================================================

def supervisor_node(state: ParallelTeamState) -> dict:
    print("\n[supervisor 节点]")

    plan: ParallelPlan = supervisor_llm.invoke([
        SystemMessage(
            content=(
                "你是并行多Agent团队的主管。"
                "请把用户请求拆成1到4项互不依赖、可以同时执行的任务。"
                "researcher负责查询和解释课程知识；"
                "calculator负责数学运算。"
                "不要创建writer任务，最终整理会由系统统一完成。"
                "如果两个任务存在先后依赖，就不要把它们拆成并行任务。"
            )
        ),
        HumanMessage(content=state["user_request"])
    ])

    assignments: list[dict] = []

    for index, assignment in enumerate(plan.assignments, start=1):
        assignments.append({
            "assignment_id": f"parallel-{index:03d}",
            **assignment.model_dump()
        })

    print("团队目标：", plan.goal)
    print("并行任务数量：", len(assignments))

    for assignment in assignments:
        print(
            f"{assignment['assignment_id']} | "
            f"{assignment['worker']} | "
            f"{assignment['task']}"
        )

    return {
        "team_goal": plan.goal,
        "assignments": assignments
    }


# ============================================================
# 5. dispatch_workers：使用Send动态扇出
# ============================================================

def dispatch_workers(state: ParallelTeamState) -> list[Send]:
    """为每项任务创建一次specialist_worker执行。"""

    return [
        Send(
            "specialist_worker",
            {
                "assignment": {
                    **assignment,
                    "attempt": 1,
                    "feedback": ""
                }
            }
        )
        for assignment in state["assignments"]
    ]


# ============================================================
# 6. specialist_worker：并行专业工作单元
# ============================================================

def specialist_worker_node(state: SpecialistInput) -> dict:
    assignment = state["assignment"]
    worker = assignment["worker"]

    print(
        f"\n[specialist_worker：{assignment['assignment_id']}]"
    )
    print("专业角色：", worker)
    print("任务：", assignment["task"])
    print("执行次数：", assignment["attempt"])

    if worker == "researcher":
        bound_llm = researcher_llm
        tool_map = specialist_tool_maps["researcher"]
        role_instruction = (
            "你是researcher，必须使用search_course_knowledge查询资料，"
            "根据工具结果解释知识并保留来源。"
        )
    else:
        bound_llm = calculator_llm
        tool_map = specialist_tool_maps["calculator"]
        role_instruction = (
            "你是calculator，必须使用calculate完成计算和数值验证。"
        )

    messages = [
        SystemMessage(content=role_instruction),
        HumanMessage(
            content=(
                f"独立任务：{assignment['task']}\n"
                f"预期输出：{assignment['expected_output']}\n"
                f"主管返工意见：{assignment.get('feedback') or '无'}"
            )
        )
    ]

    tool_trace: list[str] = []
    result = ""

    for tool_round in range(1, 6):
        response = bound_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            result = response.content
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_arguments = tool_call["args"]
            selected_tool = tool_map.get(tool_name)

            if selected_tool is None:
                tool_result = f"无权调用工具：{tool_name}"
            else:
                tool_result = selected_tool.invoke(tool_arguments)

            tool_result_text = str(tool_result)
            tool_trace.append(
                f"{tool_name}({tool_arguments}) → {tool_result_text}"
            )
            messages.append(ToolMessage(
                content=tool_result_text,
                tool_call_id=tool_call["id"]
            ))
    else:
        result = "工具调用达到5轮上限。"

    if tool_trace:
        result = (
            f"{result}\n\n工具执行记录：\n- "
            + "\n- ".join(tool_trace)
        )

    worker_result = {
        "assignment_id": assignment["assignment_id"],
        "worker": worker,
        "task": assignment["task"],
        "expected_output": assignment["expected_output"],
        "attempt": assignment["attempt"],
        "result": result
    }

    print("专业结果：", result)

    # 注意：返回的是只有一个元素的列表。
    # reducer会把所有并行节点返回的列表相加。
    return {
        "worker_results": [worker_result]
    }


# ============================================================
# 7. batch_review：批量检查最新并行结果
# ============================================================

def get_latest_results(state: ParallelTeamState) -> dict[str, dict]:
    """每个assignment_id只保留attempt最大的那条结果。"""

    latest: dict[str, dict] = {}

    for result in state["worker_results"]:
        assignment_id = result["assignment_id"]
        previous = latest.get(assignment_id)

        if (
            previous is None
            or result["attempt"] > previous["attempt"]
        ):
            latest[assignment_id] = result

    return latest


def batch_review_node(state: ParallelTeamState) -> dict:
    print("\n[batch_review 节点]")

    latest_results = get_latest_results(state)
    assignment_map = {
        assignment["assignment_id"]: assignment
        for assignment in state["assignments"]
    }

    reviews: dict[str, dict] = {}
    retry_assignments: list[dict] = []

    for assignment_id, result in latest_results.items():
        previous_review = state["result_reviews"].get(assignment_id)

        # 已经批准且没有新attempt的结果保持不变，不重复调用检查模型。
        if (
            previous_review is not None
            and previous_review["status"] == "approved"
            and previous_review["attempt"] == result["attempt"]
        ):
            reviews[assignment_id] = previous_review
            print(f"{assignment_id} | 保留上一轮approved结果")
            continue

        review: WorkerReview = reviewer_llm.invoke([
            SystemMessage(
                content=(
                    "你是并行团队的主管检查员。"
                    "只检查当前专业结果是否完成任务并满足预期输出。"
                    "不合格时给出具体可操作的返工意见。"
                )
            ),
            HumanMessage(
                content=(
                    f"专业成员：{result['worker']}\n"
                    f"任务：{result['task']}\n"
                    f"预期输出：{result['expected_output']}\n\n"
                    f"实际结果：\n{result['result']}"
                )
            )
        ])

        if review.approved:
            status = "approved"
        elif result["attempt"] < MAX_WORKER_ATTEMPTS:
            status = "retry"
            retry_assignments.append({
                **assignment_map[assignment_id],
                "attempt": result["attempt"] + 1,
                "feedback": review.feedback
            })
        else:
            status = "failed"

        reviews[assignment_id] = {
            "status": status,
            "reason": review.reason,
            "feedback": review.feedback,
            "attempt": result["attempt"]
        }

        print(
            f"{assignment_id} | {result['worker']} | "
            f"第{result['attempt']}次 | {status}"
        )
        print("原因：", review.reason)

    return {
        "result_reviews": reviews,
        "retry_assignments": retry_assignments
    }


def route_after_batch_review(
    state: ParallelTeamState
) -> list[Send] | Literal["writer"]:
    """只把不合格任务重新并行分发；没有返工任务就进入writer。"""

    if not state["retry_assignments"]:
        return "writer"

    return [
        Send(
            "specialist_worker",
            {"assignment": assignment}
        )
        for assignment in state["retry_assignments"]
    ]


# ============================================================
# 8. writer：等待检查和返工结束后统一汇总
# ============================================================

def writer_node(state: ParallelTeamState) -> dict:
    print("\n[writer 汇总节点]")
    latest_results = get_latest_results(state)
    print("收到最终专业结果数量：", len(latest_results))

    result_parts: list[str] = []

    # 使用主管计划中的order显示，避免并行完成先后影响最终顺序。
    ordered_assignments = sorted(
        state["assignments"],
        key=lambda item: item["assignment_id"]
    )

    for assignment in ordered_assignments:
        assignment_id = assignment["assignment_id"]
        result = latest_results[assignment_id]
        review = state["result_reviews"][assignment_id]
        result_parts.append(
            f"任务ID：{result['assignment_id']}\n"
            f"成员：{result['worker']}\n"
            f"执行次数：{result['attempt']}\n"
            f"主管状态：{review['status']}\n"
            f"任务：{result['task']}\n"
            f"结果：{result['result']}"
        )

    combined_results = "\n\n".join(result_parts)

    response = llm.invoke([
        SystemMessage(
            content=(
                "你是writer专业Agent。"
                "请根据所有并行专业结果回答用户原始请求。"
                "不得修改真实计算结果，不得编造资料来源。"
                "请整理成结构清晰、可直接使用的最终文本。"
            )
        ),
        HumanMessage(
            content=(
                f"用户请求：{state['user_request']}\n\n"
                f"团队目标：{state['team_goal']}\n\n"
                f"并行专业结果：\n{combined_results}"
            )
        )
    ])

    print("writer最终结果：", response.content)
    return {"final_answer": response.content}


# ============================================================
# 9. 构建并行检查图
# ============================================================

builder = StateGraph(ParallelTeamState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("specialist_worker", specialist_worker_node)
builder.add_node("batch_review", batch_review_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "supervisor")

# dispatch_workers返回多个Send，动态创建多个并行worker任务。
builder.add_conditional_edges(
    "supervisor",
    dispatch_workers,
    ["specialist_worker"]
)

# 所有并行worker完成后，先集中检查。
builder.add_edge("specialist_worker", "batch_review")

builder.add_conditional_edges(
    "batch_review",
    route_after_batch_review,
    ["specialist_worker", "writer"]
)

builder.add_edge("writer", END)

graph = builder.compile()


# ============================================================
# 10. 运行
# ============================================================

user_request = input("请输入一个可并行处理的复合任务：").strip()

if not user_request:
    raise ValueError("任务不能为空。")

initial_state: ParallelTeamState = {
    "user_request": user_request,
    "team_goal": "",
    "assignments": [],
    "worker_results": [],
    "result_reviews": {},
    "retry_assignments": [],
    "final_answer": ""
}

final_state = graph.invoke(initial_state)


print("\n" + "=" * 60)
print("并行团队任务完成。")
print("专业结果数量：", len(final_state["worker_results"]))

for result in final_state["worker_results"]:
    print(
        f"- {result['assignment_id']} | "
        f"{result['worker']} | {result['task']}"
    )

print("\n最终主管检查：")
for assignment_id, review in final_state["result_reviews"].items():
    print(
        f"- {assignment_id} | {review['status']} | "
        f"第{review['attempt']}次"
    )

print("\n最终回答：")
print(final_state["final_answer"])


# ============================================================
# 数据流
# ============================================================
#
# supervisor
#   ↓ assignments
# dispatch_workers
#   ├── Send → specialist_worker(researcher)
#   ├── Send → specialist_worker(calculator)
#   └── Send → specialist_worker(...)
#                    ↓ 每个节点返回list[dict]
# worker_results: Annotated[list[dict], operator.add]
#                    ↓ reducer合并
# batch_review
#   ├── 不合格且未到上限 → Send返工任务 → specialist_worker
#   └── 全部通过或已到上限
#                    ↓
# writer
#   ↓
# END
