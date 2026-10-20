"""Day11 第三步：使用unittest测试LangGraph节点和条件路由。"""

import unittest
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    task: str
    attempts: int
    result: str
    decision: str
    events: list[str]


MAX_ATTEMPTS = 2


# ============================================================
# 1. 构建一个完全不调用模型的测试图
# ============================================================

def planner_node(state: WorkflowState) -> dict:
    return {
        "events": state["events"] + ["planner"]
    }


def worker_node(state: WorkflowState) -> dict:
    next_attempt = state["attempts"] + 1

    # 第一次故意返回不合格结果，第二次返回合格结果。
    if next_attempt == 1:
        result = "第一次结果：信息不完整"
    else:
        result = "第二次结果：任务已完整完成"

    return {
        "attempts": next_attempt,
        "result": result,
        "events": state["events"] + [
            f"worker-attempt-{next_attempt}"
        ]
    }


def grader_node(state: WorkflowState) -> dict:
    if "完整完成" in state["result"]:
        decision = "approved"
    elif state["attempts"] < MAX_ATTEMPTS:
        decision = "retry"
    else:
        decision = "failed"

    return {
        "decision": decision,
        "events": state["events"] + [
            f"grader-{decision}"
        ]
    }


def route_after_grader(
    state: WorkflowState
) -> Literal["worker", "finish"]:
    if state["decision"] == "retry":
        return "worker"

    return "finish"


def build_test_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("planner", planner_node)
    builder.add_node("worker", worker_node)
    builder.add_node("grader", grader_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker", "grader")
    builder.add_conditional_edges(
        "grader",
        route_after_grader,
        {
            "worker": "worker",
            "finish": END
        }
    )

    return builder.compile()


# ============================================================
# 2. 测试图的执行结果
# ============================================================

class TestGraphWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_test_graph()

    def make_initial_state(self) -> WorkflowState:
        return {
            "task": "完成一个测试任务",
            "attempts": 0,
            "result": "",
            "decision": "",
            "events": []
        }

    def test_failed_first_attempt_retries_and_finishes(self) -> None:
        final_state = self.graph.invoke(
            self.make_initial_state()
        )

        self.assertEqual(final_state["attempts"], 2)
        self.assertEqual(final_state["decision"], "approved")
        self.assertIn("完整完成", final_state["result"])
        self.assertEqual(
            final_state["events"],
            [
                "planner",
                "worker-attempt-1",
                "grader-retry",
                "worker-attempt-2",
                "grader-approved"
            ]
        )

    def test_route_returns_worker_for_retry(self) -> None:
        state = self.make_initial_state()
        state["decision"] = "retry"
        self.assertEqual(route_after_grader(state), "worker")

    def test_route_returns_finish_for_approved(self) -> None:
        state = self.make_initial_state()
        state["decision"] = "approved"
        self.assertEqual(route_after_grader(state), "finish")

    def test_max_attempt_constant_is_two(self) -> None:
        self.assertEqual(MAX_ATTEMPTS, 2)


if __name__ == "__main__":
    print("开始运行 Day11 LangGraph 测试……")
    unittest.main(verbosity=2)


# ============================================================
# 数据流
# ============================================================
#
# START
#   ↓
# planner
#   ↓
# worker（第一次）
#   ↓
# grader → retry
#   ↓
# worker（第二次）
#   ↓
# grader → approved
#   ↓
# END
#
# 测试通过的依据不是“屏幕打印了什么”，而是最终状态：
# attempts、decision、result和events都符合预期。
