"""Day34 第4步：严格工具参数与可观察、可重试的笔记写入失败。

python day34/python/day34_guarded_operations.py demo
python day34/python/day34_guarded_operations.py start guarded-001
python day34/python/day34_guarded_operations.py resume guarded-001 yes
python day34/python/day34_guarded_operations.py status guarded-001
python day34/python/day34_guarded_operations.py retry guarded-001

复用前3步模块；使用独立checkpoint数据库。工具供第6步最终Agent绑定。
"""

from __future__ import annotations

import argparse
import errno
import math
import tempfile
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from day34_note_contract import NOTE_ROOT, NoteDraft, PendingNote, validate_snapshot
from day34_note_interrupt import approval_node, config_for, prepare_node, resume_flow, show_status, start_flow
from day34_note_writer import write_approved_note


DATABASE_PATH = Path(__file__).resolve().parents[1] / "database/day34_guarded.db"
MAX_RETRIES = 2
TRANSIENT_ERRNOS = {errno.EAGAIN, errno.EBUSY, errno.EINTR, errno.ETIMEDOUT}


class CalculationArgs(BaseModel):
    """严格数值、有限范围，不把数字字符串或布尔值悄悄转换成参数。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    a: float = Field(ge=-1e12, le=1e12, allow_inf_nan=False)
    b: float = Field(ge=-1e12, le=1e12, allow_inf_nan=False)


def checked_result(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("计算结果不是有限数值。")
    return value


@tool("add", args_schema=CalculationArgs)
def guarded_add(a: float, b: float) -> float:
    """计算两个有限数值之和，每个输入绝对值不超过一万亿。"""
    result = checked_result(a + b)
    print(f"[执行 add] {a} + {b} = {result}")
    return result


@tool("multiply", args_schema=CalculationArgs)
def guarded_multiply(a: float, b: float) -> float:
    """计算两个有限数值之积，每个输入绝对值不超过一万亿。"""
    result = checked_result(a * b)
    print(f"[执行 multiply] {a} × {b} = {result}")
    return result


GUARDED_TOOLS = [guarded_add, guarded_multiply]


class GuardedState(TypedDict):
    draft: dict
    pending_note: dict
    approval: bool | None
    status: str
    result: str
    saved_file: str
    reused_existing: bool
    error_type: str
    retryable: bool
    retry_count: int


def failed_update(error: Exception, retryable: bool) -> dict:
    return {
        "status": "failed", "result": "笔记写入未确认成功，请查看错误类型和重试资格。",
        "saved_file": "", "reused_existing": False,
        "error_type": type(error).__name__, "retryable": retryable,
    }


def build_graph(checkpointer, note_root: Path = NOTE_ROOT):
    def prepare(state: GuardedState) -> dict:
        return {**prepare_node(state), "retry_count": 0, "retryable": False, "error_type": ""}

    def save(state: GuardedState) -> dict:
        try:
            target, reused = write_approved_note(state, note_root)
        except ValueError as error:
            # 含模型校验、快照不一致和内容冲突；不能靠重复执行解决。
            return failed_update(error, False)
        except OSError as error:
            # 磁盘满/权限不足等需要人工修复，不泛化为可重试错误。
            return failed_update(error, error.errno in TRANSIENT_ERRNOS)
        return {
            "status": "saved", "result": f"笔记已保存：{target}",
            "saved_file": str(target), "reused_existing": reused,
            "error_type": "", "retryable": False,
        }

    def reject(state: GuardedState) -> dict:
        if state["approval"] is not False:
            raise ValueError("拒绝分支要求明确的False。")
        return {
            "status": "rejected", "result": "人工拒绝，没有写入笔记。",
            "saved_file": "", "reused_existing": False,
            "error_type": "", "retryable": False,
        }

    def select(state: GuardedState) -> str:
        if state["approval"] is True:
            return "save"
        if state["approval"] is False:
            return "reject"
        raise ValueError("审批必须返回明确布尔值。")

    builder = StateGraph(GuardedState)
    for name, node in (("prepare", prepare), ("approval", approval_node), ("save", save), ("reject", reject)):
        builder.add_node(name, node)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "approval")
    builder.add_conditional_edges("approval", select, {"save": "save", "reject": "reject"})
    builder.add_edge("save", END)
    builder.add_edge("reject", END)
    return builder.compile(checkpointer=checkpointer)


def retry_flow(graph, flow_id: str) -> dict:
    """只恢复临时失败的写入，不重跑prepare，不生成新审批对象。"""
    config = config_for(flow_id)
    snapshot = graph.get_state(config)
    state = snapshot.values
    if snapshot.next or state.get("status") != "failed":
        raise ValueError("仅支持已记录的写入失败状态。")
    if state.get("approval") is not True or state.get("retryable") is not True:
        raise ValueError("没有批准或不是临时故障，拒绝重试。")
    pending = PendingNote.model_validate(state["pending_note"])
    validate_snapshot(pending)
    if pending.draft != NoteDraft.model_validate(state["draft"]):
        raise ValueError("草稿已变化，必须重新发起审批。")
    count = state.get("retry_count", 0)
    if count >= MAX_RETRIES:
        raise ValueError("已达到2次人工重试上限，请排查存储原因。")
    # 更新状态并声明来自approval，条件边重新调度save；不再调用interrupt。
    graph.update_state(config, {
        "status": "pending", "retry_count": count + 1,
        "error_type": "", "retryable": False,
    }, as_node="approval")
    return graph.invoke(None, config=config)


def demonstrate() -> None:
    assert guarded_add.invoke({"a": 20, "b": 4}) == 24
    assert guarded_multiply.invoke({"a": 24, "b": 3}) == 72
    for label, args in (
        ("数字字符串", {"a": "20", "b": 4}),
        ("布尔值", {"a": True, "b": 4}),
        ("非有限值", {"a": float("inf"), "b": 4}),
        ("未知字段", {"a": 20, "b": 4, "admin": True}),
        ("超范围", {"a": 1e13, "b": 4}),
    ):
        try:
            guarded_add.invoke(args)
        except ValidationError:
            print("[预期拒绝]", label)
        else:
            raise AssertionError(f"非法参数未拒绝：{label}")

    with tempfile.TemporaryDirectory(prefix="day34-guarded-") as directory:
        root = Path(directory)
        draft = NoteDraft(session_id="xiaolin", title="异常处理", content="失败必须可观察，重试必须绑定原审批。")
        with SqliteSaver.from_conn_string(str(root / "demo.db")) as saver:
            graph = build_graph(saver, root / "notes")
            paused = start_flow(graph, "busy-001", draft)
            original = paused["pending_note"]
            with patch(__name__ + ".write_approved_note", side_effect=OSError(errno.EBUSY, "模拟存储忙")):
                failed = resume_flow(graph, "busy-001", True)
            assert failed["status"] == "failed" and failed["retryable"]
            assert not (root / "notes").exists()
            print("[通过] 临时存储错误记录failed，未误报saved")
            recovered = retry_flow(graph, "busy-001")
            assert recovered["status"] == "saved" and recovered["retry_count"] == 1
            assert recovered["pending_note"] == original
            print("[通过] 同一已批准快照重试成功，没有重新审批")
            start_flow(graph, "permission-001", draft)
            with patch(__name__ + ".write_approved_note", side_effect=PermissionError(errno.EACCES, "模拟权限不足")):
                denied = resume_flow(graph, "permission-001", True)
            assert denied["status"] == "failed" and not denied["retryable"]
            try:
                retry_flow(graph, "permission-001")
            except ValueError:
                print("[通过] 权限错误阻止直接重试")
            else:
                raise AssertionError("不可重试错误被放行。")
    print("全部验证通过；演示使用临时目录和模拟存储错误。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day34工具校验与存储错误处理")
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start")
    start.add_argument("flow_id")
    start.add_argument("--session-id", default="xiaolin-capstone-001")
    resume = commands.add_parser("resume")
    resume.add_argument("flow_id")
    resume.add_argument("decision", choices=("yes", "no"))
    for name in ("status", "retry"):
        command = commands.add_parser(name)
        command.add_argument("flow_id")
    commands.add_parser("demo")
    args = parser.parse_args()
    print("Day34 第4步：参数校验与异常处理")
    try:
        if args.action == "demo":
            demonstrate()
            return
        config_for(args.flow_id)
        draft = None
        if args.action == "start":
            draft = NoteDraft(session_id=args.session_id, title=input("笔记标题："), content=input("笔记正文："))
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(DATABASE_PATH)) as saver:
            graph = build_graph(saver)
            if args.action == "start":
                start_flow(graph, args.flow_id, draft)
            elif args.action == "resume":
                resume_flow(graph, args.flow_id, args.decision == "yes")
            elif args.action == "retry":
                retry_flow(graph, args.flow_id)
            show_status(graph, args.flow_id)
            state = graph.get_state(config_for(args.flow_id)).values
            print("错误类型：", state.get("error_type", ""))
            print("可重试：", state.get("retryable", False), "已重试：", state.get("retry_count", 0))
    except (ValueError, OSError) as error:
        print("操作失败：", error)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
