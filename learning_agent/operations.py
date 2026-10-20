"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

import errno
import math
from pathlib import Path
from typing import TypedDict
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from .notes import NOTE_ROOT, NoteDraft, PendingNote, validate_snapshot
from .approval import approval_node, config_for, prepare_node
from .writer import write_approved_note
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

