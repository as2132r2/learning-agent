"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

import re
from typing import Literal, TypedDict
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field
from .notes import NoteDraft, PendingNote, prepare_note, validate_snapshot


class ApprovalDecision(BaseModel):
    """恢复输入必须对应同一会话、笔记编号与快照哈希。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str
    note_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved: bool


class ApprovalState(TypedDict):
    draft: dict
    pending_note: dict
    approval: bool | None
    status: Literal["preparing", "pending", "approved", "rejected"]
    result: str


def config_for(flow_id: str) -> dict:
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", flow_id) is None:
        raise ValueError("流程编号只能包含英文字母、数字、短横线和下划线。")
    return {"configurable": {"thread_id": flow_id}}


def prepare_node(state: ApprovalState) -> dict:
    # 编号在独立前置节点生成并checkpoint，不在interrupt所在节点里随机生成。
    pending = prepare_note(NoteDraft.model_validate(state["draft"]))
    return {"pending_note": pending.model_dump(), "status": "pending"}


def approval_node(state: ApprovalState) -> dict:
    pending = PendingNote.model_validate(state["pending_note"])
    validate_snapshot(pending)
    print("[approval] 进入审批节点；恢复时会从节点开头再次执行")
    raw_decision = interrupt({
        "question": "是否批准这份学习笔记？",
        "session_id": pending.draft.session_id,
        "note_id": pending.note_id,
        "title": pending.draft.title,
        "content": pending.draft.content,
        "content_sha256": pending.content_sha256,
    })
    decision = ApprovalDecision.model_validate(raw_decision)
    if (
        decision.session_id != pending.draft.session_id
        or decision.note_id != pending.note_id
        or decision.content_sha256 != pending.content_sha256
    ):
        raise ValueError("审批决定与当前待审批快照不匹配。")
    print("[approval] 收到有效决定：", decision.approved)
    return {"approval": decision.approved}


def start_flow(graph, flow_id: str, draft: NoteDraft) -> dict:
    config = config_for(flow_id)
    if graph.get_state(config).values:
        raise ValueError("流程编号已存在，请使用status/resume，或用新的流程编号发起。")
    return graph.invoke({
        "draft": draft.model_dump(), "pending_note": {},
        "approval": None, "status": "preparing", "result": "",
    }, config=config)


def resume_flow(graph, flow_id: str, approved: bool) -> dict:
    config = config_for(flow_id)
    snapshot = graph.get_state(config)
    if snapshot.values.get("status") != "pending" or not any(
        task.interrupts for task in snapshot.tasks
    ):
        raise ValueError("该流程没有等待人工决定的interrupt，不能恢复或重复审批。")
    pending = PendingNote.model_validate(snapshot.values["pending_note"])
    validate_snapshot(pending)
    decision = ApprovalDecision(
        session_id=pending.draft.session_id, note_id=pending.note_id,
        content_sha256=pending.content_sha256, approved=approved,
    )
    # CLI先读取本流程实际快照，再将人工yes/no与该快照绑定。
    return graph.invoke(Command(resume=decision.model_dump()), config=config)

