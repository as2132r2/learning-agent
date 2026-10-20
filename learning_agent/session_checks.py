"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

from . import operations
from .notes import NoteDraft, PendingNote, validate_snapshot
from .approval import config_for, resume_flow


def require_flow_session(graph, flow_id: str, session_id: str) -> dict:
    # 用已有输入模型验证编号。占位标题正文不会写入任何状态。
    session_id = NoteDraft(session_id=session_id, title="归属检查", content="检查").session_id
    state = graph.get_state(config_for(flow_id)).values
    if not state:
        raise ValueError("流程不存在。")
    pending = PendingNote.model_validate(state["pending_note"])
    validate_snapshot(pending)
    if pending.draft.session_id != session_id:
        raise ValueError("当前会话不属于这笔审批流程。")
    return state


def checked_resume(graph, flow_id: str, session_id: str, approved: bool) -> dict:
    require_flow_session(graph, flow_id, session_id)
    return resume_flow(graph, flow_id, approved)


def checked_retry(graph, flow_id: str, session_id: str) -> dict:
    require_flow_session(graph, flow_id, session_id)
    return operations.retry_flow(graph, flow_id)

