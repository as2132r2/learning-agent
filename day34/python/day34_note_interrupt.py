"""Day34 第2步：待审批笔记的暂停、持久化与恢复。

python day34/python/day34_note_interrupt.py start note-flow-001
python day34/python/day34_note_interrupt.py status note-flow-001
python day34/python/day34_note_interrupt.py resume note-flow-001 yes
python day34/python/day34_note_interrupt.py resume note-flow-001 no
python day34/python/day34_note_interrupt.py demo

本步只保存工作流checkpoint，不写学习笔记文件。
流程编号与session_id是两个概念：一个会话可以发起多笔独立审批流程。
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from day34_note_contract import NoteDraft, PendingNote, prepare_note, validate_snapshot


DATABASE_PATH = Path(__file__).resolve().parents[1] / "database/day34_approval.db"


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


def approved_node(state: ApprovalState) -> dict:
    if state["approval"] is not True:
        raise ValueError("未批准不能进入批准分支。")
    return {"status": "approved", "result": "人工已批准；本步只记录决定，尚未写入笔记。"}


def rejected_node(state: ApprovalState) -> dict:
    return {"status": "rejected", "result": "人工已拒绝，没有写入笔记。"}


def build_graph(checkpointer):
    builder = StateGraph(ApprovalState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("approval", approval_node)
    builder.add_node("approved", approved_node)
    builder.add_node("rejected", rejected_node)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "approval")
    builder.add_conditional_edges(
        "approval", lambda state: "approved" if state["approval"] else "rejected",
        {"approved": "approved", "rejected": "rejected"},
    )
    builder.add_edge("approved", END)
    builder.add_edge("rejected", END)
    return builder.compile(checkpointer=checkpointer)


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


def show_status(graph, flow_id: str) -> None:
    snapshot = graph.get_state(config_for(flow_id))
    if not snapshot.values:
        print("没有找到该流程。")
        return
    print("流程编号：", flow_id)
    print("状态：", snapshot.values["status"])
    print("批准决定：", snapshot.values["approval"])
    print("下一节点：", snapshot.next or "无")
    for task in snapshot.tasks:
        for paused in task.interrupts:
            print("审批问题与内容：", paused.value)
    if snapshot.values.get("result"):
        print("结果：", snapshot.values["result"])


def demonstrate() -> None:
    with tempfile.TemporaryDirectory(prefix="day34-interrupt-") as directory:
        database = str(Path(directory) / "demo.db")
        draft = NoteDraft(session_id="xiaolin-capstone-001", title="RAG复习", content="先检索资料，再生成回答。")
        for flow_id, approved in (("demo-yes", True), ("demo-no", False)):
            with SqliteSaver.from_conn_string(database) as saver:
                graph = build_graph(saver)
                paused = start_flow(graph, flow_id, draft)
                assert paused.get("__interrupt__") and paused["approval"] is None
                original = paused["pending_note"]
                show_status(graph, flow_id)
            print("\n[已关闭连接，重新打开数据库恢复]")
            with SqliteSaver.from_conn_string(database) as saver:
                graph = build_graph(saver)
                result = resume_flow(graph, flow_id, approved)
                assert result["status"] == ("approved" if approved else "rejected")
                assert result["pending_note"] == original
                assert not graph.get_state(config_for(flow_id)).next
                show_status(graph, flow_id)
                try:
                    resume_flow(graph, flow_id, approved)
                except ValueError:
                    print("[预期拒绝] 重复审批已完成流程")
                else:
                    raise AssertionError("重复审批未被拒绝。")
    print("\n暂停、重建连接恢复、批准/拒绝、快照稳定、重复审批拦截：全部通过。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day34笔记审批暂停与恢复")
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start")
    start.add_argument("flow_id")
    start.add_argument("--session-id", default="xiaolin-capstone-001")
    status = commands.add_parser("status")
    status.add_argument("flow_id")
    resume = commands.add_parser("resume")
    resume.add_argument("flow_id")
    resume.add_argument("decision", choices=("yes", "no"))
    commands.add_parser("demo")
    args = parser.parse_args()
    print("Day34 第2步：笔记审批interrupt与恢复")
    try:
        if args.action == "demo":
            demonstrate()
            return
        config_for(args.flow_id)
        draft = None
        if args.action == "start":
            draft = NoteDraft(
                session_id=args.session_id, title=input("笔记标题："),
                content=input("笔记正文："),
            )
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(DATABASE_PATH)) as saver:
            graph = build_graph(saver)
            if args.action == "start":
                start_flow(graph, args.flow_id, draft)
            elif args.action == "resume":
                resume_flow(graph, args.flow_id, args.decision == "yes")
            show_status(graph, args.flow_id)
        print("数据库：", DATABASE_PATH)
        print("本步未写学习笔记；checkpoint已保存。")
    except (ValueError, OSError) as error:
        print("操作失败：", error)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
