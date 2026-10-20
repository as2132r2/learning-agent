"""Day34 第3步：人工批准后的笔记写入与拒绝隔离。

python day34/python/day34_note_writer.py demo
python day34/python/day34_note_writer.py start write-flow-001
python day34/python/day34_note_writer.py resume write-flow-001 yes
python day34/python/day34_note_writer.py status write-flow-001

依赖同目录前两步文件。本步使用独立checkpoint数据库，不修改第2步流程。
"""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from day34_note_contract import NOTE_ROOT, NoteDraft, PendingNote, validate_snapshot
from day34_note_interrupt import (
    approval_node, config_for, prepare_node, resume_flow, show_status, start_flow,
)


DATABASE_PATH = Path(__file__).resolve().parents[1] / "database/day34_note_writer.db"


class WriteState(TypedDict):
    draft: dict
    pending_note: dict
    approval: bool | None
    status: str
    result: str
    saved_file: str
    reused_existing: bool


def render_note(pending: PendingNote) -> bytes:
    """仅使用审批快照生成确定性字节，不添加会在重试时变化的时间戳。"""
    text = (
        f"# {pending.draft.title}\n\n"
        f"笔记编号：{pending.note_id}\n"
        f"所属会话：{pending.draft.session_id}\n"
        f"审批快照SHA-256：{pending.content_sha256}\n\n"
        f"{pending.draft.content}\n"
    )
    return text.encode("utf-8")


def verify_existing(target: Path, expected: bytes) -> None:
    """同一笔记编号只能对应相同内容，拒绝覆盖或跟随文件链接。"""
    metadata = target.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("已有目标必须是普通文件，不能是目录或符号链接。")
    if target.read_bytes() != expected:
        raise ValueError("同一笔记编号已存在不同内容，拒绝覆盖。")


def write_approved_note(state: WriteState, note_root: Path) -> tuple[Path, bool]:
    """本地可信工作流调用边界：批准、会话、快照均检查后才触碰目录。"""
    if state.get("approval") is not True or state.get("status") not in {"pending", "saved"}:
        raise ValueError("没有有效批准状态，禁止写入。")
    pending = PendingNote.model_validate(state["pending_note"])
    validate_snapshot(pending)
    original = NoteDraft.model_validate(state["draft"])
    if original != pending.draft:
        raise ValueError("原始草稿与待审批快照不一致，禁止写入。")

    root = note_root.resolve()
    session_directory = root / pending.draft.session_id
    if session_directory.is_symlink():
        raise ValueError("会话目录不能是符号链接。")
    if not session_directory.resolve().is_relative_to(root):
        raise ValueError("目标目录超出笔记根目录。")
    session_directory.mkdir(parents=True, exist_ok=True)
    target = session_directory / f"{pending.note_id}.md"
    data = render_note(pending)
    # 先完整写好同目录临时文件，再用硬链接以不覆盖方式原子发布。
    # 这依赖当前Ubuntu本地文件系统支持硬链接。
    descriptor, temporary_name = tempfile.mkstemp(prefix=".note-", dir=session_directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, target)
            reused = False
        except FileExistsError:
            verify_existing(target, data)
            reused = True
        directory_fd = os.open(session_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target, reused
    finally:
        temporary.unlink(missing_ok=True)


def build_graph(checkpointer, note_root: Path = NOTE_ROOT):
    def save_node(state: WriteState) -> dict:
        target, reused = write_approved_note(state, note_root)
        print("[save]", "相同内容已存在，幂等复用" if reused else "首次写入", target)
        return {
            "status": "saved", "saved_file": str(target),
            "reused_existing": reused, "result": f"批准的笔记已保存：{target}",
        }

    def reject_node(state: WriteState) -> dict:
        if state["approval"] is not False:
            raise ValueError("拒绝分支要求明确的False。")
        # 不调用写入函数，甚至不创建saved_notes目录。
        return {
            "status": "rejected", "saved_file": "", "reused_existing": False,
            "result": "人工已拒绝，没有创建笔记文件。",
        }

    def after_approval(state: WriteState) -> str:
        if state["approval"] is True:
            return "save"
        if state["approval"] is False:
            return "reject"
        raise ValueError("审批结果必须是明确的True或False。")

    builder = StateGraph(WriteState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("approval", approval_node)
    builder.add_node("save", save_node)
    builder.add_node("reject", reject_node)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "approval")
    builder.add_conditional_edges("approval", after_approval, {"save": "save", "reject": "reject"})
    builder.add_edge("save", END)
    builder.add_edge("reject", END)
    return builder.compile(checkpointer=checkpointer)


def demonstrate() -> None:
    with tempfile.TemporaryDirectory(prefix="day34-writer-") as directory:
        root = Path(directory)
        note_root = root / "saved_notes"
        database = str(root / "checkpoints.db")
        draft = NoteDraft(session_id="xiaolin", title="RAG学习笔记", content="先检索，再根据资料生成回答。")
        with SqliteSaver.from_conn_string(database) as saver:
            graph = build_graph(saver, note_root)
            start_flow(graph, "reject-001", draft)
            assert not note_root.exists()
            rejected = resume_flow(graph, "reject-001", False)
            assert rejected["status"] == "rejected" and not note_root.exists()
            print("[通过] 暂停和拒绝均未创建笔记目录")
            paused = start_flow(graph, "save-001", draft)
            assert paused["approval"] is None and not note_root.exists()
        with SqliteSaver.from_conn_string(database) as saver:
            graph = build_graph(saver, note_root)
            saved = resume_flow(graph, "save-001", True)
            target = Path(saved["saved_file"])
            assert saved["status"] == "saved" and target.is_file()
            pending = PendingNote.model_validate(saved["pending_note"])
            assert target.read_bytes() == render_note(pending)
            before = target.read_bytes()
            again, reused = write_approved_note(saved, note_root)
            assert again == target and reused and target.read_bytes() == before
            assert len(list(note_root.rglob("*.md"))) == 1
            print("[通过] 重连批准后保存；同一写入重试不产生第二份笔记")
            # 模拟目标内容发生冲突；只改临时测试目录文件。
            target.write_text("冲突内容", encoding="utf-8")
            try:
                write_approved_note(saved, note_root)
            except ValueError:
                assert target.read_text(encoding="utf-8") == "冲突内容"
                print("[通过] 不同内容冲突拒绝覆盖")
            else:
                raise AssertionError("冲突文件被错误覆盖。")
            unapproved = {**saved, "approval": None}
            try:
                write_approved_note(unapproved, root / "unauthorized")
            except ValueError:
                assert not (root / "unauthorized").exists()
                print("[通过] 未批准直接调用写入函数被拒绝")
            else:
                raise AssertionError("未批准写入未被拒绝。")
    print("\n全部演示通过；临时数据库与演示笔记自动清理。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day34批准后的笔记写入")
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start")
    start.add_argument("flow_id")
    start.add_argument("--session-id", default="xiaolin-capstone-001")
    resume = commands.add_parser("resume")
    resume.add_argument("flow_id")
    resume.add_argument("decision", choices=("yes", "no"))
    status = commands.add_parser("status")
    status.add_argument("flow_id")
    commands.add_parser("demo")
    args = parser.parse_args()
    print("Day34 第3步：批准写入与拒绝隔离")
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
            show_status(graph, args.flow_id)
        print("工作流数据库：", DATABASE_PATH)
    except (ValueError, OSError) as error:
        print("操作失败：", error)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
