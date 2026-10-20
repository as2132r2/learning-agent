"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import TypedDict
from .notes import NoteDraft, PendingNote, validate_snapshot


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

