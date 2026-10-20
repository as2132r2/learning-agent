"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
NOTE_ROOT = Path("runtime/saved_notes")


class NoteDraft(BaseModel):
    """用户可提交的业务内容，没有文件路径和审批结果字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)
    session_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("title", "content")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\t" for character in value):
            raise ValueError("笔记不能包含NUL等控制字符。")
        return value


class PendingNote(BaseModel):
    """由程序创建的待审批快照；不是已经获准写入的笔记。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    note_id: str = Field(pattern=r"^note-[0-9a-f]{32}$")
    draft: NoteDraft
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending"] = "pending"


def snapshot_hash(note_id: str, draft: NoteDraft) -> str:
    """把请求编号、所属会话、标题、正文共同绑定到审批快照。"""
    data = {"note_id": note_id, **draft.model_dump()}
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_note(draft: NoteDraft) -> PendingNote:
    note_id = "note-" + uuid.uuid4().hex
    return PendingNote(
        note_id=note_id, draft=draft,
        content_sha256=snapshot_hash(note_id, draft),
    )


def validate_snapshot(pending: PendingNote) -> None:
    """结构校验后再检查业务一致性，未来恢复checkpoint时继续复用。"""
    if pending.content_sha256 != snapshot_hash(pending.note_id, pending.draft):
        raise ValueError("待审批快照哈希不一致，需要重新生成审批提案。")

