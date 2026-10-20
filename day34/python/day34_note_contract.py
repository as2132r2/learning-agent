"""Day34 第1步：待审批学习笔记模型与边界。

python day34/python/day34_note_contract.py
python day34/python/day34_note_contract.py --demo

只在终端预览待审批对象；不调用模型、不创建笔记文件、不模拟已批准。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


DAY34_DIR = Path(__file__).resolve().parents[1]
NOTE_ROOT = DAY34_DIR / "saved_notes"


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


def planned_destination(pending: PendingNote) -> Path:
    """只计算路径。文件名由系统编号生成，标题不能变成任意写入路径。"""
    validate_snapshot(pending)
    return NOTE_ROOT / pending.draft.session_id / f"{pending.note_id}.md"


def print_preview(pending: PendingNote) -> None:
    validate_snapshot(pending)
    print("\n待审批对象：")
    print(pending.model_dump_json(indent=2))
    print("\n预期写入位置：", planned_destination(pending))
    print("当前状态：pending；尚未审批，尚未写入。")
    print("审批绑定内容：note_id + session_id + title + content + 快照哈希")


def demonstrate_validation() -> None:
    """检查空白、越界路径输入和伪造审批字段都被拒绝。"""
    base = {"session_id": "xiaolin-capstone-001", "title": "RAG复习", "content": "先检索，再生成。"}
    invalid_cases = [
        ("空白正文", {**base, "content": "   "}),
        ("非法会话路径", {**base, "session_id": "../../other"}),
        ("提交端伪造批准", {**base, "approved": True}),
    ]
    for label, payload in invalid_cases:
        try:
            NoteDraft.model_validate(payload)
        except ValidationError:
            print(f"[预期拒绝] {label}")
        else:
            raise AssertionError(f"未能拦截：{label}")
    pending = prepare_note(NoteDraft(**base))
    # 修改为仍然符合结构约束的内容，保留旧哈希：一致性检查必须失败。
    changed = pending.model_dump()
    changed["draft"]["content"] = "修改后的正文。"
    try:
        validate_snapshot(PendingNote.model_validate(changed))
    except ValueError:
        print("[预期拒绝] 正文变化但仍沿用旧快照哈希")
    else:
        raise AssertionError("未能识别快照变化。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day34待审批笔记约定")
    parser.add_argument("--demo", action="store_true", help="固定样例与边界验证")
    parser.add_argument("--session-id", default="xiaolin-capstone-001")
    args = parser.parse_args()
    print("Day34 第1步：学习笔记数据模型与写入边界")
    try:
        if args.demo:
            draft = NoteDraft(
                session_id=args.session_id, title="RAG的基本流程",
                content="RAG是检索增强生成：先从知识库检索资料，再把资料和问题交给大语言模型。",
            )
        else:
            draft = NoteDraft(
                session_id=args.session_id, title=input("笔记标题："),
                content=input("笔记正文（本步先输入一行）："),
            )
        print_preview(prepare_note(draft))
        if args.demo:
            print("\n边界验证：")
            demonstrate_validation()
    except (ValidationError, ValueError) as error:
        print("输入或快照无效：", error)
        raise SystemExit(2) from error
    print("\n数据流：原始标题/正文 → NoteDraft → prepare_note → PendingNote → 终端预览")
    print("下一步把PendingNote交给LangGraph interrupt，等待人工决定。")
    print("快照哈希用于绑定内容版本，不是身份认证或数字签名。")


if __name__ == "__main__":
    main()
