"""Day34 第5步：会话归属检查与失败场景测试。

python day34/python/day34_session_checks.py

使用临时SQLite和临时笔记目录，不调用大模型。
checked_resume/checked_retry供下一步交互入口复用。
session_id归属检查不是用户认证，可信调用方仍需负责提供当前会话。
"""

from __future__ import annotations

import contextlib
import errno
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import day34_guarded_operations as operations
from day34_note_contract import NoteDraft, PendingNote, validate_snapshot
from day34_note_interrupt import config_for, resume_flow, start_flow


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


class SessionAndFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resources = contextlib.ExitStack()
        self.addCleanup(self.resources.close)
        directory = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="day34-cases-"))
        root = Path(directory)
        self.note_root = root / "notes"
        saver = self.resources.enter_context(SqliteSaver.from_conn_string(str(root / "checks.db")))
        self.graph = operations.build_graph(saver, self.note_root)
        self.alice = NoteDraft(session_id="alice", title="Alice笔记", content="Alice内容")
        self.bob = NoteDraft(session_id="bob", title="Bob笔记", content="Bob内容")

    def start(self, flow_id: str, draft: NoteDraft | None = None) -> dict:
        return start_flow(self.graph, flow_id, draft or self.alice)

    def files(self) -> list[Path]:
        return sorted(self.note_root.rglob("*.md"))

    def test_01_cross_session_resume_is_rejected_without_mutation(self) -> None:
        self.start("alice-flow")
        before = self.graph.get_state(config_for("alice-flow")).values
        with self.assertRaisesRegex(ValueError, "不属于"):
            checked_resume(self.graph, "alice-flow", "bob", True)
        self.assertEqual(before, self.graph.get_state(config_for("alice-flow")).values)
        self.assertFalse(self.note_root.exists())

    def test_02_two_sessions_save_to_separate_directories(self) -> None:
        self.start("alice-flow")
        self.start("bob-flow", self.bob)
        alice = checked_resume(self.graph, "alice-flow", "alice", True)
        bob = checked_resume(self.graph, "bob-flow", "bob", True)
        self.assertEqual(Path(alice["saved_file"]).parent.name, "alice")
        self.assertEqual(Path(bob["saved_file"]).parent.name, "bob")
        self.assertNotEqual(alice["pending_note"]["note_id"], bob["pending_note"]["note_id"])
        self.assertEqual(len(self.files()), 2)

    def test_03_rejection_creates_no_files_and_cannot_retry(self) -> None:
        self.start("rejected-flow")
        result = checked_resume(self.graph, "rejected-flow", "alice", False)
        self.assertEqual(result["status"], "rejected")
        with self.assertRaises(ValueError):
            checked_retry(self.graph, "rejected-flow", "alice")
        self.assertFalse(self.note_root.exists())

    def test_04_completed_flow_cannot_be_approved_twice(self) -> None:
        self.start("finished-flow")
        result = checked_resume(self.graph, "finished-flow", "alice", True)
        before = Path(result["saved_file"]).read_bytes()
        with self.assertRaises(ValueError):
            checked_resume(self.graph, "finished-flow", "alice", True)
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(Path(result["saved_file"]).read_bytes(), before)

    def test_05_wrong_approval_hash_is_rejected_at_interrupt(self) -> None:
        paused = self.start("bad-decision")
        pending = paused["pending_note"]
        decision = {
            "session_id": "alice", "note_id": pending["note_id"],
            "content_sha256": "0" * 64, "approved": True,
        }
        with self.assertRaisesRegex(ValueError, "不匹配"):
            self.graph.invoke(Command(resume=decision), config=config_for("bad-decision"))
        self.assertFalse(self.note_root.exists())

    def test_06_retry_limit_and_cross_session_retry(self) -> None:
        paused = self.start("busy-flow")
        with patch.object(operations, "write_approved_note", side_effect=OSError(errno.EBUSY, "模拟存储忙")):
            result = checked_resume(self.graph, "busy-flow", "alice", True)
            self.assertTrue(result["retryable"])
            with self.assertRaisesRegex(ValueError, "不属于"):
                checked_retry(self.graph, "busy-flow", "bob")
            for count in (1, 2):
                result = checked_retry(self.graph, "busy-flow", "alice")
                self.assertEqual(result["retry_count"], count)
                self.assertEqual(result["pending_note"], paused["pending_note"])
            with self.assertRaisesRegex(ValueError, "上限"):
                checked_retry(self.graph, "busy-flow", "alice")
        self.assertFalse(self.note_root.exists())

    def test_07_permission_error_does_not_become_success(self) -> None:
        self.start("denied-flow")
        with patch.object(operations, "write_approved_note", side_effect=PermissionError(errno.EACCES, "模拟权限不足")):
            result = checked_resume(self.graph, "denied-flow", "alice", True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "PermissionError")
        self.assertFalse(result["retryable"])
        with self.assertRaises(ValueError):
            checked_retry(self.graph, "denied-flow", "alice")
        self.assertEqual(self.files(), [])

    def test_08_changed_draft_is_rejected_before_write(self) -> None:
        self.start("changed-flow")
        saved = checked_resume(self.graph, "changed-flow", "alice", True)
        altered = {**saved, "draft": {**saved["draft"], "session_id": "bob"}}
        with self.assertRaisesRegex(ValueError, "不一致"):
            operations.write_approved_note(altered, self.note_root)
        self.assertEqual(len(self.files()), 1)
        self.assertFalse((self.note_root / "bob").exists())

    def test_09_unknown_flow_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不存在"):
            checked_resume(self.graph, "missing-flow", "alice", True)
        self.assertFalse(self.note_root.exists())


def main() -> None:
    print("Day34 第5步：会话隔离与失败场景验证", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SessionAndFailureTests)
    # 保存节点输出只在失败时查看；终端保留清晰的逐项测试结果。
    with contextlib.redirect_stdout(io.StringIO()):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"测试数={result.testsRun}，失败={len(result.failures)}，错误={len(result.errors)}")
    print("当前会话 → 流程归属检查 → 快照检查 → 恢复/重试 → 写入边界")
    print("归属检查不等于身份认证；测试仅使用临时数据库和笔记目录。")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
