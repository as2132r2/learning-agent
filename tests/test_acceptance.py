import contextlib
import errno
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import ValidationError
from learning_agent import app, operations
from learning_agent.notes import NoteDraft, PendingNote
from learning_agent.writer import render_note
from learning_agent.approval import config_for as flow_config, start_flow
from learning_agent.session_checks import checked_resume, checked_retry


def messages_evidence(state: dict) -> list[dict]:
    return [{
        "type": type(message).__name__, "content": str(message.content),
        "tool_calls": getattr(message, "tool_calls", []),
        "tool_call_id": getattr(message, "tool_call_id", None),
    } for message in state.get("messages", [])]


class GraduationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {}
        self.resources = contextlib.ExitStack()
        self.addCleanup(self.resources.close)
        directory = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="day35-fake-"))
        self.root = Path(directory)
        self.notes = self.root / "notes"
        self.open_graphs()
        self.addCleanup(lambda: self.connections.close())

    def open_graphs(self):
        self.connections = contextlib.ExitStack()
        chat_saver = self.connections.enter_context(SqliteSaver.from_conn_string(str(self.root / "chat.db")))
        self.saver = self.connections.enter_context(SqliteSaver.from_conn_string(str(self.root / "approvals.db")))
        self.chat = app.build_chat_graph(chat_saver, True)
        self.approvals = operations.build_graph(self.saver, self.notes)

    def turn(self, text, session="session-a"):
        return app.run_turn(self.chat, session, text)

    def start_note(self, flow="flow-001"):
        draft = NoteDraft(session_id="session-a", title="RAG复习", content="先检索，再根据资料生成回答。")
        return start_flow(self.approvals, flow, draft)

    def files(self):
        return list(self.notes.rglob("*.md"))

    def test_G01(self):
        self.turn("我叫小林。")
        state = self.turn("我叫什么名字？")
        self.evidence = {"answer": state["answer"], "messages": messages_evidence(state)}
        self.assertIn("小林", state["answer"])
        self.assertEqual([type(m).__name__ for m in state["messages"]], ["HumanMessage", "AIMessage"] * 2)

    def test_G02(self):
        first = self.turn("我叫小林。")
        second = self.turn("我叫什么名字？", "session-b")
        self.evidence = {"session_a": messages_evidence(first), "session_b": messages_evidence(second)}
        self.assertNotIn("小林", second["answer"])
        self.assertTrue(all("我叫小林" not in str(m.content) for m in second["messages"]))

    def test_G03(self):
        state = self.turn("请计算(20+4)*3。")
        calls = [call for m in state["messages"] if isinstance(m, AIMessage) for call in m.tool_calls]
        results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        self.evidence = {"messages": messages_evidence(state), "answer": state["answer"]}
        self.assertEqual([(c["name"], c["args"]) for c in calls], [
            ("add", {"a": 20, "b": 4}), ("multiply", {"a": 24.0, "b": 3}),
        ])
        self.assertEqual([m.tool_call_id for m in results], [c["id"] for c in calls])
        self.assertEqual([float(m.content) for m in results], [24, 72])
        self.assertIn("72", state["answer"])

    def test_G04(self):
        state = self.turn("根据课程资料解释RAG。")
        refs = [int(n) for n in re.findall(r"\[资料(\d+)\]", state["answer"])]
        self.evidence = {"answer": state["answer"], "evidence": state["evidence"], "citations": refs}
        self.assertTrue(state["evidence"] and refs)
        self.assertTrue(all(1 <= n <= len(state["evidence"]) for n in refs))
        self.assertIn("检索增强生成", state["answer"])

    def test_G05(self):
        # 注入可计数的Fake回答模型，真实执行knowledge节点的空证据分支。
        calculator, answer = app.create_models(True)
        with patch.object(app, "create_models", return_value=(calculator, answer)):
            graph = app.build_chat_graph(self.chat.checkpointer, True)
        state = app.run_turn(graph, "empty", "根据资料解释未收录课程。")
        self.evidence = {"answer": state["answer"], "evidence": state["evidence"], "answer_model_calls": answer.calls}
        self.assertEqual(state["evidence"], [])
        self.assertIn("不足", state["answer"])
        self.assertEqual(answer.calls, 0)

    def test_G06(self):
        inputs = iter(["/note", "RAG复习", "先检索，再根据资料生成回答。", "/approve latest", "exit"])
        self.assertFalse(self.notes.exists())
        self.assertTrue(app.run_console(self.chat, self.approvals, self.saver, "session-a", lambda _: next(inputs)))
        flow, state = app.owned_flows(self.saver, "session-a")[0]
        pending = PendingNote.model_validate(state["pending_note"])
        content = Path(state["saved_file"]).read_bytes()
        chat = self.chat.get_state(app.config_for("session-a")).values
        self.evidence = {"flow_id": flow, "approval_state": state, "file_utf8": content.decode(), "messages": messages_evidence(chat)}
        self.assertEqual(state["status"], "saved")
        self.assertEqual(content, render_note(pending))
        self.assertTrue(any("已保存" in str(m.content) for m in chat["messages"]))
        self.assertEqual(len(self.files()), 1)
        # 分离验证暂停时确实没有写入另一笔文件。
        self.start_note("still-pending")
        self.assertEqual(len(self.files()), 1)

    def test_G07(self):
        self.start_note()
        result = checked_resume(self.approvals, "flow-001", "session-a", False)
        with self.assertRaises(ValueError):
            checked_retry(self.approvals, "flow-001", "session-a")
        self.evidence = {"state": result, "file_count": len(self.files()), "retry_rejected": True}
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(self.notes.exists())

    def test_G08(self):
        self.turn("我叫小林。")
        paused = self.start_note()
        old_pending = paused["pending_note"]
        self.connections.close()
        self.open_graphs()
        restored = self.turn("我叫什么名字？")
        before = self.approvals.get_state(flow_config("flow-001")).values["pending_note"]
        saved = checked_resume(self.approvals, "flow-001", "session-a", True)
        self.evidence = {"recovery_method": "close_and_recreate_connections_and_graphs", "answer": restored["answer"], "before": old_pending, "restored": before, "final_status": saved["status"]}
        self.assertIn("小林", restored["answer"])
        self.assertEqual(old_pending, before)
        self.assertEqual(old_pending, saved["pending_note"])
        self.assertEqual(saved["status"], "saved")

    def test_G09(self):
        rejected = []
        for args in ({"a": "20", "b": 4}, {"a": True, "b": 4}, {"a": float("inf"), "b": 4}, {"a": 1, "b": 2, "admin": True}, {"a": 1e13, "b": 4}):
            with self.assertRaises(ValidationError):
                operations.guarded_add.invoke(args)
            rejected.append(repr(args))
        result = operations.guarded_multiply.invoke({"a": operations.guarded_add.invoke({"a": 20, "b": 4}), "b": 3})
        self.evidence = {"rejected_inputs": rejected, "valid_result": result}
        self.assertEqual(result, 72)

    def test_G10(self):
        paused = self.start_note()
        with patch.object(operations, "write_approved_note", side_effect=OSError(errno.EBUSY, "模拟存储忙")):
            failed = checked_resume(self.approvals, "flow-001", "session-a", True)
        self.assertEqual(failed["status"], "failed")
        saved = checked_retry(self.approvals, "flow-001", "session-a")
        target, reused = operations.write_approved_note(saved, self.notes)
        self.assertTrue(reused)
        self.assertEqual(saved["pending_note"], paused["pending_note"])
        self.assertEqual(len(self.files()), 1)
        target.write_text("模拟冲突", encoding="utf-8")
        with self.assertRaises(ValueError):
            operations.write_approved_note(saved, self.notes)
        self.assertEqual(target.read_text(encoding="utf-8"), "模拟冲突")
        self.evidence = {"failure": failed, "recovery": saved, "reused": reused, "file_count": 1, "conflict_rejected": True}

    def test_G11(self):
        paused = self.start_note()
        for operation in (lambda: checked_resume(self.approvals, "flow-001", "session-b", True), lambda: checked_retry(self.approvals, "flow-001", "session-b")):
            with self.assertRaises(ValueError):
                operation()
        pending = paused["pending_note"]
        bad = {"session_id": "session-a", "note_id": pending["note_id"], "content_sha256": "0" * 64, "approved": True}
        with self.assertRaises(ValueError):
            self.approvals.invoke(Command(resume=bad), config=flow_config("flow-001"))
        self.assertEqual(self.files(), [])
        self.start_note("completed")
        checked_resume(self.approvals, "completed", "session-a", True)
        with self.assertRaises(ValueError):
            checked_resume(self.approvals, "completed", "session-a", True)
        self.assertEqual(len(self.files()), 1)
        self.evidence = {"cross_session_resume_rejected": True, "cross_session_retry_rejected": True, "wrong_hash_rejected": True, "duplicate_approval_rejected": True, "final_file_count": 1}

    def test_G12(self):
        self.start_note("permission")
        with patch.object(operations, "write_approved_note", side_effect=PermissionError(errno.EACCES, "模拟权限不足")):
            denied = checked_resume(self.approvals, "permission", "session-a", True)
        with self.assertRaises(ValueError):
            checked_retry(self.approvals, "permission", "session-a")
        self.start_note("busy")
        with patch.object(operations, "write_approved_note", side_effect=OSError(errno.EBUSY, "模拟存储忙")):
            checked_resume(self.approvals, "busy", "session-a", True)
            for count in (1, 2):
                state = checked_retry(self.approvals, "busy", "session-a")
                self.assertEqual(state["retry_count"], count)
            with self.assertRaises(ValueError):
                checked_retry(self.approvals, "busy", "session-a")
        self.evidence = {"permission_failure": denied, "retry_count": state["retry_count"], "third_retry_rejected": True, "files": len(self.files())}
        self.assertFalse(denied["retryable"])
        self.assertEqual(self.files(), [])

