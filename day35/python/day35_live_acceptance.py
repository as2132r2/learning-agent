"""Day35 第3步：真实DeepSeek与真实课程索引验收。

python day35/python/day35_live_acceptance.py

将产生真实模型请求与Token费用。每次运行使用独立数据库，报告不包含密钥。
真实结果按passed/failed/blocked记录，不使用Fake兜底。
"""

from __future__ import annotations

import io
import contextlib
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in ("day33/python", "day34/python"):
    sys.path.insert(0, str(ROOT / directory))

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import APIError

import day34_learning_assistant as app
import day34_guarded_operations as operations
from day34_note_contract import NoteDraft
from day34_note_interrupt import config_for as flow_config, start_flow
from day34_session_checks import checked_resume
from day35_acceptance_plan import CASES, DAY35_DIR, PLAN_VERSION, save_plan


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def message_records(state: dict) -> list[dict]:
    return [{
        "type": type(message).__name__, "content": str(message.content),
        "tool_calls": getattr(message, "tool_calls", []),
        "tool_call_id": getattr(message, "tool_call_id", None),
    } for message in state.get("messages", [])]


class LiveCases:
    def __init__(self, directory: Path):
        self.directory = directory
        self.evidence = {}

    @contextlib.contextmanager
    def graphs(self):
        with contextlib.ExitStack() as stack:
            chat_saver = stack.enter_context(SqliteSaver.from_conn_string(str(self.directory / "chat.db")))
            note_saver = stack.enter_context(SqliteSaver.from_conn_string(str(self.directory / "approvals.db")))
            yield app.build_chat_graph(chat_saver, False), operations.build_graph(note_saver, self.directory / "notes")

    def G01(self):
        with self.graphs() as (chat, _):
            app.run_turn(chat, "g01-a", "我叫小林。")
            state = app.run_turn(chat, "g01-a", "我叫什么名字？请简短回答。")
            self.evidence = {"answer": state["answer"], "messages": message_records(state)}
            require("小林" in state["answer"], "同会话没有正确回答姓名。")
            require(len(state["messages"]) == 4, "两轮聊天消息数不符。")

    def G02(self):
        with self.graphs() as (chat, _):
            a = app.run_turn(chat, "g02-a", "我叫小林。")
            b = app.run_turn(chat, "g02-b", "我叫什么名字？不知道就直接说明。")
            self.evidence = {"session_a": message_records(a), "session_b": message_records(b)}
            require(all("小林" not in str(m.content) for m in b["messages"]), "新会话出现了另一会话的姓名。")

    def G03(self):
        with self.graphs() as (chat, _):
            state = app.run_turn(chat, "g03", "请计算(20+4)*3，必须按先加后乘调用工具。")
            calls = [c for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
            outputs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
            self.evidence = {"answer": state["answer"], "messages": message_records(state)}
            require([c["name"] for c in calls] == ["add", "multiply"], "未按要求调用add和multiply。")
            require([m.tool_call_id for m in outputs] == [c["id"] for c in calls], "工具结果ID与请求不匹配。")
            require([float(m.content) for m in outputs] == [24, 72], "工具结果不是24、72。")
            require(set(calls[0]["args"].values()) == {20, 4} and set(calls[1]["args"].values()) == {24, 3}, "工具参数不符。")
            require("72" in state["answer"], "最终回答不包含72。")

    def G04(self):
        with self.graphs() as (chat, _):
            state = app.run_turn(chat, "g04", "根据课程资料解释RAG。")
            refs = [int(n) for n in re.findall(r"\[资料(\d+)\]", state["answer"])]
            self.evidence = {"answer": state["answer"], "evidence": state["evidence"], "citations": refs}
            require(bool(state["evidence"]), "真实检索没有合格证据。")
            require(bool(refs) and all(1 <= n <= len(state["evidence"]) for n in refs), "引用缺失或越界。")
            require("检索增强生成" in state["answer"], "回答没有解释RAG中文含义。")

    def G08(self):
        with self.graphs() as (chat, notes):
            app.run_turn(chat, "g08", "我叫小林。")
            paused = start_flow(notes, "g08-note", NoteDraft(
                session_id="g08", title="真实验收恢复", content="这份笔记在重新打开数据库后批准。",
            ))
            original = paused["pending_note"]
            self.evidence = {"original_pending": original}
        # 重新建立两张图、两个连接；不冒充独立进程重启。
        with self.graphs() as (chat, notes):
            state = app.run_turn(chat, "g08", "我叫什么名字？请简短回答。")
            recovered = notes.get_state(flow_config("g08-note")).values["pending_note"]
            require(recovered == original, "恢复后审批快照改变。")
            saved = checked_resume(notes, "g08-note", "g08", True)
            self.evidence.update({"answer": state["answer"], "recovered_pending": recovered, "approval_result": saved, "recovery_method": "recreated_graphs_and_connections"})
            require("小林" in state["answer"], "重新连接后未恢复姓名记忆。")
            require(saved["status"] == "saved" and Path(saved["saved_file"]).is_file(), "原提案恢复后未保存。")


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = DAY35_DIR / "runtime" / ("live-" + run_id)
    directory.mkdir(parents=True, exist_ok=False)
    runner = LiveCases(directory)
    records = []
    print("Day35 真实模型验收：使用DeepSeek和已有课程索引", flush=True)
    print("独立运行目录：", directory, flush=True)
    for case in CASES:
        if "live" not in case.validation_modes:
            continue
        started = time.perf_counter()
        runner.evidence = {}
        output = io.StringIO()
        status, detail = "passed", ""
        try:
            with contextlib.redirect_stdout(output):
                getattr(runner, case.case_id)()
        except (APIError, OSError, ImportError, sqlite3.Error) as error:
            status = "blocked"
            # 原始SDK异常可能含连接配置，不将其全文写入报告。
            detail = f"{type(error).__name__}：依赖、网络、供应商或存储问题阻止验证。"
        except Exception as error:
            status = "failed"
            detail = str(error) if isinstance(error, AssertionError) else f"{type(error).__name__}：执行或返回结构不符合预期。"
        records.append({
            "case_id": case.case_id, "capability": case.capability, "status": status,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "detail": detail, "evidence": runner.evidence, "execution_log": output.getvalue(),
        })
        print(case.case_id, case.capability, status, detail, flush=True)
        report = {
            "plan_version": PLAN_VERSION, "mode": "live", "run_id": run_id,
            "runtime_directory": str(directory),
            "cases": records,
            "remaining": [c.case_id for c in CASES if "live" in c.validation_modes and c.case_id not in {r['case_id'] for r in records}],
            "status": "running",
        }
        save_plan(directory / "live_acceptance.json", report)
        if status == "blocked":
            # 避免供应商不可用时继续重复请求；未执行项目明确记not_run。
            break
    for case in CASES:
        if "live" in case.validation_modes and case.case_id not in {r["case_id"] for r in records}:
            records.append({"case_id": case.case_id, "capability": case.capability, "status": "not_run", "detail": "前置环境故障，停止后续真实请求。", "evidence": {}})
    report.update(cases=records, remaining=[], finished_at=datetime.now(timezone.utc).isoformat())
    report["status"] = "passed" if all(r["status"] == "passed" for r in records) else ("failed" if any(r["status"] == "failed" for r in records) else "blocked")
    save_plan(directory / "live_acceptance.json", report)
    output_path = DAY35_DIR / "reports/live_acceptance.json"
    save_plan(output_path, report)
    print("最终结果：", report["status"], "报告：", output_path)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
