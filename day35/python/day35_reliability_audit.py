"""Day35 第4步：验收证据核查与独立进程恢复测试。

python day35/python/day35_reliability_audit.py

读取前两份报告，使用两个独立子进程与临时数据库测试恢复。
Fake模型不发网络请求；本步不重新运行真实模型验收。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ("day33/python", "day34/python"):
    sys.path.insert(0, str(ROOT / name))

from langgraph.checkpoint.sqlite import SqliteSaver
import day34_learning_assistant as app
import day34_guarded_operations as operations
from day34_note_contract import NoteDraft
from day34_note_interrupt import start_flow, config_for as flow_config
from day34_note_writer import render_note
from day34_note_contract import PendingNote
from day34_session_checks import checked_resume
from day35_acceptance_plan import CASES, DAY35_DIR, PLAN_VERSION, save_plan


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def worker(phase: str, directory: Path) -> dict:
    """子进程函数。所有待恢复数据只通过磁盘传递。"""
    with contextlib.ExitStack() as stack:
        chat_saver = stack.enter_context(SqliteSaver.from_conn_string(str(directory / "chat.db")))
        note_saver = stack.enter_context(SqliteSaver.from_conn_string(str(directory / "notes.db")))
        chat = app.build_chat_graph(chat_saver, True)
        notes = operations.build_graph(note_saver, directory / "saved_notes")
        if phase == "prepare":
            app.run_turn(chat, "restarted-a", "我叫小林。")
            paused = start_flow(notes, "restart-note", NoteDraft(
                session_id="restarted-a", title="独立进程恢复", content="只有批准才保存这份笔记。",
            ))
            require(not (directory / "saved_notes").exists(), "批准前出现笔记目录。")
            return {
                "pid": os.getpid(), "phase": phase,
                "pending": paused["pending_note"], "approval": paused["approval"],
                "next": list(notes.get_state(flow_config("restart-note")).next),
                "file_count": 0,
            }
        before = notes.get_state(flow_config("restart-note")).values
        require(before["status"] == "pending", "新进程未读到待审批状态。")
        known = app.run_turn(chat, "restarted-a", "我叫什么名字？")
        other = app.run_turn(chat, "restarted-b", "我叫什么名字？")
        saved = checked_resume(notes, "restart-note", "restarted-a", True)
        target = Path(saved["saved_file"])
        contents = target.read_bytes()
        require(contents == render_note(PendingNote.model_validate(saved["pending_note"])), "文件内容与审批快照不一致。")
        require(saved["status"] == "saved", "没有进入saved状态。")
        return {
            "pid": os.getpid(), "phase": phase,
            "before_pending": before["pending_note"], "after_pending": saved["pending_note"],
            "answer": known["answer"], "other_session_answer": other["answer"],
            "status": saved["status"], "file_utf8": contents.decode("utf-8"),
            "file_sha256": hashlib.sha256(contents).hexdigest(),
            "file_count": len(list((directory / "saved_notes").rglob("*.md"))),
        }


def run_worker(phase: str, directory: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--workspace", str(directory)],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"子进程{phase}失败：{completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def audit_reports() -> dict:
    reports = {}
    for mode in ("fake", "live"):
        path = DAY35_DIR / "reports" / f"{mode}_acceptance.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        require(report["plan_version"] == PLAN_VERSION, f"{mode}报告计划版本不匹配。")
        require(report["mode"] == mode, f"{mode}报告模式不匹配。")
        rows = report["cases"]
        expected = {case.case_id for case in CASES if mode in case.validation_modes}
        actual = {row["case_id"] for row in rows}
        require(len(rows) == len(actual) and actual == expected, f"{mode}报告重复或缺少场景。")
        require(report["status"] == "passed" and all(row["status"] == "passed" for row in rows), f"{mode}存在未通过项。")
        require(all(row.get("evidence") for row in rows), f"{mode}存在缺失证据的通过项。")
        reports[mode] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "case_ids": sorted(actual), "status": report["status"],
        }
        if mode == "fake":
            cases = {row["case_id"]: row["evidence"] for row in rows}
            require(cases["G07"]["state"]["status"] == "rejected" and cases["G07"]["file_count"] == 0, "拒绝不写入证据不符。")
            require(cases["G10"]["failure"]["status"] == "failed" and cases["G10"]["reused"], "故障或幂等证据不符。")
            require(cases["G12"]["retry_count"] == 2 and cases["G12"]["third_retry_rejected"], "重试上限证据不符。")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Day35可靠性与持久化核查")
    parser.add_argument("--worker", choices=("prepare", "recover"), help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.workspace is None:
            parser.error("子进程缺少workspace。")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = worker(args.worker, args.workspace)
        result["execution_log"] = captured.getvalue()
        print(json.dumps(result, ensure_ascii=False))
        return
    report = {
        "plan_version": PLAN_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "checks": [],
    }
    try:
        report["source_reports"] = audit_reports()
        report["checks"].append({"name": "report_coverage_and_failure_evidence", "status": "passed"})
        with tempfile.TemporaryDirectory(prefix="day35-process-") as directory:
            prepared = run_worker("prepare", Path(directory))
            recovered = run_worker("recover", Path(directory))
            report["process_evidence"] = {"prepare": prepared, "recover": recovered}
            require(prepared["pid"] != recovered["pid"], "未使用两个独立子进程。")
            require(prepared["pending"] == recovered["before_pending"] == recovered["after_pending"], "进程重启后快照改变。")
            require("小林" in recovered["answer"] and "小林" not in recovered["other_session_answer"], "重启后记忆或隔离不正确。")
            require(recovered["file_count"] == 1, "批准写入数量不正确。")
        report["checks"].append({"name": "independent_process_recovery", "status": "passed"})
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report["known_limits"] = [
        "本机项目：会话归属检查不代替用户认证。",
        "路由仍为关键词规则，引用编号检查不等于语义事实审计。",
        "聊天与审批数据库没有跨库原子事务，需以审批状态和文件核对保存结果。",
        "本次独立进程恢复使用Fake模型；真实模型恢复结果见live报告。",
    ]
    output = DAY35_DIR / "reports/reliability_audit.json"
    save_plan(output, report)
    print("Day35 第4步：可靠性核查", report["status"])
    for check in report["checks"]:
        print(check["name"], check["status"])
    if "process_evidence" in report:
        print("独立进程PID：", report["process_evidence"]["prepare"]["pid"], "→", report["process_evidence"]["recover"]["pid"])
    if "error" in report:
        print(report["error"])
    print("报告：", output)
    print("临时数据库已清理，恢复前后快照、文件内容和日志保留在报告。")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
