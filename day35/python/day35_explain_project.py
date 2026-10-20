"""Day35 第6步：读取验收证据，练习独立讲解。

python day35/python/day35_explain_project.py
python day35/python/day35_explain_project.py --answers
python day35/python/day35_explain_project.py --practice

不调用模型。practice保存用户原文，不自动判定其理解程度。
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone

from day35_acceptance_plan import DAY35_DIR, save_plan


QUESTIONS = (
    (
        "Q01", "用户输入‘请计算(20+4)*3’，从CLI到最终回复经过哪些函数和节点？",
        "最终入口转调Day34.main，创建两份SqliteSaver；run_console把普通问题交给run_turn，"
        "UserRequest校验后invoke；router选择calculator；模型请求add，ToolNode返回24；"
        "模型请求multiply，ToolNode返回72；模型给最终回答，tools_condition转END，checkpoint保存状态。",
    ),
    (
        "Q02", "模型请求工具和工具实际执行有什么区别？为什么一次计算有6条消息？",
        "tool_calls是AIMessage里的调用提议，本机ToolNode才执行Python函数。"
        "用户消息1条、两次工具提议2条、两次ToolMessage结果2条、最终AI回复1条，共6条。"
        "tool_call_id将结果与对应请求关联。",
    ),
    (
        "Q03", "关闭程序后，姓名和待审批笔记为什么还能恢复？三个编号分别有什么用途？",
        "SQLite保存checkpoint。聊天thread_id使用session_id；审批thread_id使用flow_id；"
        "note_id决定笔记文件名并参与快照绑定。同一会话可以有多笔审批；"
        "不能把session_id隔离当成已实现用户身份认证。",
    ),
    (
        "Q04", "RAG回答模型收到什么？哪些内容进入聊天历史？没有资料时怎么办？",
        "检索与RRF、精排筛选后，将本轮问题和合格证据交给回答模型；"
        "最终知识节点没有把整段聊天历史用于检索改写。合格回答或拒答提示写入AIMessage；"
        "证据作为独立状态字段保存，不直接追加成聊天消息。证据为空跳过回答模型；"
        "引用编号有效不等于已做语义事实审计。",
    ),
    (
        "Q05", "保存笔记时，批准决定如何对应具体内容？为什么prepare与save要分开？",
        "prepare生成note_id和内容快照，approval中的interrupt展示内容并暂停。"
        "恢复决定携带session_id、note_id和哈希，全部匹配才继续；"
        "节点恢复会从开头重放，因此随机准备放在前置节点，写入放在批准后的save。"
        "CLI根据实际审批结果调用record_event，而不是让模型猜测保存成功。",
    ),
    (
        "Q06", "文件已写入但checkpoint尚未确认就中断，怎样避免重复？什么错误能重试？",
        "同一note_id生成固定字节：已有相同内容复用，不同内容拒绝覆盖。"
        "临时errno可人工重试最多2次，重试仍绑定原批准快照；权限不足与快照冲突不直接重试。"
        "聊天与审批数据库之间没有跨库事务，保存真相需核对审批状态和文件。",
    ),
)


def show_evidence() -> None:
    path = DAY35_DIR / "reports/fake_acceptance.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in report["cases"]}
    print("证据来源：", path)
    print("模式：Fake模型；图、工具与存储真实执行。")
    print("\nG03 已记录的计算消息流：")
    for index, message in enumerate(cases["G03"]["evidence"]["messages"], start=1):
        print(f"{index}. {message['type']}: {message['content']}")
        if message["tool_calls"]:
            print("   请求：", message["tool_calls"])
        if message["tool_call_id"]:
            print("   对应调用：", message["tool_call_id"])
    note = cases["G06"]["evidence"]["approval_state"]
    print("\nG06 已记录的审批：", note["status"], "批准：", note["approval"])
    print("会话：", note["pending_note"]["draft"]["session_id"])
    print("G10 重试后复用已有文件：", cases["G10"]["evidence"]["reused"])
    print("以上来自保留证据；测试临时目录已清理，不表示对应临时路径仍有文件。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day35独立讲解练习")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--answers", action="store_true", help="显示参考答案")
    mode.add_argument("--practice", action="store_true", help="逐题输入自己的讲解并保存")
    args = parser.parse_args()
    print("Day35 第6步：独立讲解与查漏补缺")
    show_evidence()
    responses = []
    for case_id, question, answer in QUESTIONS:
        print(f"\n{case_id} {question}")
        if args.answers:
            print("参考：", answer)
        elif args.practice:
            try:
                response = input("你的讲解（可留空，或输入exit结束）：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if response.lower() == "exit":
                break
            responses.append({"question_id": case_id, "question": question, "response": response, "status": "awaiting_review" if response else "not_answered"})
    if args.practice:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        output = DAY35_DIR / "reports/practice" / f"{run_id}.json"
        save_plan(output, {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "awaiting_human_review", "expected_questions": len(QUESTIONS),
            "responses": responses,
        })
        print("\n你的讲解已保存：", output)
        print("未自动评分；运行成功不表示已掌握。")
    elif not args.answers:
        print("\n先口头讲解，再用--answers对照；也可用--practice保存回答。")


if __name__ == "__main__":
    main()
