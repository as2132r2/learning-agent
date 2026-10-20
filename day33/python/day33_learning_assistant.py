"""Day33 第6步：连续交互式毕业项目助手。

python day33/python/day33_learning_assistant.py xiaolin-capstone-001
python day33/python/day33_learning_assistant.py --demo
python day33/python/day33_learning_assistant.py --check

正常模式复用第5步业务数据库；demo使用独立演示数据库；check使用临时数据库。
本文件是完整交互入口，业务图复用同目录day33_persistent_agent.py。
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from day33_project_contract import UserRequest
from day33_persistent_agent import DATABASE_PATH, build_graph, config_for, run_turn


DEMO_DATABASE_PATH = DATABASE_PATH.with_name("day33_agent_demo.db")


def validated_session_id(value: str) -> str:
    """复用第1步输入约束，不使用会话编号拼接数据库或文件路径。"""
    return UserRequest(session_id=value, question="会话检查").session_id


def show_help() -> None:
    print("\n命令：")
    print("  /help             查看帮助")
    print("  /session 会话编号  切换到指定会话，不删除原会话")
    print("  /history          查看当前会话最近12条消息")
    print("  /status           查看会话状态")
    print("  exit              结束并关闭数据库连接")
    print("计算示例：请计算(20+4)*3。")
    print("检索示例：根据课程资料解释RAG。")
    print("保存笔记入口将在Day34接入人工审批，目前不会写笔记。")


def show_status(graph, session_id: str) -> None:
    snapshot = graph.get_state(config_for(session_id))
    values = snapshot.values
    print("当前会话：", session_id)
    print("累计消息数量：", len(values.get("messages", [])))
    print("最后路线：", values.get("route"))
    print("待执行节点：", snapshot.next or "无")


def show_history(graph, session_id: str) -> None:
    messages = graph.get_state(config_for(session_id)).values.get("messages", [])
    if not messages:
        print("当前会话还没有历史消息。")
        return
    start = max(0, len(messages) - 12)
    for index, message in enumerate(messages[start:], start=start + 1):
        print(f"{index}. {type(message).__name__}: {message.content}")
        if isinstance(message, AIMessage) and message.tool_calls:
            print("   tool_calls：", message.tool_calls)
        if isinstance(message, ToolMessage):
            print("   tool_call_id：", message.tool_call_id)


def chat_loop(
    graph,
    session_id: str,
    read_input: Callable[[str], str] = input,
) -> tuple[str, bool]:
    """返回最后会话编号和是否正常退出；允许注入脚本输入验证同一循环。"""
    session_id = validated_session_id(session_id)
    show_help()
    show_status(graph, session_id)
    while True:
        try:
            text = read_input(f"\n[{session_id}] 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n结束聊天。")
            return session_id, True
        if not text:
            print("请输入内容。")
            continue
        if text.lower() == "exit":
            print("聊天结束。")
            return session_id, True
        if text == "/help":
            show_help()
            continue
        if text == "/status":
            show_status(graph, session_id)
            continue
        if text == "/history":
            show_history(graph, session_id)
            continue
        if text.startswith("/session "):
            try:
                new_session = validated_session_id(text.split(maxsplit=1)[1])
            except ValidationError as error:
                print("会话编号无效：", error)
            else:
                session_id = new_session
                show_status(graph, session_id)
            continue
        if text.startswith("/"):
            print("未知命令，请输入/help。")
            continue
        try:
            # 先验证输入，避免无效问题进入checkpoint。
            request = UserRequest(session_id=session_id, question=text)
        except ValidationError as error:
            print("输入无效：", error)
            continue
        snapshot = graph.get_state(config_for(session_id))
        if snapshot.next:
            print("该会话存在未完成步骤：", snapshot.next)
            print("本步尚未实现异常任务恢复，请切换新会话；原状态保留。")
            continue
        try:
            run_turn(graph, request.session_id, request.question)
        except KeyboardInterrupt:
            print("\n本轮被中断，检查点保留；结束程序。")
            return session_id, False
        except Exception as error:
            print(f"本轮失败：{type(error).__name__}: {error}")
            print("程序停止，本轮可能未完成；修复原因后用/status检查或使用新会话。")
            return session_id, False


def check_interaction() -> None:
    """脚本化输入验证真实交互循环，不发起模型API请求。"""
    inputs = iter([
        "", "/help", "我叫小林。", "请计算(20+4)*3。",
        "根据课程资料解释RAG。", "/history", "/status",
        "/session other-user", "我叫什么名字？",
        "/session xiaolin-demo", "我叫什么名字？",
        "把RAG的定义保存为学习笔记。", "exit",
    ])

    def scripted_input(prompt: str) -> str:
        value = next(inputs)
        print(prompt + value)
        return value

    with tempfile.TemporaryDirectory(prefix="day33-interaction-") as directory:
        database = str(Path(directory) / "check.db")
        with SqliteSaver.from_conn_string(database) as saver:
            graph = build_graph(saver, demo=True)
            session, success = chat_loop(graph, "xiaolin-demo", scripted_input)
            assert success and session == "xiaolin-demo"
            other = graph.get_state(config_for("other-user")).values
            assert "不知道" in other["answer"]
            known = graph.get_state(config_for("xiaolin-demo")).values
            assert any("你叫小林" in str(m.content) for m in known["messages"])
            assert known["route"] == "save_note" and known["approval"] is None
        # 重建连接后，通过同一交互循环继续问名字。
        resumed_inputs = iter(["我叫什么名字？", "exit"])
        with SqliteSaver.from_conn_string(database) as saver:
            graph = build_graph(saver, demo=True)
            _, success = chat_loop(graph, "xiaolin-demo", lambda _: next(resumed_inputs))
            assert success
            assert "小林" in graph.get_state(config_for("xiaolin-demo")).values["answer"]
    print("\n交互脚本验证通过：计算、检索、记忆、切换会话、命令和重新连接恢复。")
    print("保存笔记目前只返回未实现说明，未建立审批或写入流程。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day33编程学习助手交互入口")
    parser.add_argument("session_id", nargs="?", default="xiaolin-capstone-001")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Fake模型交互，计算仅支持固定样例")
    mode.add_argument("--check", action="store_true", help="临时数据库脚本化验收")
    args = parser.parse_args()
    print("Day33 编程学习助手")
    if args.check:
        check_interaction()
        return
    session_id = validated_session_id(args.session_id)
    database = DEMO_DATABASE_PATH if args.demo else DATABASE_PATH
    database.parent.mkdir(parents=True, exist_ok=True)
    print("模式：", "离线固定样例，知识仅支持RAG演示" if args.demo else "DeepSeek与已有课程索引")
    print("数据库：", database)
    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_graph(saver, demo=args.demo)
        _, success = chat_loop(graph, session_id)
    print("数据库连接已关闭。")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
