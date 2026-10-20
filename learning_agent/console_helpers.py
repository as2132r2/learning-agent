"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from .contracts import UserRequest
from .memory import config_for


def validated_session_id(value: str) -> str:
    """复用第1步输入约束，不使用会话编号拼接数据库或文件路径。"""
    return UserRequest(session_id=value, question="会话检查").session_id


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

