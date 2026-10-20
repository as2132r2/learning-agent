"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

import re
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from .contracts import UserRequest
from .demo_tools import DemoCalculator


class PersistentState(TypedDict):
    """证据保存为普通dict，避免checkpoint依赖自定义类的反序列化。"""
    session_id: str
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    route: str | None
    evidence: list[dict[str, str]]
    answer: str
    pending_note: str | None
    approval: bool | None


def current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """从最近HumanMessage起取消息，供固定计算样例使用。"""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


class DemoChatModel:
    """确定性读历史，验证存储接线；不代表真实自然语言理解能力。"""
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        humans = [message.content for message in messages if isinstance(message, HumanMessage)]
        question = str(humans[-1])
        if "叫什么" in question:
            names = [re.search(r"我叫([\u4e00-\u9fff]{1,8})[。！!，,\s]", str(text)) for text in humans]
            names = [match[1] for match in names if match]
            return AIMessage(content=f"你叫{names[-1]}。" if names else "当前会话还不知道你的名字。")
        if "刚才" in question and "结果" in question:
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            return AIMessage(content=f"刚才的工具结果是{tools[-1].content}。" if tools else "当前会话没有工具结果。")
        return AIMessage(content="已收到，内容将保存到当前会话。（离线演示）")


class TurnScopedDemoCalculator:
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        # 前几轮的工具结果不能污染DemoCalculator的固定步骤判断。
        return DemoCalculator().invoke(current_turn(messages))


def config_for(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 20}


def run_turn(graph, session_id: str, question: str) -> dict:
    request = UserRequest(session_id=session_id, question=question)
    # 只传当前用户消息，checkpointer加载旧状态，add_messages自动合并。
    update = {
        "session_id": request.session_id, "question": request.question,
        "messages": [HumanMessage(content=request.question)],
        "route": None, "evidence": [], "answer": "",
        "pending_note": None, "approval": None,
    }
    result = graph.invoke(update, config=config_for(request.session_id))
    print(f"\n[{request.session_id}] 你：{request.question}")
    print("助手：", result["answer"])
    print("路线：", result["route"], "累计消息数：", len(result["messages"]))
    for number, item in enumerate(result["evidence"], start=1):
        print(f"[资料{number}] {item['source']} | {item['chunk_id']}")
    return result

