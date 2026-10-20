"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from .contracts import Route, Evidence


class GraphState(TypedDict):
    """字段承接第1步；messages新增LangGraph消息合并规则。"""

    session_id: str
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    route: Route | None
    evidence: list[Evidence]
    answer: str
    pending_note: str | None
    approval: bool | None


def router_node(state: GraphState) -> dict:
    """只负责写入路由。本步采用可观察的规则，优先处理保存笔记。"""
    question = state["question"].lower()
    if any(term in question for term in ("保存", "记录")) and "笔记" in question:
        route = Route.SAVE_NOTE
    elif "计算" in question:
        route = Route.CALCULATE
    elif any(term in question for term in ("资料", "rag", "langchain", "langgraph")):
        route = Route.KNOWLEDGE
    else:
        route = Route.CHAT
    print(f"[router] route={route.value}")
    return {"route": route}


def response_update(answer: str) -> dict:
    """节点返回增量更新；add_messages合并这一条新助手消息。"""
    return {"answer": answer, "messages": [AIMessage(content=answer)]}

