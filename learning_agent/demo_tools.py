"""Extracted and adapted from the programming-learning capstone."""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage


class DemoCalculator:
    """按固定样例生成工具请求；真实结果仍来自ToolNode执行。"""

    def invoke(self, messages: list) -> AIMessage:
        results = [message for message in messages if isinstance(message, ToolMessage)]
        if not results:
            return AIMessage(content="", tool_calls=[{
                "name": "add", "args": {"a": 20, "b": 4},
                "id": "demo-add-001", "type": "tool_call",
            }])
        if len(results) == 1:
            intermediate = float(results[-1].content)
            return AIMessage(content="", tool_calls=[{
                "name": "multiply", "args": {"a": intermediate, "b": 3},
                "id": "demo-multiply-001", "type": "tool_call",
            }])
        value = float(results[-1].content)
        return AIMessage(content=f"(20+4)×3 = {value:g}（离线固定样例）")

