"""Day33 第3步：计算工具与LangGraph工具循环。

python day33/python/day33_tool_agent.py
python day33/python/day33_tool_agent.py --demo

正常模式调用.env配置的DeepSeek；demo只演示固定算式(20+4)*3，使用Fake模型。
复用第1步输入约定和第2步非计算节点，本文件完整定义本步的图与计算能力。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import ValidationError

from day33_project_contract import UserRequest, initial_state
from day33_graph_basic import (
    GraphState, chat_node, choose_next_node, knowledge_node, note_node, router_node,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def finite_result(value: float) -> float:
    """拒绝溢出或非有限数，避免把inf/nan当成有效计算结果。"""
    if not math.isfinite(value):
        raise ValueError("计算结果必须是有限数值。")
    return value


@tool
def add(a: float, b: float) -> float:
    """计算两个有限数值的和。"""
    if not all(math.isfinite(value) for value in (a, b)):
        raise ValueError("输入必须是有限数值。")
    result = finite_result(a + b)
    print(f"[执行 add] {a} + {b} = {result}")
    return result


@tool
def multiply(a: float, b: float) -> float:
    """计算两个有限数值的乘积。"""
    if not all(math.isfinite(value) for value in (a, b)):
        raise ValueError("输入必须是有限数值。")
    result = finite_result(a * b)
    print(f"[执行 multiply] {a} × {b} = {result}")
    return result


TOOLS = [add, multiply]


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


def create_calculator_model(demo: bool):
    if demo:
        return DemoCalculator()
    load_dotenv(PROJECT_ROOT / ".env")
    names = ("DEEPSEEK_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL")
    if any(not os.getenv(name, "").strip() for name in names):
        raise ValueError("请在项目.env中配置DEEPSEEK_MODEL、DEEPSEEK_API_KEY和DEEPSEEK_BASE_URL。")
    llm = ChatOpenAI(
        model=os.environ["DEEPSEEK_MODEL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        temperature=0,
        timeout=30,
        max_retries=1,
    )
    return llm.bind_tools(TOOLS)


def build_graph(calculator_model):
    """通过参数注入模型，真实与Fake运行同一张图。"""
    def calculator_node(state: GraphState) -> dict:
        response = calculator_model.invoke([
            SystemMessage(content=(
                "你是简洁的计算助手。加法和乘法必须调用工具，"
                "根据实际工具结果回答，不得虚构工具执行。"
                "遇到不支持的运算请说明能力限制。"
            )),
            *state["messages"],
        ])
        if response.tool_calls:
            print("[calculator] 请求工具：", response.tool_calls)
            return {"messages": [response]}
        print("[calculator] 生成最终回答")
        return {"messages": [response], "answer": response.content}

    builder = StateGraph(GraphState)
    builder.add_node("router", router_node)
    builder.add_node("chat", chat_node)
    builder.add_node("calculator", calculator_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("note", note_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", choose_next_node, {
        "chat": "chat", "calculate": "calculator",
        "knowledge": "knowledge", "save_note": "note",
    })
    # tools_condition查看最后AIMessage是否带tool_calls。
    builder.add_conditional_edges(
        "calculator", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "calculator")
    for name in ("chat", "knowledge", "note"):
        builder.add_edge(name, END)
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Day33计算工具Agent")
    parser.add_argument("--demo", action="store_true", help="离线固定样例，无模型API请求")
    args = parser.parse_args()
    print("Day33 第3步：计算工具与工具循环")
    print("模式：", "Fake模型 + 真实Python工具" if args.demo else "DeepSeek + 真实Python工具")
    question = "请计算(20+4)*3。" if args.demo else input("请输入问题（计算问题请写“计算”）：")
    try:
        request = UserRequest(session_id="xiaolin-capstone-001", question=question)
        graph = build_graph(create_calculator_model(args.demo))
        result = graph.invoke(initial_state(request), config={"recursion_limit": 20})
    except (ValidationError, ValueError) as error:
        print("输入或配置无效：", error)
        raise SystemExit(2) from error

    print("\n最终回答：", result["answer"])
    print("\n完整消息流：")
    for index, message in enumerate(result["messages"], start=1):
        print(f"{index}. {type(message).__name__}: {message.content!r}")
        if isinstance(message, AIMessage) and message.tool_calls:
            print("   tool_calls：", message.tool_calls)
        if isinstance(message, ToolMessage):
            print("   tool_call_id：", message.tool_call_id)
    if args.demo:
        tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if len(tool_results) != 2 or float(tool_results[-1].content) != 72:
            raise AssertionError("离线样例工具执行不符合预期。")
        print("\n离线工具循环验证通过：24 → 72。")
    print("\n数据流：router → calculator → tools → calculator → … → END")
    print("本步未接入持久化；其他业务路线继续使用第2步的占位节点。")


if __name__ == "__main__":
    main()
