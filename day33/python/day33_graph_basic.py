"""Day33 第2步：LangGraph节点与条件路由骨架。

python day33/python/day33_graph_basic.py
python day33/python/day33_graph_basic.py --demo

四个分支目前返回占位说明，用于学习图结构，尚未接入真实业务能力。
依赖同目录的day33_project_contract.py，运行时无需手动配置导入路径。
"""

from __future__ import annotations

import argparse
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from day33_project_contract import Evidence, Route, UserRequest, initial_state


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


def choose_next_node(state: GraphState) -> str:
    """条件边读取router已经写入的结果，不再重复判断用户意图。"""
    route = state["route"]
    if route is None:
        raise ValueError("router尚未设置路线。")
    return route.value


def response_update(answer: str) -> dict:
    """节点返回增量更新；add_messages合并这一条新助手消息。"""
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


def chat_node(state: GraphState) -> dict:
    print("[chat] 进入聊天分支")
    return response_update("已进入聊天节点。后续接入模型与会话记忆。")


def calculator_node(state: GraphState) -> dict:
    print("[calculator] 进入计算分支")
    return response_update("已进入计算节点。下一步接入add、multiply与工具循环。")


def knowledge_node(state: GraphState) -> dict:
    print("[knowledge] 进入资料分支")
    return response_update("已进入知识检索节点。后续接入课程资料与来源引用。")


def note_node(state: GraphState) -> dict:
    print("[save_note] 进入笔记分支，当前不写文件")
    return response_update("已进入保存笔记节点。审批功能将在Day34实现，当前没有保存。")


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("router", router_node)
    builder.add_node("chat", chat_node)
    builder.add_node("calculator", calculator_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("note", note_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        choose_next_node,
        {
            "chat": "chat",
            "calculate": "calculator",
            "knowledge": "knowledge",
            "save_note": "note",
        },
    )
    for node_name in ("chat", "calculator", "knowledge", "note"):
        builder.add_edge(node_name, END)
    return builder.compile()


def run_once(graph, question: str) -> dict:
    request = UserRequest(session_id="xiaolin-capstone-001", question=question)
    print("\n问题：", request.question)
    result = graph.invoke(initial_state(request))
    print("最终路线：", result["route"].value)
    print("助手：", result["answer"])
    print("消息数量：", len(result["messages"]))
    for index, message in enumerate(result["messages"], start=1):
        print(f"  {index}. {type(message).__name__}: {message.content}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Day33 LangGraph路由骨架")
    parser.add_argument("--demo", action="store_true", help="运行四条路线示例")
    args = parser.parse_args()
    graph = build_graph()
    print("Day33 第2步：LangGraph节点与条件路由")
    try:
        if args.demo:
            examples = (
                ("你好，我叫小林。", Route.CHAT),
                ("请计算(20+4)*3。", Route.CALCULATE),
                ("根据课程资料解释RAG。", Route.KNOWLEDGE),
                ("把RAG的定义保存为学习笔记。", Route.SAVE_NOTE),
            )
            for question, expected_route in examples:
                result = run_once(graph, question)
                if result["route"] != expected_route or len(result["messages"]) != 2:
                    raise AssertionError("示例路线或消息合并结果不符合预期。")
            print("\n四条示例路线及消息合并：全部通过。")
        else:
            run_once(graph, input("请输入问题："))
    except ValidationError as error:
        print("输入无效：", error)
        raise SystemExit(2) from error
    print("\n数据流：UserRequest → 初始状态 → START → router → 条件边")
    print("→ 对应业务节点 → add_messages合并回复 → END → 最终状态")
    print("当前每次调用独立，尚未持久化历史；占位回复不表示业务已完成。")


if __name__ == "__main__":
    main()
