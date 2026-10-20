"""Day16 第十一步：一个 LangChain Agent 同时连接两个 MCP Server。

终端一：
    python day16/python/day16_mcp_http_server.py

终端二：
    python day16/python/day16_mcp_knowledge_server.py

终端三：
    python day16/python/day16_mcp_multi_server_agent.py

可测试：
    请计算 (20+4)*3。
    请解释什么是 RAG。
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI


load_dotenv()


# 标准的多 MCP Server 配置。每个名字会成为工具名称的前缀。
MCP_CONFIG = {
    "mcpServers": {
        "calculator": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "http",
        },
        "knowledge": {
            "url": "http://127.0.0.1:8001/mcp",
            "transport": "http",
        },
    }
}


def create_llm() -> ChatOpenAI:
    """从 .env 创建 DeepSeek 模型。"""

    model = os.getenv("DEEPSEEK_MODEL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    missing = [
        name
        for name, value in {
            "DEEPSEEK_MODEL": model,
            "DEEPSEEK_API_KEY": api_key,
            "DEEPSEEK_BASE_URL": base_url,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(".env 缺少配置：" + "、".join(missing))

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def show_tools(tools: list) -> None:
    """显示两个服务汇总后的 LangChain 工具。"""

    print("\n" + "=" * 70)
    print("从多个 MCP Server 汇总的工具：")

    for index, tool in enumerate(tools, start=1):
        print(f"\n工具 {index}")
        print("名称：", tool.name)
        print("说明：", tool.description)
        print("参数：", tool.args)


def show_messages(messages: list) -> None:
    """显示 Agent 完整消息流。"""

    print("\n" + "=" * 70)
    print("完整消息流：")

    for index, message in enumerate(messages, start=1):
        print(f"\n第 {index} 条：{type(message).__name__}")
        print("content：", repr(message.content))

        if isinstance(message, AIMessage) and message.tool_calls:
            print("tool_calls：", message.tool_calls)

        if isinstance(message, ToolMessage):
            print("工具名称：", message.name)
            print("tool_call_id：", message.tool_call_id)


async def main() -> None:
    """同时连接计算服务和知识服务，然后运行多服务 Agent。"""

    question = input("请输入问题：").strip()
    if not question:
        print("问题不能为空。")
        return

    llm = create_llm()

    print("\n准备连接两个 MCP Server：")
    for name, config in MCP_CONFIG["mcpServers"].items():
        print(f"- {name}: {config['url']}")

    try:
        async with MCPAdapter(MCP_CONFIG) as adapter:
            tools = await adapter.list_tools(cache_mode="refresh")
            show_tools(tools)

            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=(
                    "你是一名简洁、可靠的学习助手。"
                    "数学运算必须使用 calculator 服务的工具；"
                    "LangChain、LangGraph、RAG、MCP 课程问题必须使用 "
                    "knowledge 服务的工具。"
                    "根据用户问题选择正确的工具，可以连续调用多个工具。"
                ),
            )

            print("\n" + "=" * 70)
            print("Agent 开始处理：", question)
            result = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                }
            )
    except Exception as error:
        print("\n连接或执行失败。")
        print("错误类型：", type(error).__name__)
        print("错误信息：", error)
        print("请确认 8000 和 8001 两个 MCP Server 都已启动。")
        return

    print("\n" + "=" * 70)
    print("最终回答：")
    print(result["messages"][-1].content)

    show_messages(result["messages"])

    print("\n" + "=" * 70)
    print("本步数据流：")
    print("MCP_CONFIG")
    print("├── calculator → 127.0.0.1:8000/mcp")
    print("└── knowledge  → 127.0.0.1:8001/mcp")
    print("          ↓")
    print("MCPAdapter 汇总并转换工具")
    print("          ↓")
    print("带服务名前缀的 LangChain Tools")
    print("          ↓")
    print("DeepSeek 根据问题选择正确服务")


if __name__ == "__main__":
    asyncio.run(main())
