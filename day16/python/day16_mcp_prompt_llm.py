"""Day16 第七步：把 MCP Prompt 转换为 LangChain 消息并调用大模型。

运行方式：
    python day16/python/day16_mcp_prompt_llm.py

数据流：
    用户参数
    → MCP Prompt Server
    → GetPromptResult.messages
    → LangChain BaseMessage 列表
    → ChatOpenAI / DeepSeek
    → 模型回答
"""

from __future__ import annotations

import asyncio
import os

import mcp_types as types
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver import Message, UserMessage


load_dotenv()


def create_llm() -> ChatOpenAI:
    """根据项目根目录的 .env 创建大模型客户端。"""

    model = os.getenv("DEEPSEEK_MODEL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    missing_names = [
        name
        for name, value in {
            "DEEPSEEK_MODEL": model,
            "DEEPSEEK_API_KEY": api_key,
            "DEEPSEEK_BASE_URL": base_url,
        }.items()
        if not value
    ]
    if missing_names:
        raise RuntimeError(
            ".env 缺少配置：" + "、".join(missing_names)
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


mcp_server = MCPServer(
    name="day16-learning-prompts",
    title="Day16 编程学习 Prompt 服务",
    description="为编程学习助手提供统一的教学提示词。",
    version="1.0.0",
)


@mcp_server.prompt(
    name="teach-programming-concept",
    title="讲解编程概念",
    description="根据主题和学习者水平生成教学消息。",
)
async def teach_programming_concept(
    topic: str,
    level: str = "初学者",
) -> list[Message]:
    """生成一条结构清晰的编程教学请求。"""

    return [
        UserMessage(
            f"请面向{level}讲解“{topic}”。\n\n"
            "回答要求：\n"
            "1. 开头用一句话给出定义；\n"
            "2. 解释它解决了什么问题；\n"
            "3. 给出一个可以直接运行的最小 Python 示例；\n"
            "4. 说明一个初学者常见错误；\n"
            "5. 最后给出一道练习题，不要直接给答案。\n"
            "请使用简洁的中文。"
        )
    ]


def mcp_messages_to_langchain(
    prompt_messages: list[types.PromptMessage],
) -> list[BaseMessage]:
    """把 MCP PromptMessage 列表转换成 LangChain 消息列表。"""

    langchain_messages: list[BaseMessage] = [
        SystemMessage(
            content="你是一名准确、耐心、循序渐进的编程学习助手。"
        )
    ]

    for message in prompt_messages:
        content = message.content
        if not isinstance(content, types.TextContent):
            raise TypeError(
                "当前练习只支持文本 Prompt，"
                f"收到的内容类型是 {type(content).__name__}。"
            )

        if message.role == "user":
            langchain_messages.append(HumanMessage(content=content.text))
        elif message.role == "assistant":
            langchain_messages.append(AIMessage(content=content.text))
        else:
            raise ValueError(f"不支持的 MCP 消息角色：{message.role}")

    return langchain_messages


def show_messages(title: str, messages: list[BaseMessage]) -> None:
    """显示即将发送给模型的 LangChain 消息。"""

    print("\n" + "=" * 70)
    print(title)

    for index, message in enumerate(messages, start=1):
        print(f"\n第 {index} 条：{type(message).__name__}")
        print(message.content)


async def main() -> None:
    """获取 MCP Prompt，转换消息，然后异步调用大模型。"""

    topic = input("请输入想学习的编程概念：").strip()
    if not topic:
        print("主题不能为空。")
        return

    level = input("请输入你的学习水平（直接回车使用“初学者”）：").strip()
    if not level:
        level = "初学者"

    llm = create_llm()

    client_info = types.Implementation(
        name="day16-prompt-llm-client",
        version="1.0.0",
    )

    print("\n正在从 MCP Server 获取 Prompt……")

    async with Client(
        mcp_server,
        client_info=client_info,
    ) as client:
        prompt_result = await client.get_prompt(
            "teach-programming-concept",
            arguments={
                "topic": topic,
                "level": level,
            },
        )

    print("MCP 返回类型：", type(prompt_result))
    print("MCP 消息数量：", len(prompt_result.messages))

    langchain_messages = mcp_messages_to_langchain(
        prompt_result.messages
    )
    show_messages("转换后的 LangChain 消息", langchain_messages)

    print("\n" + "=" * 70)
    print("正在调用大模型……")
    response = await llm.ainvoke(langchain_messages)

    print("\n模型回答：")
    print(response.content)

    print("\n" + "=" * 70)
    print("最终数据流：")
    print("topic、level")
    print("→ Client.get_prompt()")
    print("→ MCP PromptMessage")
    print("→ HumanMessage / AIMessage")
    print("→ llm.ainvoke(messages)")
    print("→ AIMessage")


if __name__ == "__main__":
    asyncio.run(main())
