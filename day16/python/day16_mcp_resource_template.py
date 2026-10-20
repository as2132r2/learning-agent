"""Day16 第五步：使用 MCP Resource Template 动态读取资源。

运行方式：
    python day16/python/day16_mcp_resource_template.py

固定 Resource：
    course://notes/rag

Resource Template：
    course://notes/{topic}

其中 {topic} 会从实际 URI 中提取，并传给 Python 资源函数。
"""

from __future__ import annotations

import asyncio

import mcp_types as types
from mcp import Client
from mcp.server import MCPServer


mcp_server = MCPServer(
    name="day16-dynamic-resources",
    title="Day16 动态课程资料服务",
    description="使用一个 URI 模板读取不同主题的学习资料。",
    version="1.0.0",
)


# 模拟服务端数据库。实际项目中也可以从文件、SQLite 或远程 API 读取。
COURSE_NOTES: dict[str, str] = {
    "langchain": (
        "LangChain 用于开发大语言模型应用，"
        "可以连接模型、Prompt、Tool、Memory 和 Retriever。"
    ),
    "langgraph": (
        "LangGraph 使用图结构组织有状态工作流，"
        "节点处理数据，边决定流程走向。"
    ),
    "rag": (
        "RAG 是检索增强生成：先检索外部资料，"
        "再把资料和问题一起交给大语言模型。"
    ),
    "mcp": (
        "MCP 是模型上下文协议，用统一方式向大模型应用提供"
        "工具、资源和提示词。"
    ),
}


@mcp_server.resource(
    "course://notes/{topic}",
    name="course-note-by-topic",
    title="按主题读取课程笔记",
    description="把 topic 替换成 langchain、langgraph、rag 或 mcp。",
    mime_type="text/plain",
)
async def read_course_note(topic: str) -> str:
    """根据 URI 中的 topic 参数返回课程笔记。"""

    print(f"[Resource Provider] 收到 topic={topic!r}")

    note = COURSE_NOTES.get(topic.lower())
    if note is None:
        available = "、".join(COURSE_NOTES)
        raise ValueError(f"未知主题 {topic!r}；可用主题：{available}")

    return note


def show_templates(result: types.ListResourceTemplatesResult) -> None:
    """显示服务端公开的 URI 模板。"""

    print("\n" + "=" * 70)
    print("Resource Template 数量：", len(result.resource_templates))

    for index, template in enumerate(result.resource_templates, start=1):
        print(f"\n模板 {index}")
        print("Python 类型：", type(template))
        print("URI 模板：", template.uri_template)
        print("名称：", template.name)
        print("标题：", template.title)
        print("说明：", template.description)
        print("MIME 类型：", template.mime_type)


def show_resource(uri: str, result: types.ReadResourceResult) -> None:
    """显示一次动态资源读取结果。"""

    print("\n" + "-" * 70)
    print("请求 URI：", uri)

    for content in result.contents:
        print("返回 URI：", content.uri)
        print("内容对象类型：", type(content))
        if isinstance(content, types.TextResourceContents):
            print("正文：", content.text)


async def main() -> None:
    """列出模板，并用不同 URI 调用同一个资源函数。"""

    print("Day16 MCP Resource Template 客户端已启动。")

    client_info = types.Implementation(
        name="day16-resource-template-client",
        version="1.0.0",
    )

    async with Client(
        mcp_server,
        client_info=client_info,
    ) as client:
        print("协议版本：", client.protocol_version)

        # 动态资源出现在模板列表，而不是固定资源列表。
        resources = await client.list_resources()
        templates = await client.list_resource_templates()

        print("固定 Resource 数量：", len(resources.resources))
        show_templates(templates)

        for topic in ("rag", "langgraph", "mcp"):
            uri = f"course://notes/{topic}"
            result = await client.read_resource(uri)
            show_resource(uri, result)

    print("\n" + "=" * 70)
    print("三次读取使用了同一个 Python 函数：read_course_note(topic)")
    print("\n本步数据流：")
    print("course://notes/rag")
    print("→ 匹配 course://notes/{topic}")
    print("→ 提取 topic='rag'")
    print("→ 调用 read_course_note(topic='rag')")
    print("→ 从 COURSE_NOTES 查询正文")
    print("→ 返回 ReadResourceResult")


if __name__ == "__main__":
    asyncio.run(main())
