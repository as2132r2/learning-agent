"""Day16 第十一步（知识服务端）：独立的课程知识 MCP Server。

运行方式：
    python day16/python/day16_mcp_knowledge_server.py

MCP 地址：
    http://127.0.0.1:8001/mcp
"""

from __future__ import annotations

import sys
from typing import Literal

from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse


mcp_server = MCPServer(
    name="day16-knowledge-server",
    title="Day16 课程知识服务",
    description="提供 LangChain、LangGraph、RAG 和 MCP 基础知识。",
    version="1.0.0",
)


KNOWLEDGE: dict[str, str] = {
    "langchain": (
        "LangChain 是开发大语言模型应用的框架，可以连接模型、"
        "提示词、工具、记忆和检索系统。"
    ),
    "langgraph": (
        "LangGraph 使用图结构组织有状态工作流。"
        "节点处理状态，边决定流程走向，并支持分支、循环和持久化。"
    ),
    "rag": (
        "RAG 是检索增强生成：先从外部知识库检索相关资料，"
        "再把资料和用户问题一起交给大模型生成回答。"
    ),
    "mcp": (
        "MCP 是模型上下文协议，用统一方式向大模型应用公开"
        "Tool、Resource 和 Prompt。"
    ),
}


@mcp_server.tool()
async def explain_topic(
    topic: Literal["langchain", "langgraph", "rag", "mcp"],
) -> str:
    """查询并解释指定的课程主题。"""

    print(
        f"[知识 MCP Tool 执行 explain_topic] topic={topic}",
        file=sys.stderr,
        flush=True,
    )
    return KNOWLEDGE[topic]


@mcp_server.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """返回知识服务健康状态。"""

    return JSONResponse(
        {
            "status": "ok",
            "service": "day16-knowledge-server",
            "mcp_endpoint": "/mcp",
        }
    )


def main() -> None:
    """在 8001 端口启动知识 MCP Server。"""

    print("Day16 知识 MCP Server 正在启动……")
    print("MCP 地址：http://127.0.0.1:8001/mcp")
    print("按 Ctrl+C 停止服务。")

    try:
        mcp_server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=8001,
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
        )
    except KeyboardInterrupt:
        print("\n收到停止信号，知识 MCP Server 已关闭。")


if __name__ == "__main__":
    main()
