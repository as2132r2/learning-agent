"""Day16 最终整合（服务端）：一套完整的受保护 MCP 服务。

运行方式：
    python day16/python/day16_final_mcp_server.py

MCP 地址：
    http://127.0.0.1:8004/mcp

教学令牌：
    day16-final-reader-token
        → course:read、account:read

    day16-final-operator-token
        → course:read、account:read、account:write

本文件整合：
    1. Streamable HTTP 传输；
    2. Bearer Token 身份认证；
    3. scope 工具级授权；
    4. Tool 与安全注解；
    5. Resource；
    6. Prompt；
    7. SQLite 审计日志。

注意：固定令牌和内存账户仅用于教学，不能直接用于真实金融系统。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import mcp_types as types
from pydantic import AnyHttpUrl

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import UserMessage
from mcp.server.mcpserver.exceptions import ToolError


# ---------------------------------------------------------------------------
# 1. 服务地址、数据库地址和教学令牌
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 8004
MCP_PATH = "/mcp"
RESOURCE_URL = f"http://{HOST}:{PORT}{MCP_PATH}"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "day16" / "database" / "day16_final_mcp.db"

READER_TOKEN = "day16-final-reader-token"
OPERATOR_TOKEN = "day16-final-operator-token"


# AccessToken 是服务端验证令牌后得到的标准身份对象。
KNOWN_TOKENS: dict[str, AccessToken] = {
    READER_TOKEN: AccessToken(
        token=READER_TOKEN,
        client_id="day16-reader-client",
        subject="user-xiaolin",
        scopes=["course:read", "account:read"],
        resource=RESOURCE_URL,
    ),
    OPERATOR_TOKEN: AccessToken(
        token=OPERATOR_TOKEN,
        client_id="day16-operator-client",
        subject="operator-leo",
        scopes=["course:read", "account:read", "account:write"],
        resource=RESOURCE_URL,
    ),
}


class FinalTokenVerifier(TokenVerifier):
    """验证 Bearer Token，并返回与它对应的身份和权限。"""

    async def verify_token(self, token: str) -> AccessToken | None:
        return KNOWN_TOKENS.get(token)


# ---------------------------------------------------------------------------
# 2. 创建 MCP Server
# ---------------------------------------------------------------------------

mcp_server = MCPServer(
    name="day16-final-server",
    title="Day16 最终整合 MCP 服务",
    description="提供课程资料、账户查询、模拟转账和教学 Prompt。",
    version="1.0.0",
    token_verifier=FinalTokenVerifier(),
    auth=AuthSettings(
        # 教学示例不访问此 URL，它表示真实系统中的授权服务器。
        issuer_url=AnyHttpUrl("https://auth.example.com"),
        resource_server_url=AnyHttpUrl(RESOURCE_URL),
        # HTTP 层只要求令牌有效，具体权限由每项能力分别检查。
        required_scopes=[],
        validate_token_resource=True,
    ),
)


# ---------------------------------------------------------------------------
# 3. 数据库与审计函数
# ---------------------------------------------------------------------------


def initialize_database() -> None:
    """创建审计表；表已经存在时不会重复创建。"""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                subject TEXT NOT NULL,
                action TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                result TEXT NOT NULL
            )
            """
        )


def write_audit_log(
    *,
    subject: str,
    action: str,
    arguments: dict[str, Any],
    result: str,
) -> None:
    """把一次重要操作写入 SQLite，便于事后追踪。"""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO audit_logs (
                created_at, subject, action, arguments_json, result
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                subject,
                action,
                json.dumps(arguments, ensure_ascii=False),
                result,
            ),
        )


# ---------------------------------------------------------------------------
# 4. 通用认证、授权函数
# ---------------------------------------------------------------------------


def require_scope(required_scope: str) -> AccessToken:
    """读取当前请求身份，并强制检查指定 scope。"""

    access_token = get_access_token()

    if access_token is None:
        raise ToolError("当前请求没有通过身份认证。")

    if required_scope not in access_token.scopes:
        raise ToolError(
            f"用户 {access_token.subject!r} 缺少权限 {required_scope!r}。"
        )

    return access_token


# ---------------------------------------------------------------------------
# 5. 模拟业务数据
# ---------------------------------------------------------------------------

ACCOUNTS: dict[str, float] = {
    "account-001": 1000.0,
    "account-002": 200.0,
}

COURSE_NOTES: dict[str, str] = {
    "mcp": (
        "MCP 是连接大模型应用与外部能力的开放协议。"
        "服务端可以公开 Tool、Resource 和 Prompt。"
    ),
    "rag": (
        "RAG 是检索增强生成。它先检索相关资料，再把资料和问题一起交给模型。"
    ),
}


# ---------------------------------------------------------------------------
# 6. Tools：可由客户端或模型请求执行的动作
# ---------------------------------------------------------------------------


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        title="查看当前身份",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def whoami() -> dict[str, Any]:
    """返回当前认证用户的身份和 scopes。"""

    access_token = get_access_token()
    if access_token is None:
        raise ToolError("当前请求没有通过身份认证。")

    return {
        "subject": access_token.subject,
        "client_id": access_token.client_id,
        "scopes": access_token.scopes,
    }


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        title="查询课程知识",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def search_course(
    topic: Literal["mcp", "rag"],
) -> dict[str, str]:
    """查询一条课程笔记；要求 course:read。"""

    access_token = require_scope("course:read")
    write_audit_log(
        subject=access_token.subject or "unknown",
        action="search_course",
        arguments={"topic": topic},
        result="success",
    )
    return {"topic": topic, "content": COURSE_NOTES[topic]}


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        title="查询账户余额",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def check_balance(
    account: Literal["account-001", "account-002"],
) -> dict[str, Any]:
    """查询模拟账户余额；要求 account:read。"""

    access_token = require_scope("account:read")
    write_audit_log(
        subject=access_token.subject or "unknown",
        action="check_balance",
        arguments={"account": account},
        result="success",
    )
    return {"account": account, "balance": ACCOUNTS[account]}


@mcp_server.tool(
    annotations=types.ToolAnnotations(
        title="执行模拟转账",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    )
)
async def transfer_money(
    from_account: Literal["account-001", "account-002"],
    to_account: Literal["account-001", "account-002"],
    amount: float,
) -> dict[str, Any]:
    """执行模拟转账；要求 account:write，客户端还应进行人工审批。"""

    access_token = require_scope("account:write")
    arguments = {
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
    }

    if from_account == to_account:
        write_audit_log(
            subject=access_token.subject or "unknown",
            action="transfer_money",
            arguments=arguments,
            result="rejected:same_account",
        )
        raise ToolError("付款账户和收款账户不能相同。")

    if amount <= 0:
        write_audit_log(
            subject=access_token.subject or "unknown",
            action="transfer_money",
            arguments=arguments,
            result="rejected:invalid_amount",
        )
        raise ToolError("转账金额必须大于 0。")

    if ACCOUNTS[from_account] < amount:
        write_audit_log(
            subject=access_token.subject or "unknown",
            action="transfer_money",
            arguments=arguments,
            result="rejected:insufficient_balance",
        )
        raise ToolError("付款账户余额不足。")

    ACCOUNTS[from_account] -= amount
    ACCOUNTS[to_account] += amount

    write_audit_log(
        subject=access_token.subject or "unknown",
        action="transfer_money",
        arguments=arguments,
        result="success",
    )

    return {
        "status": "success",
        **arguments,
        "balances": dict(ACCOUNTS),
    }


# ---------------------------------------------------------------------------
# 7. Resources：通过稳定 URI 提供只读资料
# ---------------------------------------------------------------------------


@mcp_server.resource(
    "course://notes/mcp",
    name="mcp-note",
    title="MCP 学习笔记",
    description="介绍 MCP 的核心能力。",
    mime_type="text/plain",
)
async def read_mcp_note() -> str:
    """读取 MCP 笔记；要求 course:read。"""

    require_scope("course:read")
    return COURSE_NOTES["mcp"]


@mcp_server.resource(
    "course://notes/rag",
    name="rag-note",
    title="RAG 学习笔记",
    description="介绍 RAG 的基本流程。",
    mime_type="text/plain",
)
async def read_rag_note() -> str:
    """读取 RAG 笔记；要求 course:read。"""

    require_scope("course:read")
    return COURSE_NOTES["rag"]


# ---------------------------------------------------------------------------
# 8. Prompt：返回结构化消息模板，本身不调用大模型
# ---------------------------------------------------------------------------


@mcp_server.prompt(
    name="teach-topic",
    title="讲解课程主题",
    description="为指定学习水平生成课程讲解请求。",
)
async def teach_topic(
    topic: str,
    level: str = "初学者",
) -> UserMessage:
    """渲染教学 Prompt；要求 course:read。"""

    require_scope("course:read")
    return UserMessage(
        f"请面向{level}讲解“{topic}”。先给定义，再说明数据流，"
        "最后给出一个最小 Python 示例。"
    )


# ---------------------------------------------------------------------------
# 9. 启动入口
# ---------------------------------------------------------------------------


def main() -> None:
    """初始化数据库并启动 Streamable HTTP MCP Server。"""

    initialize_database()

    print("Day16 最终 MCP Server 正在启动……")
    print("MCP 地址：", RESOURCE_URL)
    print("Reader Token：", READER_TOKEN)
    print("Operator Token：", OPERATOR_TOKEN)
    print("审计数据库：", DATABASE_PATH)
    print("按 Ctrl+C 停止服务。")

    try:
        mcp_server.run(
            transport="streamable-http",
            host=HOST,
            port=PORT,
            streamable_http_path=MCP_PATH,
            stateless_http=True,
            json_response=True,
        )
    except KeyboardInterrupt:
        print("\n收到停止信号，最终 MCP Server 已关闭。")


if __name__ == "__main__":
    main()
