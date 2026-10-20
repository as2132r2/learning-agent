"""Day28 第 4 步：HTTP Trace 测试客户端。

先启动服务端，再运行本文件：
    python day28/python/day28_http_tracing_client.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession


SERVER_URL = "http://127.0.0.1:8041"


@dataclass(frozen=True)
class ClientResult:
    name: str
    status: int
    response_headers: dict[str, str]
    body: dict[str, Any]


async def send_question(
    session: ClientSession,
    name: str,
    question: str,
    headers: dict[str, str] | None = None,
) -> ClientResult:
    async with session.post(
        f"{SERVER_URL}/ask",
        json={"question": question},
        headers=headers,
    ) as response:
        body = await response.json()
        selected_headers = {
            "X-Request-ID": response.headers.get("X-Request-ID", "-"),
            "X-Trace-ID": response.headers.get("X-Trace-ID", "-"),
            "X-Span-ID": response.headers.get("X-Span-ID", "-"),
        }
        return ClientResult(
            name=name,
            status=response.status,
            response_headers=selected_headers,
            body=body,
        )


async def main() -> None:
    print("Day28 HTTP Trace 客户端")
    print("服务地址：", SERVER_URL)
    print("=" * 96)

    async with ClientSession() as session:
        # 请求 1 不传关联 ID，由服务端生成。
        generated_task = asyncio.create_task(
            send_question(
                session,
                "服务端生成ID",
                "什么是 RAG？",
            ),
            name="client-generated-id",
        )

        # 请求 2 模拟上游服务已经存在一条 Trace。
        supplied_task = asyncio.create_task(
            send_question(
                session,
                "客户端传入ID",
                "RAG 的工作流程是什么？",
                headers={
                    "X-Request-ID": "req-client-demo-002",
                    "X-Trace-ID": "trace-upstream-demo-002",
                },
            ),
            name="client-supplied-id",
        )

        results = await asyncio.gather(generated_task, supplied_task)

    for result in results:
        print("\n" + "-" * 96)
        print("请求类型：", result.name)
        print("HTTP状态：", result.status)
        print("响应头：")
        print(json.dumps(result.response_headers, ensure_ascii=False, indent=2))
        print("响应正文：")
        print(json.dumps(result.body, ensure_ascii=False, indent=2))

        header_request_id = result.response_headers["X-Request-ID"]
        header_trace_id = result.response_headers["X-Trace-ID"]
        print(
            "响应头与正文的 request_id 一致：",
            header_request_id == result.body.get("request_id"),
        )
        print(
            "响应头与正文的 trace_id 一致：",
            header_trace_id == result.body.get("trace_id"),
        )

    supplied = next(item for item in results if item.name == "客户端传入ID")
    print("\n" + "=" * 96)
    print("传播验证：")
    print(
        "服务端是否保留客户端 request_id：",
        supplied.body.get("request_id") == "req-client-demo-002",
    )
    print(
        "服务端是否保留上游 trace_id：",
        supplied.body.get("trace_id") == "trace-upstream-demo-002",
    )

    print("\n本步数据流：")
    print(
        "客户端HTTP请求\n"
        "→ 可选X-Request-ID、X-Trace-ID请求头\n"
        "→ tracing_middleware校验或生成ID\n"
        "→ ContextVar绑定 → HTTP根Span\n"
        "→ retrieve_documents子Span → generate_answer子Span\n"
        "→ JSON响应 + X-Request-ID/X-Trace-ID/X-Span-ID响应头\n"
        "→ finally清理服务端请求上下文"
    )


if __name__ == "__main__":
    asyncio.run(main())
