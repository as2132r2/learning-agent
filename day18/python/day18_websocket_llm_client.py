"""Day18 第二步（客户端）：接收 WebSocket 模型文本流。

先启动服务端：
    python day18/python/day18_websocket_llm_server.py

再运行客户端：
    python day18/python/day18_websocket_llm_client.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout, WSMsgType


WEBSOCKET_URL = "ws://127.0.0.1:8011/ws"


async def main() -> None:
    """发送一个问题，直到收到 done 或 error。"""

    question = input("请输入问题：").strip()
    if not question:
        print("问题不能为空。")
        return

    print("Day18 WebSocket 模型流客户端")
    print("服务地址：", WEBSOCKET_URL)
    print("问题：", question)
    print("=" * 72)

    timeout = ClientTimeout(total=None, connect=10)
    text_parts: list[str] = []
    token_count = 0
    started_at = time.perf_counter()
    first_token_at: float | None = None

    async with ClientSession(timeout=timeout) as session:
        async with session.ws_connect(
            WEBSOCKET_URL,
            heartbeat=20,
        ) as websocket:
            connected = await websocket.receive_json()
            print("连接消息：", connected)

            await websocket.send_json(
                {
                    "type": "ask",
                    "question": question,
                }
            )

            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    data: Any = message.json()
                    if not isinstance(data, dict):
                        print("服务端返回的 JSON 不是 object。")
                        continue

                    message_type = data.get("type")

                    if message_type == "start":
                        print("\n助手：", end="", flush=True)

                    elif message_type == "token":
                        if first_token_at is None:
                            first_token_at = time.perf_counter()

                        text = str(data.get("text", ""))
                        token_count += 1
                        text_parts.append(text)
                        print(text, end="", flush=True)

                    elif message_type == "done":
                        print("\n\n[收到 done]")
                        print("服务端 chunk_count：", data.get("chunk_count"))
                        print("服务端 answer_length：", data.get("answer_length"))
                        break

                    elif message_type == "error":
                        print("\n\n[收到 error]")
                        print("错误类型：", data.get("error_type"))
                        print("错误信息：", data.get("message"))
                        break

                elif message.type == WSMsgType.ERROR:
                    print("WebSocket 错误：", websocket.exception())
                    break

            await websocket.close()

    finished_at = time.perf_counter()
    complete_answer = "".join(text_parts)

    print("\n" + "=" * 72)
    print("客户端统计")
    print("token 消息数：", token_count)
    print("完整回答长度：", len(complete_answer))
    print("完整回答：")
    print(complete_answer)

    if first_token_at is not None:
        print(
            "首个 token 等待：",
            round(first_token_at - started_at, 3),
            "秒",
        )
    print("总耗时：", round(finished_at - started_at, 3), "秒")

    print("\n本步数据流：")
    print("客户端 send_json({type: ask, question})")
    print("→ 服务端 websocket.receive()")
    print("→ llm.astream(messages)")
    print("→ AIMessageChunk")
    print("→ send_json({type: token, text})")
    print("→ 客户端逐条接收并立即打印")
    print("→ send_json({type: done})")


if __name__ == "__main__":
    asyncio.run(main())
