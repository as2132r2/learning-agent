"""Day19 第一步：浏览器 WebSocket 基础通信服务。

运行：
    python day19/python/day19_browser_basic_server.py

浏览器访问：
    http://127.0.0.1:8016
"""

from __future__ import annotations

from pathlib import Path

from aiohttp import WSMsgType, web


HOST = "127.0.0.1"
PORT = 8016

# __file__ 是当前 Python 文件；parents[1] 是 day19 目录。
DAY19_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY19_DIR / "html" / "day19_browser_basic.html"


async def index_handler(request: web.Request) -> web.FileResponse:
    """收到 GET / 时，把 HTML 页面返回给浏览器。"""

    return web.FileResponse(HTML_FILE)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """把 HTTP 请求升级为 WebSocket，并持续处理浏览器消息。"""

    websocket = web.WebSocketResponse(heartbeat=20)
    await websocket.prepare(request)

    peername = request.transport.get_extra_info("peername")
    print(f"[浏览器已连接] {peername}")

    await websocket.send_json(
        {
            "type": "connected",
            "message": "浏览器已经成功连接 Day19 WebSocket 服务。",
        }
    )

    try:
        async for message in websocket:
            if message.type == WSMsgType.TEXT:
                user_text = message.data.strip()
                print(f"[收到浏览器消息] {user_text!r}")

                if not user_text:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "消息不能为空。",
                        }
                    )
                    continue

                await websocket.send_json(
                    {
                        "type": "reply",
                        "original": user_text,
                        "message": f"服务端已收到：{user_text}",
                    }
                )

            elif message.type == WSMsgType.ERROR:
                print("[WebSocket 错误]", websocket.exception())
                break

    finally:
        print(f"[浏览器已断开] {peername}")

    return websocket


def create_app() -> web.Application:
    """创建 aiohttp 应用并登记 HTTP、WebSocket 路由。"""

    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    return app


def main() -> None:
    print("Day19 浏览器 WebSocket 基础服务已启动。")
    print(f"浏览器页面：http://{HOST}:{PORT}")
    print(f"WebSocket：ws://{HOST}:{PORT}/ws")
    print("按 Ctrl+C 停止服务。")

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
        print=None,
    )


if __name__ == "__main__":
    main()
