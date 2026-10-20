"""Day13 第六步：在会话API外增加Bearer Token认证。"""

import argparse
import hmac
import os
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

from dotenv import load_dotenv

import day13_http_session_api as session_api


load_dotenv()
API_KEY = os.getenv("DAY13_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError(
        "缺少DAY13_API_KEY。请在项目根目录.env中设置DAY13_API_KEY。"
    )


class AuthenticatedApiHandler(session_api.ApiHandler):
    """先认证，再把请求交给上一版会话处理器。"""

    server_version = "Day13AuthenticatedAPI/1.0"

    def is_authorized(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        scheme, separator, token = authorization.partition(" ")

        return (
            separator == " "
            and scheme.lower() == "bearer"
            and hmac.compare_digest(token.strip(), API_KEY)
        )

    def require_authorization(self) -> bool:
        if self.is_authorized():
            return True

        self.send_json(401, {
            "error": "unauthorized",
            "message": "请提供有效的Authorization: Bearer <token>。"
        })
        return False

    def do_GET(self) -> None:
        # 健康检查公开，便于部署平台判断进程是否存活。
        if urlparse(self.path).path == "/health":
            super().do_GET()
            return

        if self.require_authorization():
            super().do_GET()

    def do_POST(self) -> None:
        if self.require_authorization():
            super().do_POST()

    def do_DELETE(self) -> None:
        if self.require_authorization():
            super().do_DELETE()


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), AuthenticatedApiHandler)
    print("Day13 Bearer认证会话API已启动。")
    print(f"访问地址：http://{host}:{port}")
    print("GET /health：无需认证")
    print("其他接口：需要Authorization: Bearer <token>")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到停止信号。")
    finally:
        server.server_close()
        session_api.database_connection.close()
        print("HTTP服务和数据库连接已关闭。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day13 Bearer认证HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
