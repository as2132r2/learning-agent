"""Day13 第一步：使用Python标准库创建最小HTTP API。"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4


MAX_REQUEST_BYTES = 10_000
MAX_QUESTION_LENGTH = 500


class ApiHandler(BaseHTTPRequestHandler):
    """处理GET和POST请求。"""

    server_version = "Day13LearningAPI/1.0"

    def send_json(
        self,
        status_code: int,
        payload: dict[str, Any]
    ) -> None:
        """统一把Python字典转换成JSON响应。"""

        response_body = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_GET(self) -> None:
        """处理GET请求。方法名由BaseHTTPRequestHandler规定。"""

        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "service": "day13-learning-api"
            })
            return

        if self.path == "/":
            self.send_json(200, {
                "message": "Day13 HTTP API 已启动",
                "endpoints": ["GET /health", "POST /ask"]
            })
            return

        self.send_json(404, {
            "error": "not_found",
            "message": f"没有这个接口：{self.path}"
        })

    def do_POST(self) -> None:
        """处理POST请求。"""

        if self.path != "/ask":
            self.send_json(404, {
                "error": "not_found",
                "message": f"没有这个接口：{self.path}"
            })
            return

        content_length_text = self.headers.get("Content-Length")

        if content_length_text is None:
            self.send_json(411, {
                "error": "length_required",
                "message": "请求必须包含 Content-Length。"
            })
            return

        try:
            content_length = int(content_length_text)
        except ValueError:
            self.send_json(400, {
                "error": "invalid_content_length",
                "message": "Content-Length 必须是整数。"
            })
            return

        if content_length > MAX_REQUEST_BYTES:
            self.send_json(413, {
                "error": "request_too_large",
                "message": "请求体不能超过10000字节。"
            })
            return

        request_body = self.rfile.read(content_length)

        try:
            data = json.loads(request_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {
                "error": "invalid_json",
                "message": "请求体必须是UTF-8 JSON。"
            })
            return

        if not isinstance(data, dict):
            self.send_json(422, {
                "error": "validation_error",
                "message": "JSON最外层必须是对象。"
            })
            return

        question = data.get("question")

        if not isinstance(question, str) or not question.strip():
            self.send_json(422, {
                "error": "validation_error",
                "message": "question必须是非空字符串。"
            })
            return

        question = question.strip()

        if len(question) > MAX_QUESTION_LENGTH:
            self.send_json(422, {
                "error": "validation_error",
                "message": "question不能超过500个字符。"
            })
            return

        request_id = f"req-{uuid4().hex[:12]}"

        # 第一步先返回确定性结果；下一步再连接真实模型。
        self.send_json(200, {
            "request_id": request_id,
            "question": question,
            "answer": f"服务器已经收到你的问题：{question}",
            "model_called": False
        })

    def log_message(self, format_text: str, *args: Any) -> None:
        """把默认访问日志改成简洁格式。"""

        print(
            f"[访问日志] client={self.client_address[0]} "
            f"{format_text % args}"
        )


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), ApiHandler)

    print("Day13 HTTP API 已启动。")
    print(f"访问地址：http://{host}:{port}")
    print("健康检查：GET /health")
    print("问答接口：POST /ask")
    print("按 Ctrl+C 停止服务。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到停止信号。")
    finally:
        server.server_close()
        print("HTTP 服务已关闭。")


parser = argparse.ArgumentParser(
    description="Day13 最小HTTP API"
)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

run_server(args.host, args.port)


# ============================================================
# 数据流
# ============================================================
#
# curl / 浏览器 / 客户端
#          ↓ HTTP请求
# ThreadingHTTPServer
#          ↓
# ApiHandler.do_GET() / do_POST()
#          ↓
# 路由判断 + JSON解析 + 输入校验
#          ↓
# send_json(status_code, payload)
#          ↓ HTTP JSON响应
# 客户端
