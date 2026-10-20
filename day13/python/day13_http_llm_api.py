"""Day13 第二步：把大模型封装成一个本地 HTTP API。"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


MAX_REQUEST_BYTES = 10_000
MAX_QUESTION_LENGTH = 500


load_dotenv()


def require_environment(name: str) -> str:
    """读取必需的环境变量；缺失时立即给出明确错误。"""

    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"缺少环境变量：{name}")

    return value.strip()


# 服务启动时创建一次模型对象，所有 /ask 请求复用它。
llm = ChatOpenAI(
    model=require_environment("DEEPSEEK_MODEL"),
    api_key=require_environment("DEEPSEEK_API_KEY"),
    base_url=require_environment("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=30,
    max_retries=1
)


class ApiHandler(BaseHTTPRequestHandler):
    """处理HTTP请求。"""

    server_version = "Day13LLMAPI/1.0"

    def send_json(
        self,
        status_code: int,
        payload: dict[str, Any]
    ) -> None:
        """把Python字典编码成UTF-8 JSON并发送给客户端。"""

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
        """处理GET请求。"""

        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "service": "day13-llm-api",
                "model": os.getenv("DEEPSEEK_MODEL")
            })
            return

        if self.path == "/":
            self.send_json(200, {
                "message": "Day13 LLM HTTP API 已启动",
                "endpoints": ["GET /health", "POST /ask"]
            })
            return

        self.send_json(404, {
            "error": "not_found",
            "message": f"没有这个接口：{self.path}"
        })

    def do_POST(self) -> None:
        """处理POST请求，并把 /ask 问题交给大模型。"""

        request_id = f"req-{uuid4().hex[:12]}"

        if self.path != "/ask":
            self.send_json(404, {
                "request_id": request_id,
                "error": "not_found",
                "message": f"没有这个接口：{self.path}"
            })
            return

        content_length_text = self.headers.get("Content-Length")

        if content_length_text is None:
            self.send_json(411, {
                "request_id": request_id,
                "error": "length_required",
                "message": "请求必须包含 Content-Length。"
            })
            return

        try:
            content_length = int(content_length_text)
        except ValueError:
            self.send_json(400, {
                "request_id": request_id,
                "error": "invalid_content_length",
                "message": "Content-Length 必须是整数。"
            })
            return

        if content_length < 0 or content_length > MAX_REQUEST_BYTES:
            self.send_json(413, {
                "request_id": request_id,
                "error": "request_too_large",
                "message": "请求体大小不合法，或者超过10000字节。"
            })
            return

        request_body = self.rfile.read(content_length)

        try:
            data = json.loads(request_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {
                "request_id": request_id,
                "error": "invalid_json",
                "message": "请求体必须是UTF-8 JSON。"
            })
            return

        if not isinstance(data, dict):
            self.send_json(422, {
                "request_id": request_id,
                "error": "validation_error",
                "message": "JSON最外层必须是对象。"
            })
            return

        question = data.get("question")

        if not isinstance(question, str) or not question.strip():
            self.send_json(422, {
                "request_id": request_id,
                "error": "validation_error",
                "message": "question必须是非空字符串。"
            })
            return

        question = question.strip()

        if len(question) > MAX_QUESTION_LENGTH:
            self.send_json(422, {
                "request_id": request_id,
                "error": "validation_error",
                "message": "question不能超过500个字符。"
            })
            return

        print(f"[模型调用开始] request_id={request_id} question={question!r}")

        try:
            model_response = llm.invoke([
                SystemMessage(
                    content="你是一名简洁、友好的编程学习助手。"
                ),
                HumanMessage(content=question)
            ])
        except Exception as error:
            # 服务端记录具体错误，但不把密钥等内部细节返回给客户端。
            print(
                f"[模型调用失败] request_id={request_id} "
                f"error={type(error).__name__}: {error}"
            )
            self.send_json(502, {
                "request_id": request_id,
                "error": "model_service_error",
                "message": "大模型暂时无法完成请求，请稍后重试。"
            })
            return

        print(f"[模型调用完成] request_id={request_id}")

        self.send_json(200, {
            "request_id": request_id,
            "question": question,
            "answer": model_response.content,
            "model_called": True,
            "model": os.getenv("DEEPSEEK_MODEL"),
            "usage": model_response.usage_metadata
        })

    def log_message(self, format_text: str, *args: Any) -> None:
        """输出简洁的HTTP访问日志。"""

        print(
            f"[访问日志] client={self.client_address[0]} "
            f"{format_text % args}"
        )


def run_server(host: str, port: int) -> None:
    """创建并持续运行HTTP服务器。"""

    server = ThreadingHTTPServer((host, port), ApiHandler)

    print("Day13 LLM HTTP API 已启动。")
    print(f"访问地址：http://{host}:{port}")
    print("健康检查：GET /health")
    print("模型问答：POST /ask")
    print("按 Ctrl+C 停止服务。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到停止信号。")
    finally:
        server.server_close()
        print("HTTP 服务已关闭。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day13 大模型HTTP API"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()


# ============================================================
# /ask 数据流
# ============================================================
#
# 客户端 JSON：{"question": "..."}
#          ↓ HTTP POST
# do_POST()：读取、解析、校验
#          ↓ question: str
# SystemMessage + HumanMessage
#          ↓ llm.invoke(messages)
# AIMessage
#          ↓ model_response.content / usage_metadata
# Python字典
#          ↓ send_json()
# HTTP JSON响应
