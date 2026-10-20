"""Day13 第七步：在认证会话API中增加SSE流式回答。"""

import argparse
import json
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

import day13_http_auth_api as auth_api
import day13_http_session_api as session_api


class StreamingApiHandler(auth_api.AuthenticatedApiHandler):
    """保留原接口，并增加POST /stream。"""

    server_version = "Day13StreamingAPI/1.0"

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/stream":
            super().do_POST()
            return

        if not self.require_authorization():
            return

        length_text = self.headers.get("Content-Length")
        try:
            content_length = int(length_text or "")
        except ValueError:
            self.send_json(400, {"error": "invalid_content_length"})
            return

        if content_length < 1 or content_length > session_api.MAX_REQUEST_BYTES:
            self.send_json(413, {"error": "request_too_large"})
            return

        try:
            data = json.loads(
                self.rfile.read(content_length).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "invalid_json"})
            return

        question = data.get("question") if isinstance(data, dict) else None
        if not isinstance(question, str) or not question.strip():
            self.send_json(422, {
                "error": "validation_error",
                "message": "question必须是非空字符串。"
            })
            return

        question = question.strip()
        if len(question) > session_api.MAX_QUESTION_LENGTH:
            self.send_json(422, {
                "error": "validation_error",
                "message": "question不能超过500个字符。"
            })
            return

        request_id = f"req-{uuid4().hex[:12]}"

        # SSE响应不设置Content-Length，因为总长度事先未知。
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        print(f"[流式调用开始] request_id={request_id}")

        try:
            for chunk in session_api.llm.stream([
                SystemMessage(content="你是一名简洁、友好的编程学习助手。"),
                HumanMessage(content=question)
            ]):
                if not chunk.content:
                    continue

                event = {
                    "request_id": request_id,
                    "delta": chunk.content
                }
                event_text = (
                    "event: token\n"
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                )
                self.wfile.write(event_text.encode("utf-8"))
                self.wfile.flush()

            done_text = (
                "event: done\n"
                f"data: {json.dumps({'request_id': request_id}, ensure_ascii=False)}\n\n"
            )
            self.wfile.write(done_text.encode("utf-8"))
            self.wfile.flush()
            print(f"[流式调用完成] request_id={request_id}")

        except (BrokenPipeError, ConnectionResetError):
            print(f"[客户端提前断开] request_id={request_id}")
        except Exception as error:
            error_text = (
                "event: error\n"
                "data: {\"message\": \"模型流式调用失败\"}\n\n"
            )
            try:
                self.wfile.write(error_text.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            print(
                f"[流式调用失败] request_id={request_id} "
                f"error={type(error).__name__}: {error}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Day13 SSE流式HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), StreamingApiHandler)
    print("Day13 SSE流式API已启动。")
    print(f"访问地址：http://{args.host}:{args.port}")
    print("POST /stream：流式问答")
    print("原有/chat、/history、/health接口继续可用。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到停止信号。")
    finally:
        server.server_close()
        session_api.database_connection.close()
        print("HTTP服务和数据库连接已关闭。")


if __name__ == "__main__":
    main()
