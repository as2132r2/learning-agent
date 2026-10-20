"""Day13 第四步：使用SQLite持久化多会话HTTP聊天记录。"""

import argparse
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph


MAX_REQUEST_BYTES = 10_000
MAX_QUESTION_LENGTH = 500
MAX_THREAD_ID_LENGTH = 100

DAY13_DIRECTORY = Path(__file__).resolve().parents[1]
DATABASE_PATH = DAY13_DIRECTORY / "database" / "day13_http_memory.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

load_dotenv()


def require_environment(name: str) -> str:
    """读取必需环境变量。"""

    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"缺少环境变量：{name}")
    return value.strip()


llm = ChatOpenAI(
    model=require_environment("DEEPSEEK_MODEL"),
    api_key=require_environment("DEEPSEEK_API_KEY"),
    base_url=require_environment("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=30,
    max_retries=1
)


def chatbot(state: MessagesState) -> dict[str, list]:
    """把历史消息和本轮问题交给模型。"""

    response = llm.invoke([
        SystemMessage(content="你是一名简洁、友好的编程学习助手。"),
        *state["messages"]
    ])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")

# check_same_thread=False允许HTTP工作线程共用连接；graph_lock负责串行访问。
database_connection = sqlite3.connect(
    DATABASE_PATH,
    check_same_thread=False
)
checkpointer = SqliteSaver(database_connection)
graph = builder.compile(checkpointer=checkpointer)
graph_lock = Lock()


class ApiHandler(BaseHTTPRequestHandler):
    """处理持久化聊天API请求。"""

    server_version = "Day13PersistentMemoryAPI/1.0"

    def send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        response_body = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def send_error_json(
        self,
        status_code: int,
        request_id: str,
        error_name: str,
        message: str
    ) -> None:
        self.send_json(status_code, {
            "request_id": request_id,
            "error": error_name,
            "message": message
        })

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "service": "day13-persistent-memory-api",
                "memory": "SqliteSaver",
                "database": str(DATABASE_PATH),
                "model": os.getenv("DEEPSEEK_MODEL")
            })
            return

        if self.path == "/":
            self.send_json(200, {
                "message": "Day13 SQLite记忆API已启动",
                "endpoints": ["GET /health", "POST /chat"]
            })
            return

        self.send_json(404, {
            "error": "not_found",
            "message": f"没有这个接口：{self.path}"
        })

    def do_POST(self) -> None:
        request_id = f"req-{uuid4().hex[:12]}"

        if self.path != "/chat":
            self.send_error_json(
                404, request_id, "not_found", f"没有这个接口：{self.path}"
            )
            return

        content_length_text = self.headers.get("Content-Length")
        if content_length_text is None:
            self.send_error_json(
                411, request_id, "length_required",
                "请求必须包含 Content-Length。"
            )
            return

        try:
            content_length = int(content_length_text)
        except ValueError:
            self.send_error_json(
                400, request_id, "invalid_content_length",
                "Content-Length 必须是整数。"
            )
            return

        if content_length < 0 or content_length > MAX_REQUEST_BYTES:
            self.send_error_json(
                413, request_id, "request_too_large",
                "请求体大小不合法，或者超过10000字节。"
            )
            return

        try:
            request_body = self.rfile.read(content_length)
            data = json.loads(request_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error_json(
                400, request_id, "invalid_json",
                "请求体必须是UTF-8 JSON。"
            )
            return

        if not isinstance(data, dict):
            self.send_error_json(
                422, request_id, "validation_error",
                "JSON最外层必须是对象。"
            )
            return

        thread_id = data.get("thread_id")
        question = data.get("question")

        if not isinstance(thread_id, str) or not thread_id.strip():
            self.send_error_json(
                422, request_id, "validation_error",
                "thread_id必须是非空字符串。"
            )
            return

        if not isinstance(question, str) or not question.strip():
            self.send_error_json(
                422, request_id, "validation_error",
                "question必须是非空字符串。"
            )
            return

        thread_id = thread_id.strip()
        question = question.strip()

        if len(thread_id) > MAX_THREAD_ID_LENGTH:
            self.send_error_json(
                422, request_id, "validation_error",
                "thread_id不能超过100个字符。"
            )
            return

        if len(question) > MAX_QUESTION_LENGTH:
            self.send_error_json(
                422, request_id, "validation_error",
                "question不能超过500个字符。"
            )
            return

        config = {"configurable": {"thread_id": thread_id}}

        print(
            f"[对话开始] request_id={request_id} "
            f"thread_id={thread_id!r} question={question!r}"
        )

        try:
            with graph_lock:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=question)]},
                    config=config
                )
        except Exception as error:
            print(
                f"[对话失败] request_id={request_id} "
                f"error={type(error).__name__}: {error}"
            )
            self.send_error_json(
                502, request_id, "model_service_error",
                "大模型暂时无法完成请求，请稍后重试。"
            )
            return

        latest_message = result["messages"][-1]
        message_count = len(result["messages"])

        print(
            f"[对话完成] request_id={request_id} "
            f"thread_id={thread_id!r} message_count={message_count}"
        )

        self.send_json(200, {
            "request_id": request_id,
            "thread_id": thread_id,
            "question": question,
            "answer": latest_message.content,
            "message_count": message_count,
            "memory": "sqlite",
            "database": str(DATABASE_PATH)
        })

    def log_message(self, format_text: str, *args: Any) -> None:
        print(
            f"[访问日志] client={self.client_address[0]} "
            f"{format_text % args}"
        )


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), ApiHandler)

    print("Day13 SQLite持久化记忆API已启动。")
    print(f"访问地址：http://{host}:{port}")
    print("对话接口：POST /chat")
    print("状态数据库：", DATABASE_PATH)
    print("按 Ctrl+C 停止服务。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到停止信号。")
    finally:
        server.server_close()
        database_connection.close()
        print("HTTP服务和数据库连接已关闭。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day13 SQLite持久化记忆HTTP API"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()


# ============================================================
# 数据流
# ============================================================
# POST /chat(thread_id, question)
#              ↓
# graph.invoke(..., config中的thread_id)
#              ↓
# SqliteSaver从day13_http_memory.db读取旧MessagesState
#              ↓
# chatbot(旧消息 + 新HumanMessage) → llm.invoke()
#              ↓
# 新AIMessage写入SQLite检查点
#              ↓
# JSON响应；服务重启后仍可按相同thread_id恢复
