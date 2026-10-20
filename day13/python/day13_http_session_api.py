"""Day13 第五步：带查询、删除能力的HTTP会话管理API。"""

import argparse
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph


MAX_REQUEST_BYTES = 10_000
MAX_QUESTION_LENGTH = 500
MAX_THREAD_ID_LENGTH = 100

DAY13_DIRECTORY = Path(__file__).resolve().parents[1]
DATABASE_PATH = DAY13_DIRECTORY / "database" / "day13_http_sessions.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

load_dotenv()


def require_environment(name: str) -> str:
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
    response = llm.invoke([
        SystemMessage(content="你是一名简洁、友好的编程学习助手。"),
        *state["messages"]
    ])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")

database_connection = sqlite3.connect(
    DATABASE_PATH,
    check_same_thread=False
)
checkpointer = SqliteSaver(database_connection)
graph = builder.compile(checkpointer=checkpointer)
graph_lock = Lock()


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """把LangChain消息转换成可以JSON序列化的字典。"""

    return {
        "type": message.type,
        "content": message.content,
        "id": message.id
    }


def validate_thread_id(value: Any) -> tuple[str | None, str | None]:
    """校验thread_id，返回(有效值, 错误信息)。"""

    if not isinstance(value, str) or not value.strip():
        return None, "thread_id必须是非空字符串。"

    thread_id = value.strip()
    if len(thread_id) > MAX_THREAD_ID_LENGTH:
        return None, "thread_id不能超过100个字符。"

    return thread_id, None


class ApiHandler(BaseHTTPRequestHandler):
    """聊天和会话管理HTTP处理器。"""

    server_version = "Day13SessionAPI/1.0"

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

    def query_thread_id(self) -> tuple[str | None, str | None]:
        """从URL查询参数中读取并校验thread_id。"""

        parsed_url = urlparse(self.path)
        query = parse_qs(parsed_url.query)
        value = query.get("thread_id", [None])[0]
        return validate_thread_id(value)

    def do_GET(self) -> None:
        """处理健康检查和历史查询。"""

        request_id = f"req-{uuid4().hex[:12]}"
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "service": "day13-session-api",
                "memory": "SqliteSaver",
                "database": str(DATABASE_PATH)
            })
            return

        if parsed_url.path == "/history":
            thread_id, error = self.query_thread_id()
            if error is not None:
                self.send_error_json(
                    422, request_id, "validation_error", error
                )
                return

            config = {"configurable": {"thread_id": thread_id}}

            with graph_lock:
                snapshot = graph.get_state(config)

            messages = snapshot.values.get("messages", [])
            if not messages:
                self.send_error_json(
                    404, request_id, "thread_not_found",
                    f"没有找到会话：{thread_id}"
                )
                return

            self.send_json(200, {
                "request_id": request_id,
                "thread_id": thread_id,
                "message_count": len(messages),
                "messages": [message_to_dict(item) for item in messages]
            })
            return

        if parsed_url.path == "/":
            self.send_json(200, {
                "message": "Day13会话管理API已启动",
                "endpoints": [
                    "GET /health",
                    "POST /chat",
                    "GET /history?thread_id=...",
                    "DELETE /history?thread_id=..."
                ]
            })
            return

        self.send_error_json(
            404, request_id, "not_found",
            f"没有这个接口：{parsed_url.path}"
        )

    def do_DELETE(self) -> None:
        """删除指定thread_id的全部LangGraph检查点。"""

        request_id = f"req-{uuid4().hex[:12]}"
        parsed_url = urlparse(self.path)

        if parsed_url.path != "/history":
            self.send_error_json(
                404, request_id, "not_found",
                f"没有这个接口：{parsed_url.path}"
            )
            return

        thread_id, error = self.query_thread_id()
        if error is not None:
            self.send_error_json(422, request_id, "validation_error", error)
            return

        config = {"configurable": {"thread_id": thread_id}}

        with graph_lock:
            snapshot = graph.get_state(config)
            messages = snapshot.values.get("messages", [])

            if messages:
                checkpointer.delete_thread(thread_id)

        if not messages:
            self.send_error_json(
                404, request_id, "thread_not_found",
                f"没有找到会话：{thread_id}"
            )
            return

        print(f"[会话删除] request_id={request_id} thread_id={thread_id!r}")
        self.send_json(200, {
            "request_id": request_id,
            "thread_id": thread_id,
            "deleted": True,
            "deleted_message_count": len(messages)
        })

    def do_POST(self) -> None:
        """处理POST /chat。"""

        request_id = f"req-{uuid4().hex[:12]}"
        parsed_url = urlparse(self.path)

        if parsed_url.path != "/chat":
            self.send_error_json(
                404, request_id, "not_found",
                f"没有这个接口：{parsed_url.path}"
            )
            return

        length_text = self.headers.get("Content-Length")
        if length_text is None:
            self.send_error_json(
                411, request_id, "length_required",
                "请求必须包含Content-Length。"
            )
            return

        try:
            content_length = int(length_text)
        except ValueError:
            self.send_error_json(
                400, request_id, "invalid_content_length",
                "Content-Length必须是整数。"
            )
            return

        if content_length < 0 or content_length > MAX_REQUEST_BYTES:
            self.send_error_json(
                413, request_id, "request_too_large",
                "请求体大小不合法，或者超过10000字节。"
            )
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
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

        thread_id, error = validate_thread_id(data.get("thread_id"))
        if error is not None:
            self.send_error_json(422, request_id, "validation_error", error)
            return

        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            self.send_error_json(
                422, request_id, "validation_error",
                "question必须是非空字符串。"
            )
            return

        question = question.strip()
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
        except Exception as error_value:
            print(
                f"[对话失败] request_id={request_id} "
                f"error={type(error_value).__name__}: {error_value}"
            )
            self.send_error_json(
                502, request_id, "model_service_error",
                "大模型暂时无法完成请求，请稍后重试。"
            )
            return

        messages = result["messages"]
        self.send_json(200, {
            "request_id": request_id,
            "thread_id": thread_id,
            "question": question,
            "answer": messages[-1].content,
            "message_count": len(messages),
            "memory": "sqlite"
        })

    def log_message(self, format_text: str, *args: Any) -> None:
        print(
            f"[访问日志] client={self.client_address[0]} "
            f"{format_text % args}"
        )


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), ApiHandler)
    print("Day13 HTTP会话管理API已启动。")
    print(f"访问地址：http://{host}:{port}")
    print("状态数据库：", DATABASE_PATH)
    print("POST /chat：发送消息")
    print("GET /history?thread_id=...：查询历史")
    print("DELETE /history?thread_id=...：删除历史")
    print("按Ctrl+C停止服务。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到停止信号。")
    finally:
        server.server_close()
        database_connection.close()
        print("HTTP服务和数据库连接已关闭。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day13 HTTP会话管理API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()


# POST /chat                   → 创建或继续会话
# GET /history?thread_id=abc   → graph.get_state()读取历史
# DELETE /history?thread_id=abc→ checkpointer.delete_thread()删除历史
