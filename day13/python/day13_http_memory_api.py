"""Day13 第三步：带多会话记忆的本地大模型 HTTP API。"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph


MAX_REQUEST_BYTES = 10_000
MAX_QUESTION_LENGTH = 500
MAX_THREAD_ID_LENGTH = 100


load_dotenv()


def require_environment(name: str) -> str:
    """读取必需环境变量，缺失时在启动阶段立即报错。"""

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
    """接收历史消息，调用模型，返回本轮新增的AI消息。"""

    model_messages = [
        SystemMessage(
            content="你是一名简洁、友好的编程学习助手。"
        ),
        *state["messages"]
    ]

    model_response = llm.invoke(model_messages)

    return {"messages": [model_response]}


# graph只在服务启动时构建一次。
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")

# InMemorySaver按照thread_id保存每个会话的MessagesState。
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ThreadingHTTPServer可能同时处理多个请求；本练习用锁保护图调用。
graph_lock = Lock()


class ApiHandler(BaseHTTPRequestHandler):
    """处理健康检查和带记忆的问答请求。"""

    server_version = "Day13MemoryAPI/1.0"

    def send_json(
        self,
        status_code: int,
        payload: dict[str, Any]
    ) -> None:
        """把Python字典转换成UTF-8 JSON响应。"""

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
        """统一发送错误响应。"""

        self.send_json(status_code, {
            "request_id": request_id,
            "error": error_name,
            "message": message
        })

    def do_GET(self) -> None:
        """处理GET请求。"""

        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "service": "day13-memory-api",
                "memory": "InMemorySaver",
                "model": os.getenv("DEEPSEEK_MODEL")
            })
            return

        if self.path == "/":
            self.send_json(200, {
                "message": "Day13 多会话记忆 API 已启动",
                "endpoints": ["GET /health", "POST /chat"]
            })
            return

        self.send_json(404, {
            "error": "not_found",
            "message": f"没有这个接口：{self.path}"
        })

    def do_POST(self) -> None:
        """处理POST /chat请求。"""

        request_id = f"req-{uuid4().hex[:12]}"

        if self.path != "/chat":
            self.send_error_json(
                404,
                request_id,
                "not_found",
                f"没有这个接口：{self.path}"
            )
            return

        content_length_text = self.headers.get("Content-Length")

        if content_length_text is None:
            self.send_error_json(
                411,
                request_id,
                "length_required",
                "请求必须包含 Content-Length。"
            )
            return

        try:
            content_length = int(content_length_text)
        except ValueError:
            self.send_error_json(
                400,
                request_id,
                "invalid_content_length",
                "Content-Length 必须是整数。"
            )
            return

        if content_length < 0 or content_length > MAX_REQUEST_BYTES:
            self.send_error_json(
                413,
                request_id,
                "request_too_large",
                "请求体大小不合法，或者超过10000字节。"
            )
            return

        request_body = self.rfile.read(content_length)

        try:
            data = json.loads(request_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error_json(
                400,
                request_id,
                "invalid_json",
                "请求体必须是UTF-8 JSON。"
            )
            return

        if not isinstance(data, dict):
            self.send_error_json(
                422,
                request_id,
                "validation_error",
                "JSON最外层必须是对象。"
            )
            return

        thread_id = data.get("thread_id")
        question = data.get("question")

        if not isinstance(thread_id, str) or not thread_id.strip():
            self.send_error_json(
                422,
                request_id,
                "validation_error",
                "thread_id必须是非空字符串。"
            )
            return

        if not isinstance(question, str) or not question.strip():
            self.send_error_json(
                422,
                request_id,
                "validation_error",
                "question必须是非空字符串。"
            )
            return

        thread_id = thread_id.strip()
        question = question.strip()

        if len(thread_id) > MAX_THREAD_ID_LENGTH:
            self.send_error_json(
                422,
                request_id,
                "validation_error",
                "thread_id不能超过100个字符。"
            )
            return

        if len(question) > MAX_QUESTION_LENGTH:
            self.send_error_json(
                422,
                request_id,
                "validation_error",
                "question不能超过500个字符。"
            )
            return

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        print(
            f"[对话开始] request_id={request_id} "
            f"thread_id={thread_id!r} question={question!r}"
        )

        try:
            # 锁保证同一时刻只有一个线程更新图状态。
            with graph_lock:
                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content=question)
                        ]
                    },
                    config=config
                )
        except Exception as error:
            print(
                f"[对话失败] request_id={request_id} "
                f"error={type(error).__name__}: {error}"
            )
            self.send_error_json(
                502,
                request_id,
                "model_service_error",
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
            "memory": "in_memory"
        })

    def log_message(self, format_text: str, *args: Any) -> None:
        """输出简洁的HTTP访问日志。"""

        print(
            f"[访问日志] client={self.client_address[0]} "
            f"{format_text % args}"
        )


def run_server(host: str, port: int) -> None:
    """启动HTTP服务器。"""

    server = ThreadingHTTPServer((host, port), ApiHandler)

    print("Day13 多会话记忆 API 已启动。")
    print(f"访问地址：http://{host}:{port}")
    print("健康检查：GET /health")
    print("对话接口：POST /chat")
    print("记忆方式：InMemorySaver（服务关闭后清空）")
    print("按 Ctrl+C 停止服务。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到停止信号。")
    finally:
        server.server_close()
        print("HTTP 服务已关闭，内存对话记录已清空。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day13 多会话记忆HTTP API"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()


# ============================================================
# POST /chat 数据流
# ============================================================
#
# {"thread_id": "xiaolin", "question": "..."}
#                       ↓
# HTTP解析与参数校验
#                       ↓
# config = {"configurable": {"thread_id": "xiaolin"}}
#                       ↓
# graph.invoke({"messages": [HumanMessage(...)]}, config)
#                       ↓
# InMemorySaver根据thread_id读取历史MessagesState
#                       ↓
# chatbot(state) → llm.invoke(历史消息)
#                       ↓
# AIMessage写回同一个thread_id的状态
#                       ↓
# answer + message_count → HTTP JSON响应
