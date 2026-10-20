"""Day10 第七步：使用回调处理器记录模型调用可观测数据。"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError


# ============================================================
# 1. 错误类型（与第二步保持一致）
# ============================================================

class TemporaryModelError(Exception):
    """本地模拟的模型临时故障，用于安全演示重试。"""


class InvalidQuestionError(Exception):
    """用户问题本身无效，重复调用模型也无法修复。"""


RETRYABLE_ERRORS = (
    TemporaryModelError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError
)


# ============================================================
# 2. 输入、输出和事件结构
# ============================================================

class ModelRequest(TypedDict):
    question: str
    simulate_first_failure: bool


class ModelResult(TypedDict):
    answer: str
    usage: dict
    successful_attempt: int


class TraceEvent(TypedDict):
    created_at: str
    event_type: str          # chain_start / llm_start / llm_end / ...
    run_id: str
    parent_run_id: str
    name: str
    detail: str


# ============================================================
# 3. 把事件持久化到 SQLite
# ============================================================

class SQLiteTraceStore:
    """可观测数据落盘：程序结束后仍然可以查询历史事件。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    run_id TEXT,
                    parent_run_id TEXT,
                    name TEXT,
                    detail TEXT
                )
            """)

    def write(self, event: TraceEvent) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO trace_events (
                    created_at, event_type, run_id,
                    parent_run_id, name, detail
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event["created_at"],
                    event["event_type"],
                    event["run_id"],
                    event["parent_run_id"],
                    event["name"],
                    event["detail"]
                )
            )

    def count_by_type(self) -> dict[str, int]:
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT event_type, COUNT(*)
                FROM trace_events
                GROUP BY event_type
                """
            ).fetchall()

        return {row[0]: int(row[1]) for row in rows}


# ============================================================
# 4. 可观测回调处理器
# ============================================================

class ObservabilityHandler(BaseCallbackHandler):
    """挂到最外层Runnable上，自动收集内部所有环节的运行事件。"""

    def __init__(self, store: SQLiteTraceStore) -> None:
        self.store = store
        self.events: list[TraceEvent] = []

        # run_id → 该次模型调用的开始时间，用于计算延迟。
        self.llm_started_at: dict[str, float] = {}

        # 汇总统计。
        self.llm_call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.retry_count = 0
        self.error_count = 0

    @staticmethod
    def short_run_id(run_id: Any) -> str:
        # 当前版本的run_id是UUIDv7：前面是毫秒时间戳，
        # 同一段时间内前几位完全相同，因此展示尾部随机段。
        return str(run_id)[-8:] if run_id else "-"

    def record(
        self,
        event_type: str,
        run_id: Any,
        parent_run_id: Any,
        name: str,
        detail: str = ""
    ) -> None:
        event: TraceEvent = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "name": name,
            "detail": detail
        }
        self.events.append(event)
        self.store.write(event)

    # ---------- Runnable 链路事件 ----------

    def on_chain_start(
        self, serialized: dict[str, Any] | None,
        inputs: dict[str, Any], *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None, **kwargs: Any
    ) -> None:
        # 每个Runnable（含重试包装器）开始执行都会触发。
        # 当前版本serialized传入None，名字放在kwargs中。
        name = str(
            (serialized or {}).get("name")
            or kwargs.get("name")
            or "chain"
        )

        # 本版本RunnableRetry的重试用标签标记：
        # 重试产生的子运行会带 retry:attempt:N 标签。
        retry_tags = [
            tag for tag in (tags or [])
            if tag.startswith("retry:attempt:")
        ]
        if retry_tags:
            self.retry_count += 1

        self.record(
            "chain_start", run_id, parent_run_id, name,
            f"inputs keys={sorted(inputs.keys())} "
            f"tags={retry_tags or ''}"
        )

    def on_chain_error(
        self, error: BaseException, *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self.error_count += 1
        self.record(
            "chain_error", run_id, parent_run_id, "chain",
            f"{type(error).__name__}: {error}"
        )

    # ---------- 模型调用事件 ----------

    def on_chat_model_start(
        self, serialized: dict[str, Any],
        messages: list[list[BaseMessage]], *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        # 只在真正请求模型前触发；缓存命中不会到这里。
        self.llm_started_at[str(run_id)] = time.perf_counter()
        self.record(
            "llm_start", run_id, parent_run_id, "chat_model",
            f"messages={len(messages[0])}"
        )

    def on_llm_end(
        self, response: LLMResult, *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        run_key = str(run_id)
        started_at = self.llm_started_at.pop(run_key, None)
        latency = (
            time.perf_counter() - started_at
            if started_at is not None else -1.0
        )

        self.llm_call_count += 1
        usage = self.extract_usage(response)
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)

        self.record(
            "llm_end", run_id, parent_run_id, "chat_model",
            f"latency={latency:.3f}s tokens={usage}"
        )

    def on_llm_error(
        self, error: BaseException, *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self.error_count += 1
        self.record(
            "llm_error", run_id, parent_run_id, "chat_model",
            f"{type(error).__name__}: {error}"
        )

    # ---------- 重试事件 ----------
    #
    # 注意：当前langchain-core版本中，RunnableRetry不会触发
    # on_retry回调，而是给重试子运行打上retry:attempt:N标签，
    # 因此上面的on_chain_start通过标签统计重试。

    @staticmethod
    def extract_usage(response: LLMResult) -> dict:
        """优先读取AIMessage.usage_metadata，兼容旧字段。"""

        try:
            message = response.generations[0][0].message
            if getattr(message, "usage_metadata", None):
                return dict(message.usage_metadata)
        except (IndexError, AttributeError):
            pass

        token_usage = (response.llm_output or {}).get(
            "token_usage", {}
        )

        return {
            "input_tokens": token_usage.get("prompt_tokens", 0),
            "output_tokens": token_usage.get(
                "completion_tokens", 0
            )
        }

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("可观测性报告（本次运行）")
        print("-" * 60)
        print("真实模型调用次数：", self.llm_call_count)
        print("输入Token总量：", self.total_input_tokens)
        print("输出Token总量：", self.total_output_tokens)
        print("重试次数：", self.retry_count)
        print("错误事件数：", self.error_count)

        print("\n事件明细：")
        for event in self.events:
            short_run = self.short_run_id(event["run_id"])
            print(
                f"[{event['event_type']:>12}] "
                f"run={short_run} "
                f"{event['name']} {event['detail']}"
            )


# ============================================================
# 5. 创建模型和带重试的调用链
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=20,
    max_retries=0
)

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent
trace_store = SQLiteTraceStore(
    project_directory / "day10" / "database" / "day10_trace.db"
)


class ModelGateway:
    def __init__(self) -> None:
        self.attempt_count = 0

    def invoke_model(self, request: ModelRequest) -> ModelResult:
        self.attempt_count += 1
        question = request["question"].strip()

        if not question:
            raise InvalidQuestionError("question 不能为空。")

        if (
            request["simulate_first_failure"]
            and self.attempt_count == 1
        ):
            print("[网关] 模拟第一次连接失败，没有请求模型。")
            raise TemporaryModelError("模拟第一次连接失败。")

        response = llm.invoke(question)

        result: ModelResult = {
            "answer": str(response.content),
            "usage": dict(response.usage_metadata or {}),
            "successful_attempt": self.attempt_count
        }
        return result


# ============================================================
# 6. 实验一：带回调追踪的重试调用
# ============================================================

print("=" * 60)
print("实验一：追踪一次“失败→重试→成功”的模型调用")
print("追踪数据库：", trace_store.database_path)

handler = ObservabilityHandler(trace_store)
gateway = ModelGateway()

reliable_runnable = RunnableLambda(gateway.invoke_model).with_retry(
    retry_if_exception_type=RETRYABLE_ERRORS,
    wait_exponential_jitter=False,
    stop_after_attempt=3
)

# 回调只挂在最外层，内部所有环节的事件都会向上冒泡。
result = reliable_runnable.invoke(
    {
        "question": "请用一句话解释什么是可观测性。",
        "simulate_first_failure": True
    },
    config={"callbacks": [handler]}
)

print("最终回答：", result["answer"])
print("成功发生在第几次尝试：", result["successful_attempt"])


# ============================================================
# 7. 实验二：无效输入的错误事件
# ============================================================

print("\n" + "=" * 60)
print("实验二：无效输入被记录为chain_error")

try:
    reliable_runnable.invoke(
        {
            "question": "   ",
            "simulate_first_failure": False
        },
        config={"callbacks": [handler]}
    )
except InvalidQuestionError as error:
    print("捕获：", error)


# ============================================================
# 8. 输出报告
# ============================================================

handler.print_report()

print("\nSQLite中的事件统计：", trace_store.count_by_type())
print("程序结束后事件仍保存在 day10_trace.db，可以随时回查。")


# ============================================================
# 数据流
# ============================================================
#
# reliable_runnable.invoke(request, config={callbacks:[handler]})
#      ↓
# on_chain_start(RunnableRetry)
#      ↓
# on_chain_start(RunnableLambda) 第1次尝试
#      ↓ 失败
# on_chain_error → 子运行带 retry:attempt:2 标签再次开始
#      ↓
# on_chat_model_start（即将真正请求模型）
#      ↓
# AIMessage → on_llm_end（延迟 + Token用量）
#      ↓
# on_chain_end → 返回结果
#
# 每个事件的run_id和parent_run_id组成调用树，
# 可以还原“哪个环节慢、哪个环节失败、失败后重试了几次”。
#
# TraceEvent → SQLiteTraceStore → day10/database/day10_trace.db
#
# 生产环境的替代方案：
# langsmith已经随langchain安装。设置环境变量
#   LANGSMITH_TRACING=true
#   LANGSMITH_API_KEY=...
# 之后同样的调用会自动上报到LangSmith，
# 在网页上查看完整调用树、延迟和Token统计，
# 不需要自己维护回调处理器。
