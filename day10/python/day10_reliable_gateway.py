"""Day10 第八步（最终整合）：把限流、缓存、重试、降级、熔断和可观测性组合成一个生产级模型网关。"""

import hashlib
import json
import os
import sqlite3
import time
from enum import Enum
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.caches import BaseCache, RETURN_VAL_TYPE
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError


# ============================================================
# 1. 错误类型
# ============================================================

class TemporaryModelError(Exception):
    """主模型服务临时不可用，可以重试，也可以降级。"""


class InvalidQuestionError(Exception):
    """问题内容无效：不重试、不降级，如实报错。"""


class CircuitOpenError(Exception):
    """熔断器处于OPEN状态，主服务被本地快速拒绝。"""


RETRYABLE_ERRORS = (
    TemporaryModelError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError
)

# 熔断器OPEN时抛出的CircuitOpenError同样应该触发降级。
FALLBACK_ERRORS = RETRYABLE_ERRORS + (CircuitOpenError,)


# ============================================================
# 2. 请求和统一响应结构
# ============================================================

class GatewayRequest(TypedDict):
    question: str
    force_primary_failure: bool
    last_error: NotRequired[BaseException]


class GatewayResult(TypedDict):
    answer: str
    service: str
    model_name: str
    primary_attempts: int
    fallback_attempts: int
    fallback_reason: str


# ============================================================
# 3. SQLite缓存（第六步的安静版本）
# ============================================================

class SQLiteChatCache(BaseCache):
    """持久化缓存；verbose=False时不打印，由网关统一输出统计。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.verbose = False
        self.lookup_count = 0
        self.hit_count = 0
        self.update_count = 0

        # 缓存命中会回放旧消息的usage_metadata，
        # 记录下来用于成本核算：这些Token本次并没有消费。
        self.hit_tokens_total = 0

        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    generations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @staticmethod
    def make_key(prompt: str, llm_string: str) -> str:
        raw_key = f"{prompt}\n---MODEL---\n{llm_string}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def lookup(
        self, prompt: str, llm_string: str
    ) -> RETURN_VAL_TYPE | None:
        self.lookup_count += 1
        cache_key = self.make_key(prompt, llm_string)

        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT generations_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,)
            ).fetchone()

        if row is None:
            if self.verbose:
                print("[缓存] MISS")
            return None

        self.hit_count += 1
        saved_items = json.loads(row[0])

        for item in saved_items:
            usage = (
                item.get("message", {}).get("usage_metadata") or {}
            )
            self.hit_tokens_total += usage.get("total_tokens", 0)

        if self.verbose:
            print("[缓存] HIT")

        generations: list[ChatGeneration] = []
        for item in saved_items:
            generations.append(ChatGeneration(
                message=AIMessage.model_validate(item["message"]),
                generation_info=item.get("generation_info")
            ))

        return generations

    def update(
        self, prompt: str, llm_string: str,
        return_val: RETURN_VAL_TYPE
    ) -> None:
        self.update_count += 1
        cache_key = self.make_key(prompt, llm_string)

        saved_items = [
            {
                "message": generation.message.model_dump(mode="json"),
                "generation_info": generation.generation_info
            }
            for generation in return_val
            if isinstance(generation, ChatGeneration)
        ]

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_cache (
                    cache_key, generations_json
                ) VALUES (?, ?)
                """,
                (cache_key, json.dumps(saved_items, ensure_ascii=False))
            )

    def clear(self, **kwargs: Any) -> None:
        del kwargs
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM llm_cache")


# ============================================================
# 4. 可观测回调（第七步的精简版本）
# ============================================================

class ObservabilityHandler(BaseCallbackHandler):
    """只统计关键指标：模型调用次数、延迟、Token和错误。"""

    def __init__(self) -> None:
        self.llm_started_at: dict[str, float] = {}
        self.llm_call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.error_count = 0
        self.retry_count = 0

    def on_chain_start(
        self, serialized: dict[str, Any] | None,
        inputs: dict[str, Any], *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None, **kwargs: Any
    ) -> None:
        # 本版本RunnableRetry用retry:attempt:N标签标记重试子运行。
        if any(
            tag.startswith("retry:attempt:")
            for tag in (tags or [])
        ):
            self.retry_count += 1

    def on_chain_error(
        self, error: BaseException, *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self.error_count += 1

    def on_chat_model_start(
        self, serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]], *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        # 注意：缓存命中也会触发这里，因为运行管理器
        # 在缓存查找之前就已经打开。
        self.llm_started_at[str(run_id)] = time.perf_counter()

    def on_llm_end(
        self, response: LLMResult, *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        started_at = self.llm_started_at.pop(str(run_id), None)
        if started_at is None:
            return

        self.llm_call_count += 1

        try:
            message = response.generations[0][0].message
            usage = dict(getattr(message, "usage_metadata", None) or {})
        except (IndexError, AttributeError):
            usage = {}

        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)

        # 延迟接近零的调用基本是缓存命中；真实请求的延迟
        # 还包含限流器排队等待时间。
        latency = time.perf_counter() - started_at
        if usage:
            print(
                f"[观测] llm_end：latency={latency:.3f}s "
                f"tokens={usage.get('total_tokens', 0)}"
            )
        else:
            print(f"[观测] llm_end：latency={latency:.3f}s tokens=未知")

    def on_llm_error(
        self, error: BaseException, *,
        run_id: Any,
        parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        self.error_count += 1
        print(f"[观测] llm_error：{type(error).__name__}")


# ============================================================
# 5. 熔断器（第四步的真实时间版本）
# ============================================================

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """包住“重试后的主服务”，只统计整个主服务请求级的失败。"""

    def __init__(
        self,
        operation: RunnableLambda,
        failure_threshold: int = 2,
        recovery_timeout: float = 3.0
    ) -> None:
        self.operation = operation
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at: float | None = None
        self.fast_rejections = 0
        self.transition_log: list[str] = []

    def log_transition(self, message: str) -> None:
        elapsed = time.monotonic()
        self.transition_log.append(f"[{elapsed:7.2f}s] {message}")
        print("[熔断器]", message)

    def recovery_remaining(self) -> float:
        if self.opened_at is None:
            return 0.0
        remaining = (
            self.recovery_timeout
            - (time.monotonic() - self.opened_at)
        )
        return max(remaining, 0.0)

    def call(self, request: GatewayRequest) -> GatewayResult:
        # OPEN状态先看冷却是否结束。
        if self.state == CircuitState.OPEN:
            if self.recovery_remaining() <= 0:
                self.state = CircuitState.HALF_OPEN
                self.log_transition("OPEN → HALF_OPEN，允许一次探测")
            else:
                self.fast_rejections += 1
                raise CircuitOpenError(
                    "熔断器OPEN：主服务被本地拒绝，不发起请求。"
                )

        try:
            # operation是带重试的主服务Runnable。
            result = self.operation.invoke(request)
        except RETRYABLE_ERRORS as error:
            # InvalidQuestionError不在这里，不会计入熔断。
            self.failure_count += 1
            was_half_open = self.state == CircuitState.HALF_OPEN

            if (
                was_half_open
                or self.failure_count >= self.failure_threshold
            ):
                previous = (
                    "HALF_OPEN" if was_half_open else "CLOSED"
                )
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
                self.log_transition(
                    f"{previous} → OPEN：连续失败"
                    f"{self.failure_count}/{self.failure_threshold}"
                )
            raise error

        # 成功：HALF_OPEN探测成功或CLOSED恢复正常都回到CLOSED。
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.opened_at = None
            self.log_transition("HALF_OPEN → CLOSED：探测成功，恢复")

        return result


# ============================================================
# 6. 模型、限流器和缓存的装配
# ============================================================

load_dotenv()

script_directory = Path(__file__).resolve().parent
project_directory = script_directory.parent.parent

model_cache = SQLiteChatCache(
    project_directory / "day10" / "database" / "day10_llm_cache.db"
)

# 主服务和备用服务共用同一个限流器：
# 保护的是同一个模型服务商，而不是某一条调用路径。
shared_rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,
    check_every_n_seconds=0.05,
    max_bucket_size=2
)

# 两个服务共用同一个缓存：缓存键包含模型配置，
# 备用模型不同时自然不会读到主模型的旧回答。
primary_llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,

    # 第一层保护：单次请求的超时和关闭客户端内置重试。
    timeout=20,
    max_retries=0,

    rate_limiter=shared_rate_limiter,
    cache=model_cache
)

fallback_llm = ChatOpenAI(
    model=(
        os.getenv("DEEPSEEK_FALLBACK_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
    ),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=20,
    max_retries=0,
    rate_limiter=shared_rate_limiter,
    cache=model_cache
)


# ============================================================
# 7. 主服务和备用服务
# ============================================================

class ModelGateway:
    def __init__(self) -> None:
        self.primary_attempts = 0
        self.fallback_attempts = 0

    @staticmethod
    def validate(request: GatewayRequest) -> str:
        question = request["question"].strip()
        if not question:
            raise InvalidQuestionError("question 不能为空。")
        return question

    def make_result(
        self,
        response: Any,
        service: str,
        reason: str
    ) -> GatewayResult:
        return {
            "answer": str(response.content)[:80],
            "service": service,
            "model_name": str(
                response.response_metadata.get("model_name", "未知")
            ),
            "primary_attempts": self.primary_attempts,
            "fallback_attempts": self.fallback_attempts,
            "fallback_reason": reason
        }

    def call_primary(self, request: GatewayRequest) -> GatewayResult:
        question = self.validate(request)
        self.primary_attempts += 1

        if request["force_primary_failure"]:
            # 模拟主服务故障：发生在请求API之前，不产生费用。
            print(f"[主服务] 第{self.primary_attempts}次尝试：模拟故障")
            raise TemporaryModelError("主模型服务暂时不可用。")

        response = primary_llm.invoke(question)
        print(f"[主服务] 第{self.primary_attempts}次尝试：成功")
        return self.make_result(response, "primary", "")

    def call_fallback(self, request: GatewayRequest) -> GatewayResult:
        question = self.validate(request)
        self.fallback_attempts += 1
        last_error = request.get("last_error")

        print(f"[备用服务] 第{self.fallback_attempts}次尝试")
        print("[备用服务] 降级原因：", last_error)

        response = fallback_llm.invoke(question)
        return self.make_result(
            response, "fallback", str(last_error or "未知原因")
        )


# ============================================================
# 8. 装配完整容错链
# ============================================================

gateway = ModelGateway()
handler = ObservabilityHandler()

# 第一层：主服务有限重试（总共最多2次尝试）。
retrying_primary = RunnableLambda(gateway.call_primary).with_retry(
    retry_if_exception_type=RETRYABLE_ERRORS,
    wait_exponential_jitter=False,
    stop_after_attempt=2
)

# 第二层：熔断器包住重试后的主服务。
breaker = CircuitBreaker(
    operation=retrying_primary,
    failure_threshold=2,
    recovery_timeout=3.0
)
breaker_protected = RunnableLambda(breaker.call)

# 第三层：主链路彻底失败（或被熔断快速拒绝）后降级。
resilient_gateway = breaker_protected.with_fallbacks(
    [RunnableLambda(gateway.call_fallback)],
    exceptions_to_handle=FALLBACK_ERRORS,
    exception_key="last_error"
)


# ============================================================
# 9. 场景演示
# ============================================================

scenario_rows: list[tuple[str, str]] = []


def run_scenario(title: str, request: GatewayRequest) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("-" * 60)

    try:
        result = resilient_gateway.invoke(
            request, config={"callbacks": [handler]}
        )
    except InvalidQuestionError as error:
        scenario_rows.append(
            (title, "无效输入：如实报错，不重试、不请求模型")
        )
        print("结果：无效输入，流程停止。")
        print("错误：", error)
        # 熔断器OPEN期间，校验发生在备用服务入口；
        # 无论哪条路径，无效输入都不会到达模型。
        return
    except Exception as error:
        scenario_rows.append((title, f"最终失败：{type(error).__name__}"))
        print("结果：主服务和备用服务都失败。")
        print("错误：", type(error).__name__, error)
        return

    outcome = (
        f"由{result['service']}服务回答"
        f"（累计主尝试{result['primary_attempts']}次，"
        f"备用尝试{result['fallback_attempts']}次）"
    )
    scenario_rows.append((title, outcome))

    print("使用的服务：", result["service"])
    print("模型：", result["model_name"])
    if result["fallback_reason"]:
        print("降级原因：", result["fallback_reason"])
    print("回答：", result["answer"])
    print("熔断状态：", breaker.state.value)


question_a = "请用一句话说明生产系统为什么需要限流。"
question_b = "请用一句话说明生产系统为什么需要重试。"
question_c = "请用一句话说明生产系统为什么需要熔断。"
question_d = "请用一句话说明生产系统为什么需要降级。"

# 场景1：正常请求，主服务真实调用并写入缓存。
run_scenario("场景1：正常请求（缓存MISS→真实调用）", {
    "question": question_a,
    "force_primary_failure": False
})

# 场景2：完全相同的问题，缓存命中，不消耗Token。
run_scenario("场景2：重复请求（缓存HIT，零Token）", {
    "question": question_a,
    "force_primary_failure": False
})

# 场景3：主服务故障。重试2次都失败→熔断计数1→降级备用服务。
run_scenario("场景3：主服务故障，重试用尽后降级", {
    "question": question_b,
    "force_primary_failure": True
})

# 场景4：主服务仍然故障。再失败一次达到阈值，熔断器OPEN。
run_scenario("场景4：再次故障，熔断器变成OPEN", {
    "question": question_c,
    "force_primary_failure": True
})

# 场景5：熔断器OPEN。主服务被本地拒绝（0次尝试），直接降级。
run_scenario("场景5：OPEN状态快速拒绝，直接降级", {
    "question": question_d,
    "force_primary_failure": False
})

# 场景6：无效输入：不重试、不降级、不计入熔断失败。
run_scenario("场景6：无效输入，如实报错", {
    "question": "   ",
    "force_primary_failure": False
})

# 等待熔断冷却结束，让下一次请求进入HALF_OPEN探测。
remaining = breaker.recovery_remaining()
if remaining > 0:
    print(f"\n等待熔断冷却：还需 {remaining:.1f} 秒……")
    time.sleep(remaining + 0.1)

# 场景7：HALF_OPEN探测。用场景1的问题→缓存命中→零成本恢复CLOSED。
run_scenario("场景7：HALF_OPEN探测成功，熔断器恢复", {
    "question": question_a,
    "force_primary_failure": False
})


# ============================================================
# 10. 最终可观测性报告
# ============================================================

print("\n" + "=" * 60)
print("场景结果汇总")
print("-" * 60)
for title, outcome in scenario_rows:
    print(f"{title}")
    print(f"    → {outcome}")

print("\n" + "-" * 60)
print("模型调用统计")
print("-" * 60)

# llm_end事件包含缓存命中（缓存回放旧usage），
# 真实API次数 = 事件数 - 缓存命中次数。
real_api_calls = handler.llm_call_count - model_cache.hit_count
replayed_tokens = model_cache.hit_tokens_total
billed_total = (
    handler.total_input_tokens
    + handler.total_output_tokens
    - replayed_tokens
)

print("llm_end事件总数（含缓存命中）：", handler.llm_call_count)
print("缓存命中次数：", model_cache.hit_count)
print("真实API请求次数：", real_api_calls)
print(f"真实Token消耗：约 {billed_total}（含缓存回放"
      f"{replayed_tokens}已扣除）")
print("重试次数：", handler.retry_count)
print("错误事件数：", handler.error_count)
print("熔断快速拒绝次数：", breaker.fast_rejections)

print("\n缓存统计：")
print("lookup次数：", model_cache.lookup_count)
print("hit次数：", model_cache.hit_count)
print("write次数：", model_cache.update_count)
if model_cache.lookup_count:
    hit_rate = (
        model_cache.hit_count
        / model_cache.lookup_count * 100
    )
    print(f"命中率：{hit_rate:.1f}%")

print("\n熔断器状态轨迹：")
for line in breaker.transition_log:
    print(line)
print("最终熔断状态：", breaker.state.value)


# ============================================================
# 数据流
# ============================================================
#
# GatewayRequest
#      ↓
# with_fallbacks（最外层，收到FALLBACK_ERRORS时切换备用服务）
#      ↓
# CircuitBreaker.call（熔断器）
#      ├── OPEN且冷却未结束 → CircuitOpenError → 备用服务
#      └── CLOSED / HALF_OPEN
#             ↓
#       with_retry（主服务总共最多2次尝试）
#             ├── 成功 → 返回
#             └── RETRYABLE_ERRORS → 熔断计数 → 降级备用服务
#
# 每次真实模型请求内部（ChatOpenAI._generate_with_cache）：
#      1. 先查缓存 → HIT直接返回（不占限流令牌、不产生费用）
#      2. MISS → 向限流器申请令牌（可能排队）
#      3. 超时timeout=20内的HTTP请求
#
# InvalidQuestionError不在任何捕获范围：
# 不重试、不降级、不计入熔断，直接如实报告给调用者。