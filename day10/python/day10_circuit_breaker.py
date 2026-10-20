"""Day10 第四步：为 Runnable 添加简单熔断器。"""

from enum import Enum
from typing import Callable, TypedDict

from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. 熔断状态和错误类型
# ============================================================

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TemporaryServiceError(Exception):
    """下游服务临时失败，会计入熔断失败次数。"""


class CircuitOpenError(Exception):
    """熔断器处于OPEN状态，请求被快速拒绝。"""


class ServiceRequest(TypedDict):
    text: str


class ServiceResult(TypedDict):
    processed_text: str
    service_calls: int


# ============================================================
# 2. 模拟时钟：不需要真的等待几秒
# ============================================================

class ManualClock:
    def __init__(self) -> None:
        self.current_time = 0.0

    def now(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds
        print(f"[时钟前进] 当前时间：{self.current_time:.1f} 秒")


# ============================================================
# 3. 模拟一个可能长期故障的下游服务
# ============================================================

class UnstableService:
    def __init__(self) -> None:
        self.should_fail = True
        self.call_count = 0

    def call(self, request: ServiceRequest) -> ServiceResult:
        self.call_count += 1
        print(f"[下游服务] 实际调用第 {self.call_count} 次")

        if self.should_fail:
            raise TemporaryServiceError("下游服务当前不可用。")

        return {
            "processed_text": request["text"].upper(),
            "service_calls": self.call_count
        }


# ============================================================
# 4. 熔断器
# ============================================================

class CircuitBreaker:
    def __init__(
        self,
        operation: Callable[[ServiceRequest], ServiceResult],
        clock: ManualClock,
        failure_threshold: int = 2,
        recovery_timeout: float = 5.0
    ) -> None:
        self.operation = operation
        self.clock = clock
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at: float | None = None

    def _move_to_open(self) -> None:
        self.state = CircuitState.OPEN
        self.opened_at = self.clock.now()
        print("[熔断器] CLOSED → OPEN")

    def _move_to_half_open_if_ready(self) -> None:
        if self.state != CircuitState.OPEN:
            return

        assert self.opened_at is not None
        elapsed = self.clock.now() - self.opened_at

        if elapsed >= self.recovery_timeout:
            self.state = CircuitState.HALF_OPEN
            print("[熔断器] OPEN → HALF_OPEN，允许一次探测调用")

    def _move_to_closed(self) -> None:
        previous_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at = None
        print(f"[熔断器] {previous_state.value.upper()} → CLOSED")

    def invoke(self, request: ServiceRequest) -> ServiceResult:
        """先检查熔断状态，再决定是否调用真实下游服务。"""

        self._move_to_half_open_if_ready()

        if self.state == CircuitState.OPEN:
            assert self.opened_at is not None
            remaining = (
                self.recovery_timeout
                - (self.clock.now() - self.opened_at)
            )
            raise CircuitOpenError(
                f"熔断器仍处于OPEN状态，约 {remaining:.1f} 秒后允许探测。"
            )

        try:
            result = self.operation(request)
        except TemporaryServiceError:
            self.failure_count += 1
            print(
                "[熔断器] 连续失败：",
                f"{self.failure_count}/{self.failure_threshold}"
            )

            # HALF_OPEN探测失败，或者CLOSED状态达到失败阈值，都重新OPEN。
            if (
                self.state == CircuitState.HALF_OPEN
                or self.failure_count >= self.failure_threshold
            ):
                self._move_to_open()

            raise
        else:
            # 正常调用成功或HALF_OPEN探测成功，都恢复CLOSED。
            if (
                self.state == CircuitState.HALF_OPEN
                or self.failure_count > 0
            ):
                self._move_to_closed()

            return result


# ============================================================
# 5. 辅助显示函数
# ============================================================

def show_status(
    breaker: CircuitBreaker,
    service: UnstableService
) -> None:
    print("熔断状态：", breaker.state.value)
    print("连续失败次数：", breaker.failure_count)
    print("下游服务实际调用次数：", service.call_count)


def try_invoke(
    title: str,
    runnable: RunnableLambda,
    breaker: CircuitBreaker,
    service: UnstableService
) -> None:
    print("\n" + "=" * 60)
    print(title)

    try:
        result = runnable.invoke({"text": "langchain"})
    except TemporaryServiceError as error:
        print("调用结果：下游服务失败")
        print("错误：", error)
    except CircuitOpenError as error:
        print("调用结果：熔断器快速拒绝，没有访问下游服务")
        print("错误：", error)
    else:
        print("调用结果：成功")
        print("返回数据：", result)

    show_status(breaker, service)


# ============================================================
# 6. 完整状态变化实验
# ============================================================

clock = ManualClock()
service = UnstableService()

breaker = CircuitBreaker(
    operation=service.call,
    clock=clock,
    failure_threshold=2,
    recovery_timeout=5.0
)

circuit_breaker_runnable = RunnableLambda(breaker.invoke)

try_invoke(
    "第一次失败：熔断器仍然CLOSED",
    circuit_breaker_runnable,
    breaker,
    service
)

try_invoke(
    "第二次失败：达到阈值，熔断器变成OPEN",
    circuit_breaker_runnable,
    breaker,
    service
)

try_invoke(
    "第三次请求：OPEN状态快速拒绝",
    circuit_breaker_runnable,
    breaker,
    service
)

# 冷却时间还没有结束。
clock.advance(3.0)

try_invoke(
    "三秒后再次请求：仍然快速拒绝",
    circuit_breaker_runnable,
    breaker,
    service
)

# 服务恢复，并让总时间达到五秒。
service.should_fail = False
clock.advance(2.0)

try_invoke(
    "五秒后探测：HALF_OPEN调用成功，恢复CLOSED",
    circuit_breaker_runnable,
    breaker,
    service
)

try_invoke(
    "恢复后的正常请求",
    circuit_breaker_runnable,
    breaker,
    service
)


# ============================================================
# 数据流
# ============================================================
#
# ServiceRequest
#      ↓
# CircuitBreaker.invoke()
#      ├── CLOSED → 调用下游服务
#      │      ├── 成功 → CLOSED
#      │      └── 连续失败达到阈值 → OPEN
#      ├── OPEN且冷却未结束 → 快速抛出CircuitOpenError
#      └── OPEN且冷却结束 → HALF_OPEN
#                                ├── 探测成功 → CLOSED
#                                └── 探测失败 → OPEN
#
# 快速拒绝期间不会调用下游服务，因此可以避免：
# 大量超时、无效重试、额外费用以及故障级联。
