"""Day10 第五步：使用 InMemoryRateLimiter 控制请求速率。"""

import time
from typing import TypedDict

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. 输入、输出和限流错误
# ============================================================

class TaskRequest(TypedDict):
    task_id: int
    text: str


class TaskResult(TypedDict):
    task_id: int
    processed_text: str
    started_after_seconds: float


class LocalRateLimitError(Exception):
    """非阻塞模式没有获得令牌时抛出的本地限流错误。"""


# ============================================================
# 2. 带限流保护的本地服务
# ============================================================

class RateLimitedTextService:
    def __init__(
        self,
        rate_limiter: InMemoryRateLimiter,
        blocking: bool
    ) -> None:
        self.rate_limiter = rate_limiter
        self.blocking = blocking
        self.started_at = time.perf_counter()
        self.actual_call_count = 0

    def process(self, request: TaskRequest) -> TaskResult:
        """必须先拿到令牌，才能执行真正的服务操作。"""

        print(f"任务 {request['task_id']} 正在申请令牌……")

        acquired = self.rate_limiter.acquire(
            blocking=self.blocking
        )

        if not acquired:
            raise LocalRateLimitError(
                f"任务 {request['task_id']} 没有获得令牌。"
            )

        self.actual_call_count += 1
        elapsed = time.perf_counter() - self.started_at

        print(
            f"任务 {request['task_id']} 获得令牌，"
            f"在 {elapsed:.2f} 秒时开始执行"
        )

        return {
            "task_id": request["task_id"],
            "processed_text": request["text"].upper(),
            "started_after_seconds": round(elapsed, 2)
        }


# ============================================================
# 3. 实验一：阻塞模式，等待令牌后继续执行
# ============================================================

print("=" * 60)
print("实验一：阻塞限流")
print("规则：每秒放行2个请求，令牌桶最多保存1个令牌。")

blocking_limiter = InMemoryRateLimiter(
    # 平均每秒补充2个令牌，即约每0.5秒产生一个令牌。
    requests_per_second=2,

    # 每0.05秒检查一次是否出现新令牌。
    check_every_n_seconds=0.05,

    # 最多积累1个令牌，不允许一次突发大量请求。
    max_bucket_size=1
)

blocking_service = RateLimitedTextService(
    rate_limiter=blocking_limiter,
    blocking=True
)

blocking_runnable = RunnableLambda(blocking_service.process)

blocking_results: list[TaskResult] = []

for task_number in range(1, 5):
    result = blocking_runnable.invoke({
        "task_id": task_number,
        "text": f"request-{task_number}"
    })
    blocking_results.append(result)

print("\n全部结果：")

for result in blocking_results:
    print(result)

print("实际服务调用次数：", blocking_service.actual_call_count)


# ============================================================
# 4. 实验二：非阻塞模式，没有令牌就立即拒绝
# ============================================================

print("\n" + "=" * 60)
print("实验二：非阻塞限流")

non_blocking_limiter = InMemoryRateLimiter(
    requests_per_second=2,
    check_every_n_seconds=0.05,
    max_bucket_size=1
)

non_blocking_service = RateLimitedTextService(
    rate_limiter=non_blocking_limiter,
    blocking=False
)

non_blocking_runnable = RunnableLambda(
    non_blocking_service.process
)

try:
    non_blocking_runnable.invoke({
        "task_id": 1,
        "text": "first-immediate-request"
    })
except LocalRateLimitError as error:
    print("第一次立即请求被拒绝：", error)

print("等待0.55秒，让限流器产生一个令牌……")
time.sleep(0.55)

try:
    result = non_blocking_runnable.invoke({
        "task_id": 2,
        "text": "request-after-waiting"
    })
except LocalRateLimitError as error:
    print("第二次请求仍然被拒绝：", error)
else:
    print("第二次请求成功：", result)

print(
    "非阻塞实验中的实际服务调用次数：",
    non_blocking_service.actual_call_count
)


# ============================================================
# 5. 接入 ChatOpenAI 时的形式
# ============================================================
#
# ChatOpenAI本身支持rate_limiter参数：
#
# llm = ChatOpenAI(
#     ...,
#     rate_limiter=InMemoryRateLimiter(
#         requests_per_second=2,
#         check_every_n_seconds=0.05,
#         max_bucket_size=2
#     )
# )
#
# 之后每次llm.invoke()都会先经过这个限流器。


# ============================================================
# 数据流
# ============================================================
#
# TaskRequest
#     ↓
# rate_limiter.acquire()
#     ├── blocking=True
#     │      └── 没有令牌 → 等待 → 获得令牌 → 执行服务
#     └── blocking=False
#            ├── 有令牌 → 执行服务
#            └── 没有令牌 → 立即返回False → 本地拒绝
#
# requests_per_second：令牌补充速度
# max_bucket_size：最多可以积累多少令牌
# check_every_n_seconds：阻塞等待时多久检查一次
