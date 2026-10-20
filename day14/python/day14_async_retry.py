"""Day14 第六步：异步重试、错误分类和指数退避。"""

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()


class TemporaryModelError(Exception):
    """模拟超时、限流等临时错误，可以重试。"""


class InvalidRequestError(Exception):
    """模拟参数错误，重复请求也不会成功。"""


@dataclass
class RetryResult:
    name: str
    status: str
    attempts: int
    value: str | None = None
    error: str | None = None


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
    max_tokens=60,
    timeout=30,
    max_retries=0  # 关闭SDK内部重试，只观察我们自己写的重试层。
)


async def retry_async(
    name: str,
    operation: Callable[[], Awaitable[str]],
    max_attempts: int = 4,
    base_delay: float = 0.5
) -> RetryResult:
    """只重试TemporaryModelError，并使用指数退避。"""

    for attempt in range(1, max_attempts + 1):
        print(f"[{name}] 第{attempt}/{max_attempts}次尝试")

        try:
            value = await operation()
            print(f"[{name}] 执行成功")
            return RetryResult(
                name=name,
                status="success",
                attempts=attempt,
                value=value
            )

        except TemporaryModelError as error:
            if attempt == max_attempts:
                print(f"[{name}] 已达到最大尝试次数")
                return RetryResult(
                    name=name,
                    status="failed",
                    attempts=attempt,
                    error=str(error)
                )

            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"[{name}] 临时错误：{error}；"
                f"等待{delay:.1f}秒后重试"
            )
            await asyncio.sleep(delay)

        except Exception as error:
            print(
                f"[{name}] 不可重试错误："
                f"{type(error).__name__}: {error}"
            )
            return RetryResult(
                name=name,
                status="failed",
                attempts=attempt,
                error=f"{type(error).__name__}: {error}"
            )

    raise RuntimeError("重试循环不应该运行到这里。")


def make_flaky_model_operation() -> Callable[[], Awaitable[str]]:
    """创建一个前两次失败、第三次调用真实模型的异步操作。"""

    attempt_counter = 0

    async def operation() -> str:
        nonlocal attempt_counter
        attempt_counter += 1

        if attempt_counter <= 2:
            raise TemporaryModelError(
                f"模拟上游服务繁忙（第{attempt_counter}次）"
            )

        response = await llm.ainvoke([
            HumanMessage(
                content="请用一句话解释指数退避，回答保持简短。"
            )
        ])
        return str(response.content)

    return operation


async def invalid_operation() -> str:
    """模拟一个无论重试多少次都不会成功的参数错误。"""

    raise InvalidRequestError("question字段为空。")


async def main() -> None:
    print("第一部分：临时错误会重试\n")

    retryable_result = await retry_async(
        name="临时故障任务",
        operation=make_flaky_model_operation(),
        max_attempts=4,
        base_delay=0.5
    )

    print("\n" + "=" * 70)
    print("第二部分：不可重试错误立即结束\n")

    invalid_result = await retry_async(
        name="错误参数任务",
        operation=invalid_operation,
        max_attempts=4,
        base_delay=0.5
    )

    print("\n" + "=" * 70)
    print("最终结果：")

    for result in [retryable_result, invalid_result]:
        print(f"\n任务：{result.name}")
        print("状态：", result.status)
        print("实际尝试次数：", result.attempts)
        print("返回值：", result.value)
        print("错误：", result.error)


if __name__ == "__main__":
    asyncio.run(main())


# 临时错误：尝试1失败 → 等0.5秒 → 尝试2失败 → 等1秒 → 尝试3
# 指数公式：delay = base_delay × 2^(attempt - 1)
# 参数错误：第一次失败 → 不重试 → 立即返回failed
