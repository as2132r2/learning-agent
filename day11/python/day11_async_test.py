"""Day11 第八步：使用异步Mock测试ainvoke重试和降级。"""

import unittest
from typing import Callable, Awaitable
from unittest.mock import AsyncMock


class TemporaryModelError(Exception):
    pass


class InvalidQuestionError(Exception):
    pass


AsyncModel = Callable[[str], Awaitable[str]]


class AsyncReliableGateway:
    def __init__(
        self,
        primary_model: AsyncModel,
        fallback_model: AsyncModel
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    async def ask(self, question: str) -> dict:
        question = question.strip()

        if not question:
            raise InvalidQuestionError("问题不能为空")

        for attempt in range(1, 3):
            try:
                answer = await self.primary_model(question)
            except TemporaryModelError:
                continue
            else:
                return {
                    "service": "primary",
                    "answer": answer,
                    "attempts": attempt
                }

        answer = await self.fallback_model(question)
        return {
            "service": "fallback",
            "answer": answer,
            "attempts": 2
        }


class TestAsyncReliableGateway(
    unittest.IsolatedAsyncioTestCase
):
    async def test_async_primary_success(self) -> None:
        primary = AsyncMock(return_value="异步主模型回答")
        fallback = AsyncMock(return_value="异步备用回答")
        gateway = AsyncReliableGateway(primary, fallback)

        result = await gateway.ask("什么是异步调用？")

        self.assertEqual(result["service"], "primary")
        self.assertEqual(result["answer"], "异步主模型回答")
        primary.assert_awaited_once_with("什么是异步调用？")
        fallback.assert_not_awaited()

    async def test_async_retry_then_fallback(self) -> None:
        primary = AsyncMock(side_effect=[
            TemporaryModelError("故障1"),
            TemporaryModelError("故障2")
        ])
        fallback = AsyncMock(return_value="异步备用回答")
        gateway = AsyncReliableGateway(primary, fallback)

        result = await gateway.ask("请解释异步降级。")

        self.assertEqual(result["service"], "fallback")
        self.assertEqual(primary.await_count, 2)
        fallback.assert_awaited_once_with("请解释异步降级。")

    async def test_async_invalid_question_awaits_no_model(self) -> None:
        primary = AsyncMock()
        fallback = AsyncMock()
        gateway = AsyncReliableGateway(primary, fallback)

        with self.assertRaises(InvalidQuestionError):
            await gateway.ask("   ")

        primary.assert_not_awaited()
        fallback.assert_not_awaited()


if __name__ == "__main__":
    print("开始运行 Day11 异步测试……")
    unittest.main(verbosity=2)


# ============================================================
# 数据流
# ============================================================
#
# await gateway.ask(question)
#       ↓
# await primary_model(question)
#       ├── 成功 → 返回主模型结果
#       └── TemporaryModelError → 再await一次
#                                  ↓ 仍失败
#                              await fallback_model(question)
#
# AsyncMock的assert_awaited_*断言专门验证：
# 协程是否真的被await，而不仅仅是被创建。
