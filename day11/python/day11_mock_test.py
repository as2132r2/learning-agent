"""Day11 第二步：使用Mock测试主模型、重试和备用模型。"""

import unittest
from typing import Callable
from unittest.mock import Mock


class TemporaryModelError(Exception):
    """模拟主模型临时失败。"""


class InvalidQuestionError(Exception):
    """问题本身无效。"""


class ReliableGateway:
    """把模型依赖注入进来，方便使用Mock进行测试。"""

    def __init__(
        self,
        primary_model: Callable[[str], str],
        fallback_model: Callable[[str], str]
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def ask(self, question: str) -> dict:
        question = question.strip()

        if not question:
            raise InvalidQuestionError("问题不能为空。")

        last_error: Exception | None = None

        # 主模型最多尝试两次。
        for attempt in range(1, 3):
            try:
                answer = self.primary_model(question)
            except TemporaryModelError as error:
                last_error = error
                continue
            else:
                return {
                    "service": "primary",
                    "answer": answer,
                    "primary_attempts": attempt,
                    "fallback_used": False
                }

        # 主模型两次都失败后才使用备用模型。
        fallback_answer = self.fallback_model(question)

        return {
            "service": "fallback",
            "answer": fallback_answer,
            "primary_attempts": 2,
            "fallback_used": True,
            "fallback_reason": str(last_error)
        }


class TestReliableGateway(unittest.TestCase):
    """不调用真实API，使用Mock验证网关行为。"""

    def test_primary_success_does_not_call_fallback(self) -> None:
        primary = Mock(return_value="主模型回答")
        fallback = Mock(return_value="备用回答")
        gateway = ReliableGateway(primary, fallback)

        result = gateway.ask("什么是 RAG？")

        self.assertEqual(result["service"], "primary")
        self.assertEqual(result["answer"], "主模型回答")
        self.assertFalse(result["fallback_used"])
        primary.assert_called_once_with("什么是 RAG？")
        fallback.assert_not_called()

    def test_primary_failure_retries_then_uses_fallback(self) -> None:
        primary = Mock(side_effect=[
            TemporaryModelError("第一次失败"),
            TemporaryModelError("第二次失败")
        ])
        fallback = Mock(return_value="备用模型回答")
        gateway = ReliableGateway(primary, fallback)

        result = gateway.ask("解释 LangGraph。")

        self.assertEqual(result["service"], "fallback")
        self.assertEqual(result["answer"], "备用模型回答")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["primary_attempts"], 2)
        self.assertEqual(primary.call_count, 2)
        fallback.assert_called_once_with("解释 LangGraph。")

    def test_first_failure_second_success_stops_retry(self) -> None:
        primary = Mock(side_effect=[
            TemporaryModelError("短暂故障"),
            "第二次成功"
        ])
        fallback = Mock(return_value="不应该使用")
        gateway = ReliableGateway(primary, fallback)

        result = gateway.ask("请计算 2+3。")

        self.assertEqual(result["answer"], "第二次成功")
        self.assertEqual(result["primary_attempts"], 2)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(primary.call_count, 2)
        fallback.assert_not_called()

    def test_empty_question_calls_no_model(self) -> None:
        primary = Mock()
        fallback = Mock()
        gateway = ReliableGateway(primary, fallback)

        with self.assertRaises(InvalidQuestionError):
            gateway.ask("   ")

        primary.assert_not_called()
        fallback.assert_not_called()


if __name__ == "__main__":
    print("开始运行 Day11 Mock 测试……")
    unittest.main(verbosity=2)


# ============================================================
# 数据流
# ============================================================
#
# Mock对象
#     ↓ 预设 return_value 或 side_effect
# ReliableGateway.ask()
#     ↓
# 模拟主模型成功 / 失败
#     ├── 主模型成功 → 不调用备用模型
#     ├── 第一次失败、第二次成功 → 停止重试
#     └── 两次失败 → 调用备用模型
#
# assert_called_once_with() / assert_not_called()
#     ↓
# 验证调用次数和参数是否符合设计
