"""Day11 第七步：用故障注入验证可靠性策略。"""

import unittest
from dataclasses import dataclass
from typing import Literal


class TemporaryModelError(Exception):
    pass


class InvalidQuestionError(Exception):
    pass


@dataclass
class FakePrimaryModel:
    """前failures次调用失败，之后成功。"""

    failures: int
    calls: int = 0

    def invoke(self, question: str) -> str:
        self.calls += 1

        if self.calls <= self.failures:
            raise TemporaryModelError("注入的主服务故障")

        return f"主服务回答：{question}"


@dataclass
class FakeFallbackModel:
    calls: int = 0

    def invoke(self, question: str) -> str:
        self.calls += 1
        return f"备用服务回答：{question}"


class FailureInjectionGateway:
    """一个可控的主服务重试和备用服务降级网关。"""

    def __init__(
        self,
        primary: FakePrimaryModel,
        fallback: FakeFallbackModel,
        max_attempts: int = 2
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_attempts = max_attempts

    def ask(
        self,
        question: str
    ) -> tuple[Literal["primary", "fallback"], str]:
        if not question.strip():
            raise InvalidQuestionError("问题不能为空")

        for _ in range(self.max_attempts):
            try:
                return "primary", self.primary.invoke(question)
            except TemporaryModelError:
                continue

        return "fallback", self.fallback.invoke(question)


class TestFailureMatrix(unittest.TestCase):
    """用表格一次验证多种故障场景。"""

    def test_failure_injection_matrix(self) -> None:
        cases = [
            {
                "name": "主服务第一次失败后恢复",
                "failures": 1,
                "expected_service": "primary",
                "expected_primary_calls": 2,
                "expected_fallback_calls": 0
            },
            {
                "name": "主服务连续两次失败后降级",
                "failures": 2,
                "expected_service": "fallback",
                "expected_primary_calls": 2,
                "expected_fallback_calls": 1
            },
            {
                "name": "主服务长期故障仍只尝试两次",
                "failures": 99,
                "expected_service": "fallback",
                "expected_primary_calls": 2,
                "expected_fallback_calls": 1
            },
            {
                "name": "主服务零故障",
                "failures": 0,
                "expected_service": "primary",
                "expected_primary_calls": 1,
                "expected_fallback_calls": 0
            }
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                primary = FakePrimaryModel(case["failures"])
                fallback = FakeFallbackModel()
                gateway = FailureInjectionGateway(primary, fallback)

                service, _ = gateway.ask("测试请求")

                self.assertEqual(
                    service,
                    case["expected_service"]
                )
                self.assertEqual(
                    primary.calls,
                    case["expected_primary_calls"]
                )
                self.assertEqual(
                    fallback.calls,
                    case["expected_fallback_calls"]
                )

    def test_invalid_input_does_not_invoke_any_service(self) -> None:
        primary = FakePrimaryModel(failures=99)
        fallback = FakeFallbackModel()
        gateway = FailureInjectionGateway(primary, fallback)

        with self.assertRaises(InvalidQuestionError):
            gateway.ask("   ")

        self.assertEqual(primary.calls, 0)
        self.assertEqual(fallback.calls, 0)


if __name__ == "__main__":
    print("开始运行 Day11 故障注入测试……")
    unittest.main(verbosity=2)


# ============================================================
# 数据流
# ============================================================
#
# 测试场景表
#       ↓ subTest逐项执行
# FakePrimaryModel
#       ├── 正常 → primary
#       ├── 短暂失败后恢复 → 重试后primary
#       └── 持续失败 → 重试用尽后fallback
#       ↓
# 检查服务选择、主服务调用次数、备用服务调用次数
