"""Day11 第一步：使用Python unittest测试可靠性逻辑。"""

import hashlib
import unittest
from dataclasses import dataclass


# ============================================================
# 1. 被测试的业务逻辑
# ============================================================

RETRYABLE_ERROR_NAMES = {
    "TemporaryModelError",
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError"
}


def is_retryable_error(error_name: str) -> bool:
    """判断错误名称是否属于可重试错误。"""

    return error_name in RETRYABLE_ERROR_NAMES


def make_cache_key(prompt: str, model_config: str) -> str:
    """根据问题和模型配置生成稳定的缓存键。"""

    raw_key = f"{prompt}\n---MODEL---\n{model_config}"
    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def should_open_circuit(
    failure_count: int,
    failure_threshold: int
) -> bool:
    """失败次数达到阈值时打开熔断器。"""

    if failure_count < 0:
        raise ValueError("failure_count 不能是负数。")

    if failure_threshold <= 0:
        raise ValueError("failure_threshold 必须大于0。")

    return failure_count >= failure_threshold


@dataclass
class FailureCounter:
    """演示连续失败计数，以及成功后的正确复位。"""

    count: int = 0

    def record_failure(self) -> int:
        self.count += 1
        return self.count

    def record_success(self) -> None:
        # 成功意味着失败不是连续发生，必须清零。
        self.count = 0


# ============================================================
# 2. unittest 测试类
# ============================================================

class TestRetryClassification(unittest.TestCase):
    """测试哪些错误应该重试。"""

    def test_temporary_error_is_retryable(self) -> None:
        self.assertTrue(
            is_retryable_error("TemporaryModelError")
        )

    def test_timeout_is_retryable(self) -> None:
        self.assertTrue(
            is_retryable_error("APITimeoutError")
        )

    def test_invalid_question_is_not_retryable(self) -> None:
        self.assertFalse(
            is_retryable_error("InvalidQuestionError")
        )


class TestCacheKey(unittest.TestCase):
    """测试缓存键的稳定性和区分能力。"""

    def test_same_input_has_same_key(self) -> None:
        first_key = make_cache_key("什么是RAG？", "model-a")
        second_key = make_cache_key("什么是RAG？", "model-a")
        self.assertEqual(first_key, second_key)

    def test_different_prompt_has_different_key(self) -> None:
        first_key = make_cache_key("什么是RAG？", "model-a")
        second_key = make_cache_key("什么是LangGraph？", "model-a")
        self.assertNotEqual(first_key, second_key)

    def test_different_model_has_different_key(self) -> None:
        first_key = make_cache_key("什么是RAG？", "model-a")
        second_key = make_cache_key("什么是RAG？", "model-b")
        self.assertNotEqual(first_key, second_key)


class TestCircuitLogic(unittest.TestCase):
    """测试熔断阈值和连续失败计数。"""

    def test_threshold_not_reached(self) -> None:
        self.assertFalse(should_open_circuit(1, 2))

    def test_threshold_reached(self) -> None:
        self.assertTrue(should_open_circuit(2, 2))

    def test_success_resets_consecutive_failures(self) -> None:
        counter = FailureCounter()
        counter.record_failure()
        counter.record_success()
        counter.record_failure()

        # 成功后重新从1次失败开始，不能错误地算成第2次连续失败。
        self.assertEqual(counter.count, 1)

    def test_negative_failure_count_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            should_open_circuit(-1, 2)


# ============================================================
# 3. 运行测试
# ============================================================

if __name__ == "__main__":
    print("开始运行 Day11 单元测试……")
    unittest.main(verbosity=2)


# ============================================================
# 数据流
# ============================================================
#
# 测试输入
#     ↓
# 被测试函数
#     ↓
# 实际结果
#     ↓
# unittest 断言
#     ├── 相等 → 测试通过
#     └── 不相等 → 测试失败并显示原因
#
# 本文件不调用模型、不访问网络、不消耗API Token。
