"""Day10 第三步：主模型重试失败后切换到备用模型。"""

import os
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError


# ============================================================
# 1. 错误类型
# ============================================================

class PrimaryServiceUnavailableError(Exception):
    """主模型服务临时不可用，可以重试或切换备用服务。"""


class InvalidQuestionError(Exception):
    """问题内容无效，既不应该重试，也不应该切换备用模型。"""


RETRYABLE_ERRORS = (
    PrimaryServiceUnavailableError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError
)


# ============================================================
# 2. 输入和统一输出结构
# ============================================================

class ModelRequest(TypedDict):
    question: str
    force_primary_failure: bool

    # with_fallbacks会在切换时把主服务异常放入这个字段。
    last_error: NotRequired[BaseException]


class ModelResult(TypedDict):
    answer: str
    service: str
    model_name: str
    primary_attempts: int
    fallback_attempts: int
    fallback_reason: str


# ============================================================
# 3. 创建主模型和备用模型
# ============================================================

load_dotenv()

primary_llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=20,
    max_retries=0
)

# 如果以后配置了DEEPSEEK_FALLBACK_MODEL，就使用不同的备用模型。
# 当前没有配置时，仍使用同一接口演示“备用调用路径”。
fallback_llm = ChatOpenAI(
    model=(
        os.getenv("DEEPSEEK_FALLBACK_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
    ),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    timeout=20,
    max_retries=0
)


# ============================================================
# 4. 主服务和备用服务
# ============================================================

class ModelServices:
    def __init__(self) -> None:
        self.primary_attempts = 0
        self.fallback_attempts = 0

    @staticmethod
    def validate_question(request: ModelRequest) -> str:
        question = request["question"].strip()

        if not question:
            raise InvalidQuestionError("question 不能为空。")

        return question

    def call_primary(self, request: ModelRequest) -> ModelResult:
        """调用主模型；练习中可以在API请求前强制模拟失败。"""

        question = self.validate_question(request)
        self.primary_attempts += 1

        print(f"[主服务第 {self.primary_attempts} 次尝试]")

        if request["force_primary_failure"]:
            print("主服务模拟失败，本次没有产生模型请求。")
            raise PrimaryServiceUnavailableError(
                "主模型服务暂时不可用。"
            )

        response = primary_llm.invoke(question)
        print("主模型调用成功。")

        return {
            "answer": str(response.content),
            "service": "primary",
            "model_name": str(
                response.response_metadata.get(
                    "model_name",
                    "未知模型"
                )
            ),
            "primary_attempts": self.primary_attempts,
            "fallback_attempts": self.fallback_attempts,
            "fallback_reason": ""
        }

    def call_fallback(self, request: ModelRequest) -> ModelResult:
        """主服务彻底失败后，使用备用模型完成请求。"""

        question = self.validate_question(request)
        self.fallback_attempts += 1
        last_error = request.get("last_error")

        print(f"[备用服务第 {self.fallback_attempts} 次尝试]")
        print("切换原因：", last_error)

        response = fallback_llm.invoke(question)
        print("备用模型调用成功。")

        return {
            "answer": str(response.content),
            "service": "fallback",
            "model_name": str(
                response.response_metadata.get(
                    "model_name",
                    "未知模型"
                )
            ),
            "primary_attempts": self.primary_attempts,
            "fallback_attempts": self.fallback_attempts,
            "fallback_reason": str(last_error or "未知原因")
        }


# ============================================================
# 5. 组合“主服务重试 + 备用服务”
# ============================================================

def create_resilient_model(services: ModelServices):
    primary_runnable = RunnableLambda(services.call_primary)
    fallback_runnable = RunnableLambda(services.call_fallback)

    # 主服务总共最多尝试两次。
    retrying_primary = primary_runnable.with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        wait_exponential_jitter=False,
        stop_after_attempt=2
    )

    # 主服务重试完仍失败，才进入备用服务。
    resilient_runnable = retrying_primary.with_fallbacks(
        [fallback_runnable],
        exceptions_to_handle=RETRYABLE_ERRORS,

        # 把主服务最后一次异常加入备用服务的输入字典。
        exception_key="last_error"
    )

    return retrying_primary, fallback_runnable, resilient_runnable


# ============================================================
# 6. 实验一：主服务失败两次，切换备用模型
# ============================================================

print("=" * 60)
print("实验一：主服务重试失败后自动降级")

services = ModelServices()
retrying_primary, fallback_runnable, resilient_runnable = (
    create_resilient_model(services)
)

print("主服务重试对象：", type(retrying_primary))
print("备用服务对象：", type(fallback_runnable))
print("完整容错对象：", type(resilient_runnable))

try:
    result = resilient_runnable.invoke({
        "question": "请用一句话说明为什么生产系统需要降级方案。",
        "force_primary_failure": True
    })
except Exception as error:
    print("主服务和备用服务都失败。")
    print("错误类型：", type(error).__name__)
    print("错误内容：", error)
else:
    print("\n最终执行成功。")
    print("最终使用的服务：", result["service"])
    print("模型名称：", result["model_name"])
    print("主服务尝试次数：", result["primary_attempts"])
    print("备用服务尝试次数：", result["fallback_attempts"])
    print("降级原因：", result["fallback_reason"])
    print("最终回答：", result["answer"])


# ============================================================
# 7. 实验二：无效输入不会切换备用服务
# ============================================================

print("\n" + "=" * 60)
print("实验二：无效输入不重试、不降级")

invalid_services = ModelServices()
_, _, invalid_resilient_runnable = create_resilient_model(
    invalid_services
)

try:
    invalid_resilient_runnable.invoke({
        "question": "   ",
        "force_primary_failure": False
    })
except InvalidQuestionError as error:
    print("输入无效，流程立即停止。")
    print("错误内容：", error)
    print("主服务调用次数：", invalid_services.primary_attempts)
    print("备用服务调用次数：", invalid_services.fallback_attempts)


# ============================================================
# 数据流
# ============================================================
#
# ModelRequest
#      ↓
# retrying_primary
#      ├── 主服务成功 → ModelResult(service="primary")
#      └── 可重试错误
#             ↓ 最多再尝试一次
#         仍然失败
#             ↓ with_fallbacks
#         exception_key="last_error"
#             ↓
#         fallback_runnable
#             ↓
#         ModelResult(service="fallback")
#
# InvalidQuestionError不在RETRYABLE_ERRORS中：
# 不重试，也不会进入fallback。
