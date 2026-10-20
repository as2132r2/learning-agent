"""Day10 第二步：给真实大模型调用添加超时和可控重试。"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError


# ============================================================
# 1. 定义可重试错误和不可重试错误
# ============================================================

class TemporaryModelError(Exception):
    """本地模拟的模型临时故障，用于安全演示重试。"""


class InvalidQuestionError(Exception):
    """用户问题本身无效，重复调用模型也无法修复。"""


# ============================================================
# 2. 定义 Runnable 的输入和输出
# ============================================================

class ModelRequest(TypedDict):
    question: str
    simulate_first_failure: bool


class ModelResult(TypedDict):
    answer: str
    model_name: str
    usage: dict
    successful_attempt: int


# ============================================================
# 3. 创建真实模型，并关闭模型客户端的内置重试
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,

    # 单次请求最多等待20秒。
    timeout=20,

    # 关闭ChatOpenAI内部重试，让本例只有一层重试逻辑。
    max_retries=0
)


# ============================================================
# 4. 封装一次模型调用
# ============================================================

class ReliableModelGateway:
    """验证输入、模拟故障并调用真实模型。"""

    def __init__(self) -> None:
        self.attempt_count = 0

    def invoke_model(self, request: ModelRequest) -> ModelResult:
        self.attempt_count += 1
        question = request["question"].strip()

        print(f"\n[第 {self.attempt_count} 次尝试]")

        # 输入验证必须放在真正调用模型之前。
        if not question:
            print("问题为空：这是不可重试错误。")
            raise InvalidQuestionError("question 不能为空。")

        # 第一次在请求API之前模拟网络抖动，不会产生模型费用。
        if (
            request["simulate_first_failure"]
            and self.attempt_count == 1
        ):
            print("模拟连接失败：本次没有真正请求模型。")
            raise TemporaryModelError("模拟第一次连接失败。")

        print("开始请求真实模型……")
        response = llm.invoke(question)
        print("模型请求成功。")

        model_name = str(
            response.response_metadata.get("model_name", "未知模型")
        )

        result: ModelResult = {
            "answer": str(response.content),
            "model_name": model_name,
            "usage": dict(response.usage_metadata or {}),
            "successful_attempt": self.attempt_count
        }

        return result


# ============================================================
# 5. 创建具有重试策略的 Runnable
# ============================================================

def create_reliable_model_runnable(
    gateway: ReliableModelGateway
):
    basic_runnable = RunnableLambda(gateway.invoke_model)

    reliable_runnable = basic_runnable.with_retry(
        retry_if_exception_type=(
            # 本地教学用临时错误。
            TemporaryModelError,

            # OpenAI兼容客户端可能产生的真实临时错误。
            APIConnectionError,
            APITimeoutError,
            RateLimitError
        ),

        # 为了快速观察结果，本练习不等待。
        # 生产环境通常应该启用指数退避。
        wait_exponential_jitter=False,

        # 第一次调用加两次重试，总共最多三次尝试。
        stop_after_attempt=3
    )

    return basic_runnable, reliable_runnable


# ============================================================
# 6. 实验一：模拟一次连接失败，然后调用真实模型
# ============================================================

print("=" * 60)
print("实验一：临时连接失败后自动重试")

gateway = ReliableModelGateway()
basic_runnable, reliable_runnable = (
    create_reliable_model_runnable(gateway)
)

print("基础对象类型：", type(basic_runnable))
print("重试对象类型：", type(reliable_runnable))

try:
    result = reliable_runnable.invoke({
        "question": "请用一句话解释什么是 LangGraph。",
        "simulate_first_failure": True
    })
except (
    TemporaryModelError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError
) as error:
    print("最终状态：达到最大尝试次数后仍然失败。")
    print("错误类型：", type(error).__name__)
    print("错误内容：", error)
else:
    print("\n最终状态：成功")
    print("模型回答：", result["answer"])
    print("模型名称：", result["model_name"])
    print("Token信息：", result["usage"])
    print("成功发生在第几次尝试：", result["successful_attempt"])
finally:
    print("总尝试次数：", gateway.attempt_count)


# ============================================================
# 7. 实验二：空问题不会触发重试，也不会请求模型
# ============================================================

print("\n" + "=" * 60)
print("实验二：无效输入立即失败")

invalid_gateway = ReliableModelGateway()
_, invalid_runnable = create_reliable_model_runnable(invalid_gateway)

try:
    invalid_runnable.invoke({
        "question": "   ",
        "simulate_first_failure": False
    })
except InvalidQuestionError as error:
    print("最终状态：输入无效，没有重试。")
    print("错误类型：", type(error).__name__)
    print("错误内容：", error)
finally:
    print("总尝试次数：", invalid_gateway.attempt_count)


# ============================================================
# 数据流
# ============================================================
#
# ModelRequest
#     ↓
# RunnableRetry.invoke()
#     ↓
# ReliableModelGateway.invoke_model()
#     ├── 空问题 → InvalidQuestionError → 不重试
#     ├── 临时连接错误 → 自动重试
#     └── 输入有效 → llm.invoke()
#                         ↓
#                    AIMessage
#                         ↓
#                    ModelResult
#
# 两层可靠性配置：
# ChatOpenAI(timeout=20, max_retries=0)
#     └── 控制单次HTTP请求
# Runnable.with_retry(stop_after_attempt=3)
#     └── 控制整个模型调用流程最多执行几次
