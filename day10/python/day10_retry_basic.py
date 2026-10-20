"""Day10 第一步：使用 Runnable.with_retry() 实现可控重试。"""

from typing import TypedDict

from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. 定义两类错误
# ============================================================

class TemporaryServiceError(Exception):
    """临时错误：稍后重试可能成功，例如网络抖动、服务繁忙。"""


class InvalidRequestError(Exception):
    """永久错误：请求内容有问题，重复执行也不会成功。"""


# ============================================================
# 2. 定义输入和输出的数据结构
# ============================================================

class ServiceRequest(TypedDict):
    text: str
    failures_before_success: int


class ServiceResult(TypedDict):
    original_text: str
    processed_text: str
    successful_attempt: int


# ============================================================
# 3. 模拟一个不稳定的外部服务
# ============================================================

class UnstableTextService:
    """前几次调用临时失败，达到指定次数后才成功。"""

    def __init__(self) -> None:
        # 每个服务对象独立记录自己已经被调用了多少次。
        self.attempt_count = 0

    def process(self, request: ServiceRequest) -> ServiceResult:
        """处理文本；该方法会被包装成 LangChain Runnable。"""

        self.attempt_count += 1
        text = request["text"].strip()
        failures_before_success = request["failures_before_success"]

        print(f"[第 {self.attempt_count} 次调用] text={text!r}")

        # 空文本属于请求本身有问题，重试不会改变结果。
        if not text:
            print("结果：请求无效，不应该重试。")
            raise InvalidRequestError("text 不能为空。")

        # 模拟网络抖动或服务临时繁忙。
        if self.attempt_count <= failures_before_success:
            print("结果：发生临时错误，可以重试。")
            raise TemporaryServiceError("模拟服务暂时不可用。")

        result: ServiceResult = {
            "original_text": text,
            "processed_text": text.upper(),
            "successful_attempt": self.attempt_count
        }

        print("结果：调用成功。")
        return result


# ============================================================
# 4. 创建一个带重试能力的 Runnable
# ============================================================

def create_reliable_service(
    service: UnstableTextService
):
    """把普通 Python 方法包装成最多执行三次的 Runnable。"""

    basic_runnable = RunnableLambda(service.process)

    reliable_runnable = basic_runnable.with_retry(
        # 只有临时错误才重试。
        retry_if_exception_type=(TemporaryServiceError,),

        # 今天先关闭等待时间，让练习立即完成。
        wait_exponential_jitter=False,

        # 包含第一次调用在内，最多执行三次。
        stop_after_attempt=3
    )

    return basic_runnable, reliable_runnable


# ============================================================
# 5. 运行一个测试场景
# ============================================================

def run_case(
    title: str,
    request: ServiceRequest
) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("输入：", request)
    print("-" * 60)

    service = UnstableTextService()
    basic_runnable, reliable_runnable = create_reliable_service(service)

    print("基础 Runnable 类型：")
    print(type(basic_runnable))

    print("\n带重试 Runnable 类型：")
    print(type(reliable_runnable))

    try:
        result = reliable_runnable.invoke(request)
    except TemporaryServiceError as error:
        print("\n最终状态：三次机会全部用完，仍然失败。")
        print("捕获的错误类型：", type(error).__name__)
        print("错误内容：", error)
    except InvalidRequestError as error:
        print("\n最终状态：请求本身无效，没有重试。")
        print("捕获的错误类型：", type(error).__name__)
        print("错误内容：", error)
    else:
        print("\n最终状态：执行成功。")
        print("返回结果：", result)
        print("返回类型：", type(result))
    finally:
        print("服务实际调用次数：", service.attempt_count)


# ============================================================
# 6. 三组对比实验
# ============================================================

# 前两次临时失败，第三次成功。
run_case(
    "实验一：临时失败后重试成功",
    {
        "text": "langchain retry",
        "failures_before_success": 2
    }
)

# 需要第四次才能成功，但最大执行次数是三次，因此最终失败。
run_case(
    "实验二：超过最大重试次数",
    {
        "text": "service unavailable",
        "failures_before_success": 3
    }
)

# 空文本触发不可重试错误，只执行一次。
run_case(
    "实验三：不可重试的无效请求",
    {
        "text": "   ",
        "failures_before_success": 0
    }
)


# ============================================================
# 数据流
# ============================================================
#
# ServiceRequest
#      ↓
# reliable_runnable.invoke(request)
#      ↓
# UnstableTextService.process()
#      ├── 成功 → ServiceResult
#      ├── TemporaryServiceError → with_retry再次调用
#      └── InvalidRequestError → 不重试，直接抛给调用者
#
# 注意：stop_after_attempt=3表示总共最多调用三次，
# 不是“第一次失败以后再额外重试三次”。
