"""Day12 第三步：使用Pydantic校验进入Agent的用户请求。"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AgentRequest(BaseModel):
    """进入模型或工具之前必须通过的请求契约。"""

    model_config = ConfigDict(
        # 未声明的字段直接拒绝，避免调用方偷偷传入危险参数。
        extra="forbid",
        # 去除字符串首尾空格。
        str_strip_whitespace=True
    )

    request_id: str = Field(
        min_length=4,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$"
    )
    question: str = Field(
        min_length=1,
        max_length=500,
        description="用户问题，限制长度避免过大的输入。"
    )
    thread_id: str = Field(
        min_length=4,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$"
    )

    @field_validator("question")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        """拒绝不可见控制字符，减少日志和下游解析问题。"""

        if any(ord(character) < 32 for character in value):
            raise ValueError("question 不能包含控制字符。")

        return value


def parse_request(raw_data: dict[str, Any]) -> AgentRequest:
    """统一入口：校验失败就抛出ValidationError。"""

    return AgentRequest.model_validate(raw_data)


def show_case(title: str, raw_data: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("原始输入：", raw_data)

    try:
        request = parse_request(raw_data)
    except ValidationError as error:
        print("校验失败，未进入模型或工具：")

        for detail in error.errors():
            print(
                "- 字段：",
                ".".join(str(item) for item in detail["loc"]),
                "| 原因：",
                detail["msg"]
            )
    else:
        print("校验通过：")
        print(request.model_dump())
        print("规范化后的JSON：")
        print(request.model_dump_json())


# ============================================================
# 1. 合法请求
# ============================================================

show_case(
    "实验一：合法请求",
    {
        "request_id": "req-001",
        "question": "  什么是 RAG？  ",
        "thread_id": "xiaolin-day12"
    }
)


# ============================================================
# 2. 多种非法请求
# ============================================================

show_case(
    "实验二：问题为空",
    {
        "request_id": "req-002",
        "question": "",
        "thread_id": "xiaolin-day12"
    }
)

show_case(
    "实验三：问题超过500个字符",
    {
        "request_id": "req-003",
        "question": "问题" * 300,
        "thread_id": "xiaolin-day12"
    }
)

show_case(
    "实验四：thread_id包含危险字符",
    {
        "request_id": "req-004",
        "question": "请查询资料",
        "thread_id": "xiaolin/../../secret"
    }
)

show_case(
    "实验五：传入未声明字段",
    {
        "request_id": "req-005",
        "question": "请查询资料",
        "thread_id": "xiaolin-day12",
        "is_admin": True
    }
)


# ============================================================
# 数据流
# ============================================================
#
# 外部dict / JSON
#       ↓
# AgentRequest.model_validate()
#       ├── 失败 → ValidationError → 立即返回错误
#       └── 成功 → AgentRequest
#                         ↓
#                    Prompt / Tool
#
# 输入校验不是为了替代模型安全策略，而是把明显非法输入
# 在最靠近入口的位置拦截掉。
