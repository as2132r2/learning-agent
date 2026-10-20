"""Day33 第1步：毕业项目需求、状态与验收约定。

运行：python day33/python/day33_project_contract.py
本步定义输入输出和状态，不调用模型，不执行工具或写入学习笔记。
"""

from __future__ import annotations

import json
from enum import Enum
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Route(str, Enum):
    """限定毕业项目支持的四条路线。"""

    CHAT = "chat"
    CALCULATE = "calculate"
    KNOWLEDGE = "knowledge"
    SAVE_NOTE = "save_note"


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not all(character.isascii() and (character.isalnum() or character in "-_") for character in value):
            raise ValueError("会话编号只允许英文字母、数字、短横线和下划线。")
        return value


class Evidence(BaseModel):
    """检索资料的结构；引用以chunk_id定位。"""

    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content: str = Field(min_length=1)


class AgentState(TypedDict):
    """共享状态的类型说明，后一步再加入LangGraph消息合并规则。

    TypedDict用于静态描述，并不负责运行时验证。
    """

    session_id: str
    question: str
    messages: list[BaseMessage]
    route: Route | None
    evidence: list[Evidence]
    answer: str
    pending_note: str | None
    approval: bool | None


class AcceptanceCase(BaseModel):
    case_id: str
    question: str
    capability: str
    expected_behavior: str


ACCEPTANCE_CASES = [
    AcceptanceCase(
        case_id="A01", question="我叫小林。", capability="chat",
        expected_behavior="正常交流，把用户消息和助手回复保存到当前会话。",
    ),
    AcceptanceCase(
        case_id="A02", question="我叫什么名字？", capability="memory",
        expected_behavior="同会话读到小林；其他会话不能读到该信息。",
    ),
    AcceptanceCase(
        case_id="A03", question="请计算(20+4)*3。", capability="calculate",
        expected_behavior="真实执行add和multiply工具，答案为72。",
    ),
    AcceptanceCase(
        case_id="A04", question="根据课程资料解释RAG。", capability="knowledge",
        expected_behavior="检索本项目资料，回答附来源；资料不足时明确说明。",
    ),
    AcceptanceCase(
        case_id="A05", question="把RAG的定义保存为学习笔记。", capability="approval",
        expected_behavior="先展示待保存内容并暂停；批准才写入，拒绝不写入。",
    ),
    AcceptanceCase(
        case_id="A06", question="重新启动后继续刚才的对话。", capability="persistence",
        expected_behavior="相同session_id恢复已保存消息，新会话独立。",
    ),
]


def initial_state(request: UserRequest) -> AgentState:
    """初始化一轮数据；本步尚不读取数据库历史。"""
    return {
        "session_id": request.session_id,
        "question": request.question,
        "messages": [HumanMessage(content=request.question)],
        "route": None,
        "evidence": [],
        "answer": "",
        "pending_note": None,
        "approval": None,
    }


def main() -> None:
    print("Day33 毕业项目：编程学习助手")
    request = UserRequest(
        session_id="xiaolin-capstone-001",
        question="请计算(20+4)*3。",
    )
    state = initial_state(request)
    print("\n1. 已验证的输入：")
    print(request.model_dump_json(indent=2))
    print("\n2. 当前允许的路线：")
    print([route.value for route in Route])
    print("\n3. 初始共享状态：")
    for name, value in state.items():
        print(f"{name}: {value!r} | type={type(value).__name__}")
    print("\n4. 项目验收清单（目标，不表示已经实现）：")
    for case in ACCEPTANCE_CASES:
        print(f"{case.case_id} [{case.capability}] {case.question}")
        print("  预期：", case.expected_behavior)
    print("\n5. 非法输入拦截演示：")
    try:
        UserRequest(session_id="xiaolin-capstone-001", question="   ")
    except ValidationError as error:
        print(json.dumps(error.errors(include_url=False), ensure_ascii=False, default=str))
    else:
        raise AssertionError("空白问题本应被拒绝。")
    print("\n本步数据流：原始输入 → UserRequest验证 → initial_state → AgentState")
    print("下一步：把这些状态接入LangGraph节点和条件边。")


if __name__ == "__main__":
    main()
