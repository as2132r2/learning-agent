import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ============================================================
# 1. 创建基础模型
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 定义主管必须返回的数据结构
# ============================================================

class DelegationDecision(BaseModel):
    """主管为当前请求生成的一次任务委派决定。"""

    worker: Literal[
        "researcher",
        "calculator",
        "writer"
    ] = Field(
        description="最适合执行当前任务的专业Agent"
    )

    task: str = Field(
        description="交给专业Agent的具体、独立、可执行任务"
    )

    expected_output: str = Field(
        description="专业Agent完成任务后应该返回的结果"
    )

    reason: str = Field(
        description="主管选择该专业Agent的简短原因"
    )


# 普通模型被包装成主管模型。
# 它的返回值必须是 DelegationDecision，而不是自由文本。
supervisor_llm = llm.with_structured_output(
    DelegationDecision,
    method="function_calling"
)


# ============================================================
# 3. 接收用户请求
# ============================================================

user_request = input("请输入一个需要交给专业Agent的任务：").strip()

if not user_request:
    raise ValueError("任务不能为空。")


# ============================================================
# 4. 主管选择专业Agent并改写任务
# ============================================================

decision: DelegationDecision = supervisor_llm.invoke([
    SystemMessage(
        content=(
            "你是多Agent系统的主管，只负责分派任务，不亲自完成任务。"
            "你可以选择三个专业Agent：\n"
            "1. researcher：负责查找、分析和解释知识资料；\n"
            "2. calculator：负责数学运算和数值验证；\n"
            "3. writer：负责整理、改写和生成结构清晰的成品文本。\n"
            "请选择最适合当前请求的一个Agent。"
            "task必须是一条可以脱离原始对话独立理解的完整指令。"
        )
    ),
    HumanMessage(content=user_request)
])


# ============================================================
# 5. 查看结构化分派结果
# ============================================================

print("\n" + "=" * 60)
print("主管的完整决策对象：")
print(decision)

print("\n决策对象的数据类型：")
print(type(decision))

print("\n选择的专业Agent：")
print(decision.worker)

print("\n交给专业Agent的任务：")
print(decision.task)

print("\n预期输出：")
print(decision.expected_output)

print("\n分派原因：")
print(decision.reason)


# ============================================================
# 6. 转换成普通Python字典
# ============================================================

decision_dictionary = decision.model_dump()

print("\n" + "=" * 60)
print("转换后的字典：")
print(decision_dictionary)

print("\n字典的数据类型：")
print(type(decision_dictionary))


# ============================================================
# 本步骤的数据流
# ============================================================
#
# 用户请求: str
#      ↓
# supervisor_llm.invoke()
#      ↓
# DelegationDecision
#      ├── worker
#      ├── task
#      ├── expected_output
#      └── reason
#      ↓ model_dump()
# dict

