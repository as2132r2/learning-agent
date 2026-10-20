import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ============================================================
# 1. 读取环境变量并创建基础模型
# ============================================================

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ============================================================
# 2. 定义“一条计划步骤”的数据结构
# ============================================================

class PlanStep(BaseModel):
    """复杂任务中的一个可执行步骤。"""

    step_id: int = Field(
        description="步骤编号，从 1 开始连续递增"
    )

    description: str = Field(
        description="这一小步需要完成的具体任务"
    )

    expected_output: str = Field(
        description="完成这一步后应该得到的结果"
    )

    status: Literal["pending", "completed"] = Field(
        default="pending",
        description="步骤状态；创建计划时必须是 pending"
    )


# ============================================================
# 3. 定义“完整任务计划”的数据结构
# ============================================================

class TaskPlan(BaseModel):
    """模型为用户目标生成的结构化执行计划。"""

    goal: str = Field(
        description="对用户最终目标的清晰概括"
    )

    steps: list[PlanStep] = Field(
        description="按照执行顺序排列的子任务列表"
    )


# 把普通聊天模型包装成结构化规划模型。
# 返回值不再是自由文本 AIMessage，而是 TaskPlan 对象。
planner_llm = llm.with_structured_output(
    TaskPlan,
    method="function_calling"
)


# ============================================================
# 4. 接收复杂任务
# ============================================================

user_goal = input("请输入一个需要分步骤完成的复杂任务：").strip()

if not user_goal:
    raise ValueError("任务不能为空。")


# ============================================================
# 5. 调用规划模型
# ============================================================

plan: TaskPlan = planner_llm.invoke([
    SystemMessage(
        content=(
            "你是一名任务规划器，只负责制定计划，不执行任务。"
            "请把用户目标拆成 3 到 6 个有先后顺序、可以独立执行的步骤。"
            "每一步必须具体，不要使用‘完成任务’这类空泛描述。"
            "后面的步骤可以使用前面步骤的输出。"
            "所有步骤的初始状态必须是 pending。"
        )
    ),
    HumanMessage(content=user_goal)
])


# ============================================================
# 6. 查看模型返回的数据和类型
# ============================================================

print("\n" + "=" * 60)
print("完整 TaskPlan 对象：")
print(plan)

print("\nTaskPlan 的 Python 类型：")
print(type(plan))

print("\n最终目标：")
print(plan.goal)

print("\n步骤列表的数据类型：")
print(type(plan.steps))

print("\n步骤数量：")
print(len(plan.steps))


# ============================================================
# 7. 逐项读取结构化计划
# ============================================================

print("\n" + "=" * 60)
print("结构化执行计划：")

for step in plan.steps:
    print("\n" + "-" * 60)
    print("步骤编号：", step.step_id)
    print("任务说明：", step.description)
    print("预期输出：", step.expected_output)
    print("当前状态：", step.status)
    print("步骤对象类型：", type(step))


# ============================================================
# 8. 转换为普通 Python 字典
# ============================================================

plan_dictionary = plan.model_dump()

print("\n" + "=" * 60)
print("转换后的 Python 字典：")
print(plan_dictionary)

print("\n字典的数据类型：")
print(type(plan_dictionary))

print("\n第一条步骤的字典：")
print(plan_dictionary["steps"][0])

print("\n第一条步骤字典的数据类型：")
print(type(plan_dictionary["steps"][0]))


# ============================================================
# 9. 本步骤的数据流
# ============================================================

print("\n" + "=" * 60)
print("本步骤的数据流：")
print(
    "用户目标 str"
    " → planner_llm.invoke()"
    " → TaskPlan"
    " → list[PlanStep]"
    " → model_dump()"
    " → dict"
)

