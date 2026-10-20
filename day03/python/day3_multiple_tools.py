import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import tool


load_dotenv()


@tool
def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """计算第一个数字减去第二个数字。"""

    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    return a * b


@tool
def divide(a: float, b: float) -> float:
    """计算第一个数字除以第二个数字。第二个数字不能为零。"""

    if b == 0:
        raise ValueError("除数不能为零")

    return a / b


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# 工具列表：交给模型查看
tools = [
    add,
    subtract,
    multiply,
    divide
]


# 工具字典：让 Python 根据名称查找工具
tools_by_name = {
    current_tool.name: current_tool
    for current_tool in tools
}


llm_with_tools = llm.bind_tools(tools)


user_input = input("请输入一道计算题：").strip()


messages = [
    SystemMessage(
        content=(
            "你是一名计算助手。"
            "遇到加减乘除计算时必须使用工具。"
        )
    ),
    HumanMessage(content=user_input)
]


# 第一次模型调用：选择工具并生成参数
response = llm_with_tools.invoke(messages)

messages.append(response)


print("\n模型正文：")
print(repr(response.content))

print("\n工具调用请求：")
print(response.tool_calls)


# 如果模型没有要求调用工具，就直接输出回答
if not response.tool_calls:
    print("\n最终回答：")
    print(response.content)

else:
    # 一次响应可能包含一个或多个工具调用
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n--------------------")
        print("准备执行工具：", tool_name)
        print("工具参数：", tool_args)

        # 根据名称找到真正的工具对象
        selected_tool = tools_by_name[tool_name]

        # 执行工具
        tool_result = selected_tool.invoke(tool_args)

        print("工具执行结果：", tool_result)

        # 把每个工具结果加入消息历史
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id
            )
        )

    # 第二次模型调用：根据工具结果生成最终回答
    final_response = llm_with_tools.invoke(messages)

    print("\n====================")
    print("最终回答：")
    print(final_response.content)

