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
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""

    print(f"[工具正在执行] {a} × {b}")

    return a * b


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


llm_with_tools = llm.bind_tools([
    multiply
])


messages = [
    SystemMessage(
        content="你是一名助手。乘法计算必须使用工具。"
    ),
    HumanMessage(
        content="请计算23乘以47等于多少。"
    )
]


# ==================================================
# 第一次调用：让模型决定是否使用工具
# ==================================================

response = llm_with_tools.invoke(messages)

print("第一次模型响应：")
print(response)

print("\n工具调用请求：")
print(response.tool_calls)


# 模型的 AIMessage 必须加入消息历史
messages.append(response)


# ==================================================
# 读取第一个工具调用请求
# ==================================================

tool_call = response.tool_calls[0]

tool_name = tool_call["name"]
tool_args = tool_call["args"]
tool_call_id = tool_call["id"]


print("\n--------------------")

print("模型选择的工具：", tool_name)
print("模型生成的参数：", tool_args)
print("工具调用ID：", tool_call_id)


# ==================================================
# Python 真正执行工具
# ==================================================

tool_result = multiply.invoke(tool_args)

print("\n工具执行结果：")
print(tool_result)

print("结果类型：")
print(type(tool_result))


# ==================================================
# 把工具结果包装成 ToolMessage
# ==================================================

tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call_id
)

messages.append(tool_message)


print("\n--------------------")

print("ToolMessage：")
print(tool_message)


# ==================================================
# 第二次调用：让模型根据工具结果生成最终回答
# ==================================================

final_response = llm_with_tools.invoke(messages)

print("\n--------------------")

print("最终模型响应：")
print(final_response)

print("\n最终回答：")
print(final_response.content)

