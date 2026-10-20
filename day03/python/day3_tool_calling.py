import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)
from langchain_core.tools import tool


load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""

    return a * b


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# 把工具的名称、说明和参数 Schema 交给模型
llm_with_tools = llm.bind_tools([
    multiply
])


messages = [
    SystemMessage(
        content="你是一名助手。遇到乘法计算时，必须使用提供的工具。"
    ),
    HumanMessage(
        content="请计算23乘以47等于多少。"
    )
]


response = llm_with_tools.invoke(messages)


print("完整响应：")
print(response)

print("\n--------------------")

print("响应类型：")
print(type(response))

print("\n--------------------")

print("回答正文 content：")
print(repr(response.content))

print("\n--------------------")

print("工具调用 tool_calls：")
print(response.tool_calls)

print("\n--------------------")

print("工具调用数量：")
print(len(response.tool_calls))


if response.tool_calls:
    tool_call = response.tool_calls[0]

    print("\n工具名称：")
    print(tool_call["name"])

    print("\n工具参数：")
    print(tool_call["args"])

    print("\n工具调用 ID：")
    print(tool_call["id"])
else:
    print("\n模型没有请求调用工具。")
