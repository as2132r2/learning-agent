import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

messages = [
    SystemMessage(content="你是一名耐心的 Python 老师，正在教一个零基础学生。"),
    HumanMessage(content="请用生活中的例子解释什么是 Python 变量。")
]

response = llm.invoke(messages)

print(response.content)
