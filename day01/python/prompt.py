import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名耐心的编程老师，请用适合{level}学习者的方式回答。"),
    ("human", "请解释什么是{topic}，并给出一个简单例子。")
])

messages = prompt.invoke({
    "level": "零基础",
    "topic": "Python函数"
})

response = llm.invoke(messages)

print(response.content)
