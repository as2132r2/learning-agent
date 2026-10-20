import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 读取 .env 文件
load_dotenv()

# 2. 从环境变量中读取配置
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = os.getenv("DEEPSEEK_MODEL")

# 3. 创建一个大模型对象
llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url
)

# 4. 向大模型发送消息
response = llm.invoke("你好，请用一句话介绍你自己")

print("完整 response：")
print(response)

print("\n--------------------")

print("只看 content：")
print(response.content)

print("\n--------------------")

print("response 的 Python 类型：")
print(type(response))

# 5. 打印模型回答
