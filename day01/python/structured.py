import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()


class ConceptExplanation(BaseModel):
    concept: str = Field(description="需要解释的概念名称")
    definition: str = Field(description="概念的简短定义")
    example: str = Field(description="一个简单易懂的例子")
    difficulty: int = Field(description="学习难度，1到5分")


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


structured_llm = llm.with_structured_output(
    ConceptExplanation,
    method="json_mode"
)


result = structured_llm.invoke(
    """
    请向一个 Python 初学者解释什么是 Python 函数。

    必须以 JSON 格式输出。

    difficulty 必须是 1 到 5 之间的整数，
    不允许输出 beginner、easy、hard 等字符串。

    输出格式示例：

    {
        "concept": "Python函数",
        "definition": "函数是一段可重复使用的代码",
        "example": "def hello(): ...",
        "difficulty": 2
    }
    """
)

print("完整结果：")
print(result)

print("\n--------------------")

print("数据类型：")
print(type(result))

print("\n概念：")
print(result.concept)

print("\n定义：")
print(result.definition)

print("\n例子：")
print(result.example)

print("\n难度：")
print(result.difficulty)
