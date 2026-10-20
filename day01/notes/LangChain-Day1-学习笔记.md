# LangChain Day 1 学习笔记

> 适合零基础复习。学习目标：看懂一个最小 LangChain 程序如何从环境配置出发，调用 DeepSeek，组织消息与提示词，并把模型输出校验成 Python 对象。

## 1. 今天学到了什么

```text
Linux 终端与 Python 环境
        ↓
.env 读取模型配置
        ↓
ChatOpenAI 创建模型调用对象
        ↓
invoke() 发起一次调用
        ↓
AIMessage 保存模型回复
        ↓
Messages 区分规则和用户问题
        ↓
ChatPromptTemplate 动态生成 Messages
        ↓
Chain / RunnableSequence 串联处理步骤
        ↓
Structured Output + Pydantic 得到可靠的 Python 对象
```

项目目录：

```text
/path/to/learning-agent/
├── .venv/          # 项目的独立 Python 环境
├── .env            # API Key、Base URL、模型名等配置
├── .gitignore      # 告诉 Git 不要追踪敏感或临时文件
├── hello.py        # 第一次调用模型
├── messages.py     # SystemMessage / HumanMessage
├── prompt.py       # ChatPromptTemplate
└── structured.py   # Structured Output
```

---

## 2. Python、Linux 与 venv

### 2.1 文件夹和文件不是一回事

```bash
cd /path/to/learning-agent
```

`cd` 是 **change directory**，只能切换到目录。`hello.py` 是文件，所以执行 `cd hello.py` 会得到“不是目录”的错误。

```bash
nano hello.py    # 打开并编辑文件
python day01/python/hello.py  # 用 Python 运行文件
```

### 2.2 为什么使用虚拟环境

`.venv` 是这个项目自己的 Python 运行环境。它把本项目的依赖与系统 Python、其他项目隔离开，避免不同项目需要不同版本时互相干扰。

每次重新打开终端后：

```bash
cd /path/to/learning-agent
source .venv/bin/activate
```

命令行前面出现 `(.venv)`，说明虚拟环境已经激活。然后可运行：

```bash
python day01/python/hello.py
```

退出虚拟环境可执行：

```bash
deactivate
```

### 2.3 一条完整的日常操作链

```text
打开终端
  ↓
cd 进入项目目录
  ↓
source .venv/bin/activate
  ↓
nano 文件名.py（编辑）
  ↓
python 文件名.py（运行）
```

---

## 3. `.env` 与 `.gitignore`

### 3.1 `.env` 的作用

`.env` 用来保存容易变化的配置和敏感信息，例如：

```dotenv
DEEPSEEK_API_KEY=你的API密钥
DEEPSEEK_BASE_URL=你的API地址
DEEPSEEK_MODEL=你的模型名
```

不要把真实密钥写进学习笔记、代码示例或提交到 Git 仓库。

Python 中的读取过程：

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = os.getenv("DEEPSEEK_MODEL")
```

流程：

```text
.env 保存配置
    ↓
load_dotenv() 加载配置
    ↓
os.getenv("变量名") 读取配置
    ↓
Python 把配置交给 ChatOpenAI
```

### 3.2 为什么不把 API Key 直接写进代码

- 安全：防止密钥被误传到 GitHub、Gitee 或发给别人。
- 易维护：更换模型、地址或密钥时，不必修改业务代码。
- 易复用：同一份代码可以在不同环境使用不同配置。

### 3.3 `.gitignore` 的作用

建议至少包含：

```gitignore
.env
.venv/
__pycache__/
```

含义：

- `.env`：不提交敏感配置。
- `.venv/`：不提交庞大的本地虚拟环境。
- `__pycache__/`：不提交 Python 自动产生的缓存。

注意：`.gitignore` 只阻止尚未被 Git 追踪的文件。如果密钥已经提交过，单纯补写 `.gitignore` 并不能消除历史记录，还需要立即撤销或更换密钥。

---

## 4. `ChatOpenAI`：模型调用对象

项目中的基础代码：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)
```

`ChatOpenAI` 是 LangChain 提供的聊天模型封装。这里通过 `base_url` 和相应配置连接 DeepSeek 的 OpenAI 兼容接口。

`llm` 不是模型本身，也不是模型的回答，而是一个已经配置好的 Python 对象。它大致知道：

```text
llm
├── 使用哪个模型
├── API 地址在哪里
├── 用哪个密钥认证
├── 怎样发送请求
└── 怎样把返回数据包装成 LangChain 对象
```

重要认识：**接口兼容不等于全部能力完全相同。** 一个服务可以兼容 OpenAI 的基础调用格式，但未必支持所有参数或所有结构化输出方式。

---

## 5. `invoke()` 与 `AIMessage`

### 5.1 发起一次调用

```python
response = llm.invoke("你好，请用一句话介绍你自己")
```

`invoke()` 可以先理解为“调用这个组件一次”。执行过程：

```text
Python 输入
   ↓
llm.invoke(...)
   ↓
API 请求
   ↓
模型生成回答
   ↓
AIMessage
```

LangChain 不只有模型能调用 `invoke()`；提示词模板和 Chain 等 Runnable 组件也采用相似的统一接口。

### 5.2 返回值不是普通字符串

```python
print(type(response))
```

会看到类似：

```text
<class 'langchain_core.messages.ai.AIMessage'>
```

也就是说，`response` 是 `AIMessage` 对象。它可能包含：

```text
AIMessage
├── content            回答正文
├── response_metadata  模型、结束原因等信息
├── usage_metadata     Token 使用情况
└── tool_calls         工具调用信息（以后学习）
```

因此：

```python
print(response)          # 打印整个 AIMessage 对象
print(response.content)  # 只打印回答正文
```

这体现了 Python 中重要的“对象.属性”思维：`response.content` 是从 `response` 对象里取出 `content` 属性。

---

## 6. Messages：不同角色的消息

核心代码：

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(
        content="你是一名耐心的 Python 老师，正在教一个零基础学生。"
    ),
    HumanMessage(
        content="请用生活中的例子解释什么是 Python 变量。"
    )
]

response = llm.invoke(messages)
print(response.content)
```

三个基本角色：

| 角色 | 含义 | 例子 |
|---|---|---|
| `SystemMessage` | 高层指令、行为规则和背景 | “你是一名耐心的老师，回答不超过 100 字。” |
| `HumanMessage` | 用户当前提出的具体问题或任务 | “请解释 Python 变量。” |
| `AIMessage` | 模型生成的回复 | “变量可以理解为贴了标签的盒子……” |

流程：

```text
SystemMessage：规定怎样回答
            +
HumanMessage：规定回答什么
            ↓
           模型
            ↓
AIMessage：模型的回答
```

`SystemMessage` 不只是“给模型设定身份”，它还可以规定语言、格式、篇幅和行为边界。

---

## 7. `ChatPromptTemplate`：可重复使用的动态提示词

如果问题中的主题和学习者水平经常变化，不应每次手工拼接消息。可以把变化部分写成模板变量：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名耐心的编程老师，请用适合{level}学习者的方式回答。"
    ),
    (
        "human",
        "请解释什么是{topic}，并给出一个简单例子。"
    )
])

messages = prompt.invoke({
    "level": "零基础",
    "topic": "Python函数"
})

response = llm.invoke(messages)
print(response.content)
```

这里的 `{level}`、`{topic}` 是占位符。运行时传入字典后，模板才生成真正的 Messages。

```text
输入字典
{"level": "零基础", "topic": "Python函数"}
             ↓
ChatPromptTemplate 填充变量
             ↓
SystemMessage + HumanMessage
             ↓
llm.invoke(messages)
             ↓
AIMessage
```

要分清职责：

- `ChatPromptTemplate`：生成 Messages，不负责请求模型。
- `ChatOpenAI`：调用模型。
- `AIMessage`：保存普通聊天模型的返回结果。

---

## 8. Chain 与 `RunnableSequence`

手动写法是两步：

```python
messages = prompt.invoke({
    "level": "零基础",
    "topic": "Python函数"
})
response = llm.invoke(messages)
```

可以用 `|` 把组件串成 Chain：

```python
chain = prompt | llm

response = chain.invoke({
    "level": "零基础",
    "topic": "Python函数"
})

print(response.content)
```

把 `|` 理解为：**将左边组件的输出交给右边组件。**

```text
字典输入
   ↓
prompt
   ↓ 输出 Messages
llm
   ↓
AIMessage
```

这样的顺序组合通常是 `RunnableSequence`，即“多个可运行组件按顺序组成的可执行流程”。

```python
print(type(chain))
```

会看到类似：

```text
<class 'langchain_core.runnables.base.RunnableSequence'>
```

关键理解：

- `Runnable`：可通过统一接口执行的组件。
- `RunnableSequence`：多个 Runnable 首尾相接形成的顺序流程。
- `chain.invoke(data)`：让输入依次通过整条链。

---

## 9. Structured Output：让输出成为可靠的 Python 对象

普通调用通常得到自由文本形式的 `AIMessage`。程序若要稳定读取“概念、定义、示例、难度”等字段，就需要结构化输出。

### 9.1 使用 Pydantic 定义 Schema

```python
from pydantic import BaseModel, Field


class ConceptExplanation(BaseModel):
    concept: str = Field(description="需要解释的概念名称")
    definition: str = Field(description="概念的简短定义")
    example: str = Field(description="一个简单易懂的例子")
    difficulty: int = Field(description="学习难度，1到5分")
```

概念解释：

- `BaseModel`：Pydantic 的模型基类。继承它后，这个类就能解析并校验数据。
- 类型标注（如 `str`、`int`）：规定字段应该是什么数据类型。
- `Field(description=...)`：为字段补充含义，帮助模型和开发者理解要求。
- Schema：可以理解为数据必须遵守的“结构说明书”。

更严格地限制 1 到 5，也可以写成：

```python
difficulty: int = Field(
    ge=1,
    le=5,
    description="学习难度，1到5分"
)
```

`ge=1` 表示大于等于 1，`le=5` 表示小于等于 5。

### 9.2 创建结构化模型调用对象

```python
structured_llm = llm.with_structured_output(
    ConceptExplanation,
    method="json_mode"
)
```

再调用：

```python
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
```

成功后：

```python
print(type(result))
print(result.concept)
print(result.definition)
print(result.example)
print(result.difficulty)
```

`result` 不再是普通字符串或 `AIMessage`，而是：

```text
<class '__main__.ConceptExplanation'>
```

因此可以稳定地用 `result.concept` 等属性读取数据。

完整流程：

```text
用户要求
   ↓
LLM 生成 JSON
   ↓
LangChain 解析 JSON
   ↓
Pydantic 按 Schema 校验字段和类型
   ↓
ConceptExplanation Python 对象
```

---

## 10. DeepSeek、`json_mode` 与兼容问题

学习时遇到过：

```text
This response_format type is unavailable now
```

原因不是 Python 语法错误，而是当前 DeepSeek 兼容接口不支持 LangChain 默认尝试的那种结构化输出能力。解决方式是明确使用：

```python
structured_llm = llm.with_structured_output(
    ConceptExplanation,
    method="json_mode"
)
```

这里最重要的工程认识是：

```text
“兼容 OpenAI API”
        ≠
“支持 OpenAI API 的所有功能和参数”
```

另外，JSON Mode 的重点是要求输出合法 JSON。为了让内容更符合 Schema，提示词中还应明确：

- 必须输出 JSON。
- 必须有哪些字段。
- 每个字段的数据类型与取值范围。
- 给出一份符合要求的 JSON 示例。

---

## 11. `ValidationError`：JSON 合法不代表 Schema 合法

学习时模型曾返回：

```json
{
  "concept": "Python函数",
  "definition": "……",
  "example": "……",
  "difficulty": "beginner"
}
```

这是一份合法 JSON，但 Schema 要求：

```python
difficulty: int
```

模型实际给的是字符串：

```text
"beginner"
```

所以 Pydantic 报错：

```text
ValidationError
Input should be a valid integer
```

失败发生的位置：

```text
模型成功返回
   ↓
JSON 解析成功
   ↓
Pydantic 检查 difficulty
   ↓
预期 int，实际 str
   ↓
ValidationError / OutputParserException
```

必须牢牢记住：

```text
合法 JSON ≠ 符合 Schema
```

修复思路：

1. 在提示词里明确数据类型与取值范围。
2. 提供正确的 JSON 示例。
3. 在 `Field` 中增加 `ge=1`、`le=5` 等校验规则。
4. 真实项目中捕获解析异常，并根据业务需要重试、修正或提示用户；不要假设模型每次都会完全遵守格式。

这正是 Pydantic 的价值：不让格式错误的数据悄悄进入后续业务逻辑。

---

## 12. 四个练习文件分别在训练什么

| 文件 | 重点 | 典型返回值 |
|---|---|---|
| `hello.py` | 创建模型对象并完成第一次调用 | `AIMessage` |
| `messages.py` | 区分系统规则与用户任务 | `AIMessage` |
| `prompt.py` | 用变量动态生成 Messages | `AIMessage` |
| `structured.py` | 用 Schema 约束并校验模型输出 | `ConceptExplanation` |

对比：

```text
普通 llm.invoke(...)
        ↓
AIMessage
        ↓
response.content

structured_llm.invoke(...)
        ↓
Pydantic 模型对象
        ↓
result.concept / result.difficulty
```

---

## 13. 常见错误速查

### 错误 1：`cd hello.py` 提示“不是目录”

原因：`cd` 只能进入文件夹。

```bash
nano hello.py    # 编辑文件
python day01/python/hello.py  # 运行文件
```

### 错误 2：运行时提示缺少模块

常见原因：虚拟环境没有激活，或依赖没有安装在当前环境中。先确认命令行前面是否有 `(.venv)`。

### 错误 3：配置值是 `None` 或认证失败

检查：

- 是否在正确项目目录运行。
- `.env` 是否存在，变量名是否与 `os.getenv()` 完全一致。
- 是否执行了 `load_dotenv()`。
- API Key 是否有效。

不要为了排查问题直接打印完整 API Key。

### 错误 4：把 `response` 当作字符串

`response` 通常是 `AIMessage`。需要正文时使用：

```python
print(response.content)
```

### 错误 5：模板变量没有传全或名字拼错

模板有 `{level}` 和 `{topic}`，调用时的字典就必须提供同名键：

```python
prompt.invoke({"level": "零基础", "topic": "Python函数"})
```

### 错误 6：结构化输出接口不兼容

看到 `response_format ... unavailable` 时，要想到不同模型服务的能力并不完全相同。当前练习明确使用：

```python
method="json_mode"
```

### 错误 7：JSON 能解析，但 Pydantic 校验失败

查看错误中的：

- 哪个字段出错。
- 期望什么类型。
- 实际值和实际类型是什么。

例如 `difficulty` 期望 `int`，却收到了字符串 `"beginner"`。

---

## 14. Day 1 复习问题

先遮住后面的答案，用自己的话回答。

1. `.venv` 有什么用？每次重新打开终端后如何激活？
2. `.env` 有什么用？为什么 API Key 不应直接写进 `.py` 文件？
3. `.gitignore` 为什么应该包含 `.env`、`.venv/` 和 `__pycache__/`？
4. `ChatOpenAI` 是模型本身、模型回复，还是模型调用对象？
5. `invoke()` 可以怎样理解？
6. 普通 `llm.invoke()` 的返回值为什么不是字符串？
7. `response` 和 `response.content` 有什么区别？
8. `SystemMessage` 与 `HumanMessage` 分别负责什么？
9. `AIMessage` 中除了正文还可能有什么？
10. `ChatPromptTemplate` 的职责是什么？它会自己调用模型吗？
11. `{level}` 和 `{topic}` 是什么？调用模板时怎样提供值？
12. `prompt | llm` 中的 `|` 表示什么？
13. 什么是 `RunnableSequence`？
14. Structured Output 解决了什么问题？
15. `BaseModel`、类型标注和 `Field` 各有什么作用？
16. 为什么合法 JSON 仍然可能触发 `ValidationError`？
17. `difficulty: int` 与 `"difficulty": "beginner"` 冲突在哪里？
18. 为什么连接 DeepSeek 时要显式使用 `method="json_mode"`？
19. 普通模型调用与结构化模型调用最终拿到的 Python 对象有什么不同？
20. `json_mode` 下怎样提高模型遵守 Schema 的概率？

### 参考答案（简版）

1. 隔离项目依赖；执行 `source .venv/bin/activate`。
2. 保存配置和敏感信息；避免泄露并实现代码与配置分离。
3. 防止提交敏感配置、本地环境和缓存文件。
4. 已配置好的模型调用对象。
5. 调用或执行该组件一次。
6. LangChain 用 `AIMessage` 同时保存正文、元数据和工具调用等信息。
7. 前者是完整对象，后者是其中的回答正文属性。
8. 前者给高层规则与上下文，后者给用户的具体任务。
9. 模型信息、Token 用量、结束原因、工具调用等。
10. 根据输入变量生成 Messages；不会自己请求模型。
11. 模板占位符；通过包含同名键的字典传值。
12. 把左边输出交给右边。
13. 多个 Runnable 按顺序连接成的可执行流程。
14. 将不稳定的自然语言输出变成可校验、可按字段读取的数据对象。
15. 定义可校验模型、规定字段类型、补充字段说明和限制。
16. JSON 只保证语法合法，不保证字段和值符合 Schema。
17. Schema 要整数，模型却返回了字符串。
18. 当前 DeepSeek 兼容接口不支持默认尝试的结构化输出方式。
19. 前者通常得到 `AIMessage`，后者得到自定义 Pydantic 对象。
20. 明确要求 JSON、列出字段与类型、写清范围并提供正确示例。

---

## 15. 建议的 Day 1 动手复习

只阅读不如亲手改几次。可以完成以下小实验：

1. 在 `hello.py` 中分别打印 `response`、`response.content` 和 `type(response)`。
2. 保持 `HumanMessage` 不变，修改三种不同的 `SystemMessage`，观察语气和长度变化。
3. 在 `prompt.py` 中把 `topic` 改为“Python列表”，把 `level` 改为“10岁小朋友”。
4. 把手动的 `prompt.invoke()` + `llm.invoke()` 改写成 `prompt | llm`。
5. 在 `structured.py` 中打印 `type(result)` 和 `type(result.difficulty)`。
6. 暂时把提示词中的 `difficulty` 类型说明删除，观察模型是否仍稳定输出整数；实验后恢复。
7. 给 `difficulty` 的 `Field` 增加 `ge=1, le=5`，理解 Schema 不仅能检查类型，也能检查范围。

---

## 16. Day 2：从 Tools / Function Calling 开始

明天建议创建：

```text
tools.py
```

先写一个最简单的计算工具，再让模型决定何时调用它。学习主线：

```text
用户提出问题
   ↓
LLM 判断是否需要工具
   ↓
生成 Tool Call（工具名 + 参数）
   ↓
Python 执行工具函数
   ↓
ToolMessage 把执行结果送回模型
   ↓
LLM 组织最终回答
```

计算示例：

```text
用户：“23 × 47 等于多少？”
   ↓
模型选择 calculator
   ↓
calculator(23, 47)
   ↓
1081
   ↓
结果返回模型
   ↓
模型回答：“23 × 47 = 1081”
```

Day 2 建议按这个顺序学习：

1. 什么是 Tool，为什么 LLM 需要工具。
2. 用 Python 定义最小 `calculator` 工具。
3. Function Calling：模型如何生成工具名和参数。
4. 查看 `AIMessage.tool_calls`。
5. 在 Python 中真正执行工具。
6. 用 `ToolMessage` 把工具结果送回模型。
7. 完成一次完整的“判断 → 调用 → 返回 → 最终回答”。
8. 理解 LLM、Tool Calling 与 Agent 的关系。
9. 最后再进入 `create_agent`，不要一开始就把流程藏进 Agent。

后续总路线：

```text
Tools / Function Calling
        ↓
Agent
        ↓
Memory / Context
        ↓
Streaming
        ↓
小型 Agent 项目
        ↓
RAG
        ↓
Embedding / Vector Store / Retriever
        ↓
完整 RAG 项目
        ↓
LangGraph
        ↓
MCP
        ↓
FastAPI 与工程化
```

---

## 17. Day 1 最小记忆卡

如果时间有限，只复习下面这些：

```text
1. source .venv/bin/activate
   → 激活项目独立环境

2. .env → load_dotenv() → os.getenv()
   → 安全读取模型配置

3. llm = ChatOpenAI(...)
   → 创建模型调用对象

4. llm.invoke(...)
   → 执行一次模型调用

5. 普通返回值是 AIMessage
   → 回答正文用 response.content

6. SystemMessage：怎样回答
   HumanMessage：回答什么
   AIMessage：模型回答

7. ChatPromptTemplate
   → 用变量动态生成 Messages

8. prompt | llm
   → 左边输出交给右边，组成 RunnableSequence

9. Structured Output
   → JSON → Schema → Pydantic 校验 → Python 对象

10. 合法 JSON ≠ 符合 Schema
    → 类型不符会触发 ValidationError

11. OpenAI 兼容 API ≠ 支持所有 OpenAI 能力
    → 当前 DeepSeek 练习使用 method="json_mode"

12. 下一课：Tools / Function Calling
```

一句话总结 Day 1：

> 今天学会了把配置安全地交给 LangChain，用统一的消息、模板和 Chain 调用模型，并用 Pydantic 把不稳定的模型文本转换成程序可验证、可使用的数据对象。
