from langchain_core.documents import Document


# ==================================================
# 1. 创建一个 Document 对象
# ==================================================

# Document 是 LangChain 中表示“一份文本资料”的数据类型。
#
# 主要参数：
# page_content: str
#     文档的正文内容，必须是字符串。
#
# metadata: dict
#     文档的附加信息，例如来源、标题、页码和分类。
#     metadata 不会成为正文，但检索后可以用来显示来源。
#
# id: str | None
#     文档的可选唯一编号。
document = Document(
    page_content=(
        "LangChain 是一个用于开发大语言模型应用的框架。"
        "它可以连接模型、提示词、工具、记忆和检索系统。"
    ),
    metadata={
        "source": "day5_notes.txt",
        "title": "LangChain 简介",
        "page": 1,
        "category": "编程学习"
    },
    id="document-001"
)


# ==================================================
# 2. 查看完整对象及其 Python 类型
# ==================================================

print("完整 Document 对象：")
print(document)

print("\n" + "-" * 50)
print("Document 的 Python 类型：")
print(type(document))


# ==================================================
# 3. 分别读取正文、元数据和编号
# ==================================================

print("\n" + "-" * 50)
print("正文 page_content：")
print(document.page_content)

print("\n正文的数据类型：")
print(type(document.page_content))


print("\n" + "-" * 50)
print("元数据 metadata：")
print(document.metadata)

print("\n元数据的数据类型：")
print(type(document.metadata))


print("\n" + "-" * 50)
print("文档编号 id：")
print(document.id)

print("\n文档编号的数据类型：")
print(type(document.id))


# ==================================================
# 4. 读取 metadata 字典中的单个字段
# ==================================================

source = document.metadata["source"]
title = document.metadata["title"]
page = document.metadata["page"]

print("\n" + "-" * 50)
print("来源：", source)
print("标题：", title)
print("页码：", page)

print("\n页码的数据类型：")
print(type(page))


# ==================================================
# 5. 创建多个 Document
# ==================================================

# RAG 通常处理的不是一个 Document，
# 而是由多个 Document 组成的列表。
documents = [
    document,
    Document(
        page_content=(
            "LangGraph 用图结构组织大模型应用的工作流程，"
            "并支持状态、分支、循环和持久化。"
        ),
        metadata={
            "source": "day5_notes.txt",
            "title": "LangGraph 简介",
            "page": 2,
            "category": "编程学习"
        },
        id="document-002"
    ),
    Document(
        page_content=(
            "RAG 会先从外部资料中检索相关内容，"
            "再把检索结果和用户问题一起交给模型。"
        ),
        metadata={
            "source": "day5_notes.txt",
            "title": "RAG 简介",
            "page": 3,
            "category": "编程学习"
        },
        id="document-003"
    )
]


print("\n" + "=" * 50)
print("Document 列表：")
print(documents)

print("\n列表的数据类型：")
print(type(documents))

print("\nDocument 数量：")
print(len(documents))


# ==================================================
# 6. 遍历 Document 列表
# ==================================================

print("\n" + "=" * 50)
print("逐份显示文档：")

for number, current_document in enumerate(
    documents,
    start=1
):
    print(f"\n第 {number} 份文档")
    print("编号：", current_document.id)
    print("标题：", current_document.metadata["title"])
    print("来源：", current_document.metadata["source"])
    print("页码：", current_document.metadata["page"])
    print("正文：", current_document.page_content)


# ==================================================
# 7. 最基础的数据流
# ==================================================

print("\n" + "=" * 50)
print("当前步骤的数据流：")
print(
    "原始文本 + 来源信息 "
    "→ Document "
    "→ documents 列表 "
    "→ 后续文本切分和检索"
)
