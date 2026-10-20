from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# 1. 创建一份较长的原始文档
# ==================================================

original_document = Document(
    page_content=(
        "LangChain 是一个用于开发大语言模型应用的框架。"
        "它可以统一连接模型、提示词、工具、记忆和检索系统。\n\n"
        "LangGraph 用图结构组织应用流程。"
        "节点负责处理数据，边负责决定流程走向，状态负责在节点之间传递数据。\n\n"
        "RAG 的中文名称是检索增强生成。"
        "它会先从外部知识库中找到与问题相关的资料，"
        "再把这些资料和用户问题一起交给大语言模型。\n\n"
        "文本切分是 RAG 的重要准备步骤。"
        "如果文本块太大，可能混入很多无关内容；"
        "如果文本块太小，又可能丢失完整语义。"
    ),
    metadata={
        "source": "day5_notes.txt",
        "title": "LangChain、LangGraph 与 RAG",
        "category": "编程学习"
    },
    id="document-001"
)


print("原始文档正文：")
print(original_document.page_content)

print("\n原始正文长度：")
print(len(original_document.page_content))


# ==================================================
# 2. 创建递归字符文本切分器
# ==================================================

# RecursiveCharacterTextSplitter：
# 尽量按照“段落 → 换行 → 标点 → 字符”的层次切分。
# 它比直接每隔固定字符切一刀更容易保留完整语义。
text_splitter = RecursiveCharacterTextSplitter(
    # 每个文本块期望的最大长度。
    # 默认长度函数是 len，因此这里基本按字符数量计算。
    chunk_size=80,

    # 相邻文本块尝试保留的重复内容长度。
    # 重叠可以减少切分边界造成的语义丢失。
    chunk_overlap=15,

    # 为每个文本块的 metadata 添加 start_index，
    # 表示它在原始正文中的起始位置。
    add_start_index=True,

    # 针对这份中文资料设置优先切分位置。
    # 程序从前往后尝试，优先按段落和句子边界切分。
    separators=[
        "\n\n",
        "\n",
        "。",
        "；",
        "，",
        ""
    ]
)


print("\n" + "=" * 60)
print("文本切分器类型：")
print(type(text_splitter))

print("\nchunk_size：", text_splitter._chunk_size)
print("chunk_overlap：", text_splitter._chunk_overlap)


# ==================================================
# 3. 把 Document 切分成多个小 Document
# ==================================================

# split_documents() 参数：
# documents: list[Document]
#
# 返回值：
# list[Document]
#
# 即使当前只有一份原始文档，也要放进列表传入。
chunks = text_splitter.split_documents(
    [original_document]
)


print("\n" + "=" * 60)
print("切分结果的数据类型：")
print(type(chunks))

print("\n文本块数量：")
print(len(chunks))


# ==================================================
# 4. 查看每个文本块
# ==================================================

for chunk_number, chunk in enumerate(
    chunks,
    start=1
):
    print("\n" + "-" * 60)
    print(f"第 {chunk_number} 个文本块")

    print("正文：")
    print(repr(chunk.page_content))

    print("正文长度：")
    print(len(chunk.page_content))

    print("元数据：")
    print(chunk.metadata)

    print("文本块类型：")
    print(type(chunk))


# ==================================================
# 5. 验证 metadata 的继承
# ==================================================

print("\n" + "=" * 60)
print("原始文档 metadata：")
print(original_document.metadata)

print("\n第一个文本块 metadata：")
print(chunks[0].metadata)

print(
    "\n结论：每个文本块会复制原文档的 metadata，"
    "并额外加入 start_index。"
)


# ==================================================
# 6. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "原始 Document "
    "→ split_documents([Document]) "
    "→ list[Document] 文本块 "
    "→ 下一步生成 Embedding"
)
