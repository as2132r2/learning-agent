from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# 1. 创建原始文档
# ==================================================

original_document = Document(
    page_content=(
        "LangChain 是一个用于开发大语言模型应用的框架。"
        "它可以连接模型、提示词、工具、记忆和检索系统。\n\n"
        "LangGraph 使用图结构组织应用流程。"
        "节点负责处理数据，边负责决定流程走向。\n\n"
        "RAG 会先从外部知识库检索相关资料，"
        "再把资料和用户问题一起交给大语言模型。"
    ),
    metadata={
        "source": "day5_notes.txt",
        "title": "LangChain 学习资料"
    },
    id="document-001"
)


# ==================================================
# 2. 把长文档切成多个 Document
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=65,
    chunk_overlap=10,
    add_start_index=True,
    separators=["\n\n", "\n", "。", "，", ""]
)

chunks = text_splitter.split_documents(
    [original_document]
)


print("文本块数量：", len(chunks))

for number, chunk in enumerate(chunks, start=1):
    print(f"\n第 {number} 个文本块：")
    print(chunk.page_content)
    print("metadata：", chunk.metadata)


# ==================================================
# 3. 创建本地测试 Embedding 模型
# ==================================================

# size=8 表示每段文本会转换成包含 8 个浮点数的向量。
#
# DeterministicFakeEmbedding 的作用是学习和测试数据流程：
# - 不访问网络
# - 不产生 API 费用
# - 相同文本每次得到相同向量
#
# 重要：它不理解文本语义，不能用于真正的 RAG 检索。
embedding_model = DeterministicFakeEmbedding(
    size=8
)


print("\n" + "=" * 60)
print("Embedding 模型对象：")
print(embedding_model)

print("\nEmbedding 模型类型：")
print(type(embedding_model))


# ==================================================
# 4. 从 Document 中提取字符串正文
# ==================================================

# embed_documents() 接收 list[str]，
# 不能直接接收 list[Document]。
chunk_texts = [
    chunk.page_content
    for chunk in chunks
]


print("\n" + "=" * 60)
print("传给 embed_documents() 的数据：")
print(chunk_texts)

print("\n输入数据类型：")
print(type(chunk_texts))

print("列表中单个元素的类型：")
print(type(chunk_texts[0]))


# ==================================================
# 5. 批量生成文档向量
# ==================================================

# 参数 texts: list[str]
# 返回值: list[list[float]]
document_vectors = embedding_model.embed_documents(
    chunk_texts
)


print("\n" + "=" * 60)
print("文档向量集合的数据类型：")
print(type(document_vectors))

print("\n文档向量数量：")
print(len(document_vectors))

print("\n单个向量的数据类型：")
print(type(document_vectors[0]))

print("\n单个向量的维度：")
print(len(document_vectors[0]))


for number, vector in enumerate(
    document_vectors,
    start=1
):
    print("\n" + "-" * 60)
    print(f"第 {number} 个文本块的向量：")
    print(vector)

    print("对应正文：")
    print(chunks[number - 1].page_content)


# ==================================================
# 6. 为用户问题生成查询向量
# ==================================================

question = "什么是 RAG？"

# embed_query() 接收一个 str，
# 返回一个 list[float]。
query_vector = embedding_model.embed_query(
    question
)


print("\n" + "=" * 60)
print("用户问题：")
print(question)

print("\n问题向量：")
print(query_vector)

print("\n问题向量的数据类型：")
print(type(query_vector))

print("\n问题向量的维度：")
print(len(query_vector))


# ==================================================
# 7. 验证确定性
# ==================================================

# 相同文本再次生成向量，结果仍然相同。
same_query_vector = embedding_model.embed_query(
    question
)

print("\n" + "=" * 60)
print("相同问题的两次向量是否相等：")
print(query_vector == same_query_vector)


# ==================================================
# 8. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "Document "
    "→ 文本切分 "
    "→ list[Document] "
    "→ 提取 list[str] "
    "→ embed_documents() "
    "→ list[list[float]]"
)

print(
    "用户问题 str "
    "→ embed_query() "
    "→ list[float]"
)
