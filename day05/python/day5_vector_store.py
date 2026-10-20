from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
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
# 2. 文本切分
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


# 为每个文本块设置一个容易观察的唯一编号。
# split_documents() 产生的新 Document 默认可能没有 id。
for index, chunk in enumerate(chunks, start=1):
    chunk.id = f"chunk-{index:03d}"


print("切分后的 Document 数量：", len(chunks))

for chunk in chunks:
    print("\n文本块 ID：", chunk.id)
    print("正文：", chunk.page_content)
    print("metadata：", chunk.metadata)


# ==================================================
# 3. 创建 Embedding 模型
# ==================================================

# 测试模型不理解语义，只用于学习向量数据库接口。
embedding_model = DeterministicFakeEmbedding(
    size=8
)


# ==================================================
# 4. 创建内存向量数据库
# ==================================================

# InMemoryVectorStore 参数：
# embedding: Embeddings
#     指定用哪个 Embedding 模型生成向量。
#
# 数据只存在当前 Python 进程的内存里，
# 程序退出后不会保留。
vector_store = InMemoryVectorStore(
    embedding=embedding_model
)


print("\n" + "=" * 60)
print("向量数据库对象：")
print(vector_store)

print("\n向量数据库类型：")
print(type(vector_store))


# ==================================================
# 5. 把 Document 加入向量数据库
# ==================================================

# add_documents() 内部会自动完成：
#
# Document.page_content
# → embedding_model.embed_documents()
# → 文档向量
# → 保存向量、正文、metadata 和 id
#
# 参数 documents: list[Document]
# 返回值: list[str]，表示成功保存的文档 ID。
stored_ids = vector_store.add_documents(
    documents=chunks
)


print("\n" + "=" * 60)
print("add_documents() 返回的数据：")
print(stored_ids)

print("\n返回值类型：")
print(type(stored_ids))

print("\n保存的 ID 数量：")
print(len(stored_ids))


# ==================================================
# 6. 进行相似度检索
# ==================================================

question = "什么是 RAG？"

# similarity_search_with_score() 内部会自动完成：
#
# question
# → embedding_model.embed_query()
# → 问题向量
# → 和数据库中的文档向量比较
# → 排序
# → 返回前 k 个结果
#
# 返回值类型：
# list[tuple[Document, float]]
search_results = (
    vector_store.similarity_search_with_score(
        query=question,
        k=3
    )
)


print("\n" + "=" * 60)
print("用户问题：")
print(question)

print("\n检索结果的数据类型：")
print(type(search_results))

print("\n检索结果数量：")
print(len(search_results))


# ==================================================
# 7. 查看排序后的检索结果
# ==================================================

for rank, search_result in enumerate(
    search_results,
    start=1
):
    # 每个元素是一个二元组：
    # (Document, score)
    retrieved_document, score = search_result

    print("\n" + "-" * 60)
    print("排名：", rank)
    print("文本块 ID：", retrieved_document.id)
    print("相似度分数：", round(score, 4))
    print("正文：", retrieved_document.page_content)
    print("metadata：", retrieved_document.metadata)

    print("返回元素的类型：")
    print(type(search_result))

    print("其中 Document 的类型：")
    print(type(retrieved_document))

    print("其中 score 的类型：")
    print(type(score))


# ==================================================
# 8. 只返回 Document，不返回分数
# ==================================================

# similarity_search() 的流程相同，
# 但返回值只有 list[Document]。
documents_only = vector_store.similarity_search(
    query=question,
    k=2
)


print("\n" + "=" * 60)
print("只返回 Document 的前 2 个结果：")

for document in documents_only:
    print("\nID：", document.id)
    print("正文：", document.page_content)


# ==================================================
# 9. 和上一小步对比
# ==================================================

print("\n" + "=" * 60)
print("上一小步需要自己完成：")
print(
    "生成问题向量 → 遍历文档向量 "
    "→ 计算余弦相似度 → 排序"
)

print("\n这一小步只需要调用：")
print(
    "vector_store.similarity_search(query, k)"
)


# ==================================================
# 10. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("建立数据库：")
print(
    "list[Document] "
    "→ add_documents() "
    "→ Embedding "
    "→ 向量 + Document 保存到内存"
)

print("\n查询数据库：")
print(
    "问题 str "
    "→ similarity_search() "
    "→ 问题向量 "
    "→ 相似度比较和排序 "
    "→ list[Document]"
)
