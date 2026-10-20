from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# 1. 准备原始文档
# ==================================================

original_documents = [
    Document(
        page_content=(
            "LangChain 是一个用于开发大语言模型应用的框架。"
            "它可以连接模型、提示词、工具、记忆和检索系统。"
        ),
        metadata={
            "source": "langchain.txt",
            "topic": "LangChain"
        },
        id="document-001"
    ),
    Document(
        page_content=(
            "LangGraph 使用图结构组织应用流程。"
            "节点负责处理数据，边负责决定流程走向，"
            "状态负责在节点之间传递数据。"
        ),
        metadata={
            "source": "langgraph.txt",
            "topic": "LangGraph"
        },
        id="document-002"
    ),
    Document(
        page_content=(
            "RAG 的中文名称是检索增强生成。"
            "它会先从外部知识库中检索相关资料，"
            "再把资料和用户问题一起交给大语言模型。"
        ),
        metadata={
            "source": "rag.txt",
            "topic": "RAG"
        },
        id="document-003"
    )
]


# ==================================================
# 2. 切分文档
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=10,
    add_start_index=True,
    separators=["\n\n", "\n", "。", "，", ""]
)

chunks = text_splitter.split_documents(
    original_documents
)


# 为切分后的文本块设置唯一 ID。
for index, chunk in enumerate(chunks, start=1):
    chunk.id = f"chunk-{index:03d}"


print("文本块数量：", len(chunks))


# ==================================================
# 3. 创建 Embedding 和向量数据库
# ==================================================

embedding_model = DeterministicFakeEmbedding(
    size=8
)

vector_store = InMemoryVectorStore(
    embedding=embedding_model
)

stored_ids = vector_store.add_documents(
    documents=chunks
)


print("保存的文本块 ID：")
print(stored_ids)


# ==================================================
# 4. 把向量数据库转换成 Retriever
# ==================================================

# as_retriever() 返回一个统一的检索器对象。
#
# search_type="similarity"：
#     使用相似度检索。
#
# search_kwargs={"k": 2}：
#     每次返回最相似的 2 个 Document。
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)


print("\n" + "=" * 60)
print("Retriever 对象：")
print(retriever)

print("\nRetriever 的 Python 类型：")
print(type(retriever))


# ==================================================
# 5. 使用 invoke() 检索
# ==================================================

question = "什么是 RAG？"

# retriever 是 Runnable，因此使用 invoke()。
#
# 输入：str
# 输出：list[Document]
retrieved_documents = retriever.invoke(
    question
)


print("\n" + "=" * 60)
print("用户问题：")
print(question)

print("\n检索结果的数据类型：")
print(type(retrieved_documents))

print("\n检索结果数量：")
print(len(retrieved_documents))


for rank, document in enumerate(
    retrieved_documents,
    start=1
):
    print("\n" + "-" * 60)
    print("排名：", rank)
    print("Document ID：", document.id)
    print("正文：", document.page_content)
    print("metadata：", document.metadata)
    print("对象类型：", type(document))


# ==================================================
# 6. 使用 batch() 批量检索多个问题
# ==================================================

# Retriever 继承 Runnable 接口，所以也支持 batch()。
#
# 输入：list[str]
# 输出：list[list[Document]]
questions = [
    "什么是 RAG？",
    "LangGraph 如何组织流程？"
]

batch_results = retriever.batch(
    questions
)


print("\n" + "=" * 60)
print("批量问题：")
print(questions)

print("\nbatch() 返回值类型：")
print(type(batch_results))

print("\n每个元素的类型：")
print(type(batch_results[0]))


for question_number, documents in enumerate(
    batch_results,
    start=1
):
    print("\n" + "-" * 60)
    print("问题：", questions[question_number - 1])

    for document in documents:
        print(
            f"- {document.id}："
            f"{document.page_content}"
        )


# ==================================================
# 7. VectorStore 与 Retriever 的区别
# ==================================================

print("\n" + "=" * 60)
print("VectorStore 的职责：")
print("保存向量和文档，并提供具体的相似度搜索能力。")

print("\nRetriever 的职责：")
print(
    "提供统一检索接口："
    "输入问题，输出 list[Document]。"
)

print("\n为什么还需要 Retriever：")
print(
    "后面的 RAG 链只关心怎样检索 Document，"
    "不需要关心底层使用哪一种向量数据库。"
)


# ==================================================
# 8. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "VectorStore "
    "→ as_retriever(search_kwargs={'k': 2}) "
    "→ Retriever"
)

print(
    "问题 str "
    "→ retriever.invoke() "
    "→ VectorStore 相似度搜索 "
    "→ list[Document]"
)
