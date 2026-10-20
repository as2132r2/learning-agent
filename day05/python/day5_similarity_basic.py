import numpy as np

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


# ==================================================
# 3. 生成文档向量与问题向量
# ==================================================

# 这是学习用的测试 Embedding，不理解真实语义。
embedding_model = DeterministicFakeEmbedding(
    size=8
)

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

document_vectors = embedding_model.embed_documents(
    chunk_texts
)

question = "什么是 RAG？"

query_vector = embedding_model.embed_query(
    question
)


# ==================================================
# 4. 定义余弦相似度函数
# ==================================================

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:
    """
    计算两个向量的余弦相似度。

    参数：
        vector_a: list[float]
        vector_b: list[float]

    返回值：
        float

    基本公式：
        点积 /（向量A的长度 × 向量B的长度）
    """

    # np.asarray() 把 Python 列表转换为 NumPy 数组，
    # 方便进行向量数学运算。
    array_a = np.asarray(
        vector_a,
        dtype=float
    )

    array_b = np.asarray(
        vector_b,
        dtype=float
    )

    # 两个向量必须具有相同维度。
    if array_a.shape != array_b.shape:
        raise ValueError(
            "两个向量的维度必须相同。"
        )

    # 点积：对应位置相乘后再求和。
    dot_product = np.dot(
        array_a,
        array_b
    )

    # 向量长度，也称为 L2 范数。
    norm_a = np.linalg.norm(array_a)
    norm_b = np.linalg.norm(array_b)

    # 零向量没有方向，不能计算余弦相似度。
    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "零向量不能计算余弦相似度。"
        )

    similarity = (
        dot_product
        / (norm_a * norm_b)
    )

    # NumPy 计算结果是 np.float64；
    # float() 把它转换为普通 Python float。
    return float(similarity)


# ==================================================
# 5. 先用简单向量理解分数含义
# ==================================================

print("简单向量演示：")

same_direction = cosine_similarity(
    [1.0, 0.0],
    [2.0, 0.0]
)

perpendicular = cosine_similarity(
    [1.0, 0.0],
    [0.0, 1.0]
)

opposite_direction = cosine_similarity(
    [1.0, 0.0],
    [-1.0, 0.0]
)

print("方向相同：", same_direction)
print("方向垂直：", perpendicular)
print("方向相反：", opposite_direction)

print(
    "\n通常可以理解为："
    "越接近 1 越相似，"
    "接近 0 表示方向关系弱，"
    "接近 -1 表示方向相反。"
)


# ==================================================
# 6. 比较问题和所有文本块
# ==================================================

# similarity_results 的数据类型：
# list[dict]
#
# 每个字典同时保存：
# - 文本块编号
# - 相似度分数
# - 对应的 Document
similarity_results = []

for index, document_vector in enumerate(
    document_vectors
):
    score = cosine_similarity(
        query_vector,
        document_vector
    )

    similarity_results.append({
        "chunk_number": index + 1,
        "score": score,
        "document": chunks[index]
    })


print("\n" + "=" * 60)
print("用户问题：")
print(question)

print("\n排序前的相似度结果：")

for result in similarity_results:
    print("\n文本块编号：", result["chunk_number"])
    print("相似度：", round(result["score"], 4))
    print("正文：", result["document"].page_content)


# ==================================================
# 7. 按照相似度从高到低排序
# ==================================================

# key 指定按照每个字典中的 score 排序。
# reverse=True 表示从大到小排列。
ranked_results = sorted(
    similarity_results,
    key=lambda item: item["score"],
    reverse=True
)


print("\n" + "=" * 60)
print("按照相似度从高到低排序：")

for rank, result in enumerate(
    ranked_results,
    start=1
):
    print("\n排名：", rank)
    print("文本块编号：", result["chunk_number"])
    print("相似度：", round(result["score"], 4))
    print("正文：", result["document"].page_content)


# ==================================================
# 8. 取相似度最高的文本块
# ==================================================

top_result = ranked_results[0]
retrieved_document = top_result["document"]

print("\n" + "=" * 60)
print("检索到的第一名文本块：")
print(retrieved_document.page_content)

print("\n第一名分数：")
print(round(top_result["score"], 4))

print("\n第一名 metadata：")
print(retrieved_document.metadata)


# ==================================================
# 9. 当前代码的重要限制
# ==================================================

print("\n" + "=" * 60)
print("重要说明：")
print(
    "当前使用 DeterministicFakeEmbedding，"
    "它只生成稳定的测试向量，不理解中文语义。"
)
print(
    "因此第一名不一定真的是 RAG 文本块。"
    "现在只观察相似度计算、排序和取第一名的数据流程。"
)


# ==================================================
# 10. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "问题向量 + 每个文档向量 "
    "→ 余弦相似度分数 "
    "→ 从高到低排序 "
    "→ 取最相关 Document"
)
