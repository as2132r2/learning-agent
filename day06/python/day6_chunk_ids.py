import hashlib
import warnings
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# 1. 确定知识库目录
# ==================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
knowledge_directory = day_directory / "documents"


# ==================================================
# 2. 批量加载 txt 和 Markdown 文件
# ==================================================

loader = DirectoryLoader(
    path=str(knowledge_directory),
    glob=[
        "**/*.txt",
        "**/*.md"
    ],
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    },
    recursive=True,
    silent_errors=False,
    use_multithreading=False
)

source_documents = loader.load()

# 为了让输出顺序稳定，按照源文件路径排序。
source_documents.sort(
    key=lambda document: document.metadata["source"]
)


print("原始文件产生的 Document 数量：")
print(len(source_documents))


# ==================================================
# 3. 规范化每份原始文档的 metadata
# ==================================================

for document in source_documents:
    absolute_source = Path(
        document.metadata["source"]
    ).resolve()

    # relative_to() 把绝对路径转换为相对 knowledge/ 的路径。
    #
    # 绝对路径可能因为换电脑而变化：
    # /home/a/project/knowledge/file.txt
    # /home/b/project/knowledge/file.txt
    #
    # 相对路径在两个项目中都可以保持：
    # file.txt
    relative_source = absolute_source.relative_to(
        knowledge_directory.resolve()
    )

    document.metadata = {
        "source": relative_source.as_posix(),
        "file_name": absolute_source.name,
        "file_type": absolute_source.suffix.lower(),
        "title": absolute_source.stem
    }


print("\n原始文档及规范化 metadata：")

for document in source_documents:
    print("\n来源：", document.metadata["source"])
    print("正文长度：", len(document.page_content))
    print("metadata：", document.metadata)


# ==================================================
# 4. 创建文本切分器
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    add_start_index=True,
    separators=[
        "\n\n",
        "\n",
        "。",
        "；",
        "，",
        ""
    ]
)


# ==================================================
# 5. 批量切分所有原始 Document
# ==================================================

# 输入：list[Document]
# 输出：list[Document]
chunks = text_splitter.split_documents(
    source_documents
)


print("\n" + "=" * 60)
print("切分后的 chunk 数量：")
print(len(chunks))


# ==================================================
# 6. 定义稳定 chunk ID 生成函数
# ==================================================

def create_chunk_id(chunk: Document) -> str:
    """
    根据文本块的稳定信息生成 ID。

    输入：Document
    输出：str，例如 chunk-a1b2c3d4e5f6a7b8
    """

    source = chunk.metadata["source"]
    start_index = chunk.metadata["start_index"]
    content = chunk.page_content

    # 把影响文本块身份的三个字段拼成一个字符串。
    identity_text = (
        f"source={source}\n"
        f"start_index={start_index}\n"
        f"content={content}"
    )

    # SHA-256 接收 bytes，所以先使用 UTF-8 编码。
    identity_bytes = identity_text.encode(
        "utf-8"
    )

    # hexdigest() 返回64位十六进制字符串。
    full_hash = hashlib.sha256(
        identity_bytes
    ).hexdigest()

    # 教学示例取前16位，使终端输出更容易阅读。
    short_hash = full_hash[:16]

    return f"chunk-{short_hash}"


# ==================================================
# 7. 为每个 chunk 设置 ID 和附加 metadata
# ==================================================

for chunk in chunks:
    chunk_id = create_chunk_id(chunk)

    # Document.id 是 LangChain 的文档编号字段。
    chunk.id = chunk_id

    # metadata 中也保存一份，方便某些向量数据库返回时读取。
    chunk.metadata["chunk_id"] = chunk_id


# ==================================================
# 8. 查看所有 chunk
# ==================================================

for number, chunk in enumerate(
    chunks,
    start=1
):
    print("\n" + "-" * 60)
    print("顺序编号：", number)
    print("稳定 chunk ID：", chunk.id)
    print("来源：", chunk.metadata["source"])
    print("起始位置：", chunk.metadata["start_index"])
    print("正文长度：", len(chunk.page_content))
    print("正文：")
    print(repr(chunk.page_content))
    print("完整 metadata：")
    print(chunk.metadata)


# ==================================================
# 9. 验证相同数据能生成相同 ID
# ==================================================

first_chunk = chunks[0]

generated_again = create_chunk_id(
    first_chunk
)


print("\n" + "=" * 60)
print("第一个 chunk 原 ID：")
print(first_chunk.id)

print("\n使用相同数据再次生成的 ID：")
print(generated_again)

print("\n两次 ID 是否相同：")
print(first_chunk.id == generated_again)


# ==================================================
# 10. 检查所有 ID 是否唯一
# ==================================================

chunk_ids = [
    chunk.id
    for chunk in chunks
]

unique_chunk_ids = set(
    chunk_ids
)


print("\n" + "=" * 60)
print("chunk 总数：", len(chunk_ids))
print("唯一 ID 数量：", len(unique_chunk_ids))
print(
    "所有 ID 是否唯一：",
    len(chunk_ids) == len(unique_chunk_ids)
)


# ==================================================
# 11. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "knowledge/ 文件 "
    "→ DirectoryLoader "
    "→ list[Document] "
    "→ TextSplitter "
    "→ list[Document] chunks"
)

print(
    "相对来源 + start_index + page_content "
    "→ UTF-8 bytes "
    "→ SHA-256 "
    "→ 稳定 chunk_id"
)
