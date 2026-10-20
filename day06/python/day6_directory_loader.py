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


# ==================================================
# 1. 确定知识库目录
# ==================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
knowledge_directory = day_directory / "documents"


print("知识库目录：")
print(knowledge_directory)

print("\n目录是否存在：")
print(knowledge_directory.exists())


# ==================================================
# 2. 创建目录加载器
# ==================================================

# DirectoryLoader 负责：
#     在指定目录中寻找匹配的文件。
#
# TextLoader 负责：
#     打开每一个找到的文件并创建 Document。
#
# 两者的关系：
#     DirectoryLoader 找文件
#     → 对每个文件创建 TextLoader
#     → TextLoader.load()
#     → 汇总所有 Document
loader = DirectoryLoader(
    # 要扫描的目录。
    path=str(knowledge_directory),

    # 同时寻找 txt 和 Markdown 文件。
    glob=[
        "**/*.txt",
        "**/*.md"
    ],

    # 指定每个文件都使用 TextLoader 读取。
    loader_cls=TextLoader,

    # 这些参数会继续传给每个 TextLoader。
    loader_kwargs={
        "encoding": "utf-8"
    },

    # 允许扫描子目录。
    recursive=True,

    # 读取失败时不要静默跳过，而是抛出异常。
    silent_errors=False,

    # 当前文件很少，按顺序加载更方便学习和观察。
    use_multithreading=False
)


print("\n" + "=" * 60)
print("DirectoryLoader 对象：")
print(loader)

print("\n对象类型：")
print(type(loader))


# ==================================================
# 3. 批量加载目录中的文件
# ==================================================

# load() 返回 list[Document]。
# 通常每个文本文件产生一个 Document。
documents = loader.load()


# DirectoryLoader 的文件读取顺序不应当作为业务规则。
# 为了让终端输出每次一致，这里按照 source 排序。
documents.sort(
    key=lambda document: document.metadata["source"]
)


print("\n" + "=" * 60)
print("加载结果的数据类型：")
print(type(documents))

print("\n加载到的 Document 数量：")
print(len(documents))


# ==================================================
# 4. 根据文件路径补充 metadata
# ==================================================

for document in documents:
    # TextLoader 自动保存的 source 是文件路径字符串。
    source_path = Path(
        document.metadata["source"]
    )

    # name：包含扩展名的文件名。
    # 例如 day6_langgraph.md
    document.metadata["file_name"] = (
        source_path.name
    )

    # suffix：文件扩展名。
    # 例如 .md 或 .txt
    document.metadata["file_type"] = (
        source_path.suffix.lower()
    )

    # stem：不包含扩展名的文件名。
    # 例如 day6_langgraph
    document.metadata["title"] = (
        source_path.stem
    )


# ==================================================
# 5. 查看每一份 Document
# ==================================================

for number, document in enumerate(
    documents,
    start=1
):
    print("\n" + "-" * 60)
    print("Document 编号：", number)
    print("文件名：", document.metadata["file_name"])
    print("文件类型：", document.metadata["file_type"])
    print("标题：", document.metadata["title"])
    print("完整来源：", document.metadata["source"])
    print("正文长度：", len(document.page_content))

    print("正文预览：")
    print(document.page_content[:100])

    print("Document 类型：")
    print(type(document))


# ==================================================
# 6. 按文件类型进行简单统计
# ==================================================

file_type_counts = {}

for document in documents:
    file_type = document.metadata["file_type"]

    file_type_counts[file_type] = (
        file_type_counts.get(file_type, 0)
        + 1
    )


print("\n" + "=" * 60)
print("文件类型统计：")
print(file_type_counts)


# ==================================================
# 7. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "knowledge/ 目录 "
    "→ glob 匹配 *.txt 和 *.md "
    "→ 为每个文件调用 TextLoader "
    "→ 合并成 list[Document]"
)

print(
    "Document.metadata['source'] "
    "→ Path "
    "→ file_name + file_type + title"
)
