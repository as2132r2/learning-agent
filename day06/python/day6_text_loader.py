import warnings
from pathlib import Path


# langchain-community 当前会显示整体迁移提示。
# 这里只隐藏这一条弃用提示，不会隐藏真正的程序错误。
warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_community.document_loaders import TextLoader


# ==================================================
# 1. 构造真实文件的绝对路径
# ==================================================

# __file__：当前 Python 文件的路径。
# resolve()：转换为绝对路径。
# parent：取得当前 Python 文件所在的目录。
script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent

# 使用 / 拼接路径：
# 项目目录 / knowledge / day6_langchain.txt
text_file_path = (
    day_directory
    / "documents"
    / "txt"
    / "day6_langchain.txt"
)


print("准备读取的文件：")
print(text_file_path)

print("\n文件是否存在：")
print(text_file_path.exists())


# ==================================================
# 2. 创建 TextLoader
# ==================================================

# TextLoader 参数：
#
# file_path: str | Path
#     要读取的文本文件路径。
#
# encoding: str | None
#     文件编码。中文文本通常明确使用 utf-8。
loader = TextLoader(
    file_path=text_file_path,
    encoding="utf-8"
)


print("\n" + "=" * 60)
print("Loader 对象：")
print(loader)

print("\nLoader 的 Python 类型：")
print(type(loader))


# ==================================================
# 3. 调用 load() 读取文件
# ==================================================

# load() 不需要额外参数。
#
# 返回值始终是 list[Document]。
# 即使只读取一个文件，也不会直接返回单个 Document。
documents = loader.load()


print("\n" + "=" * 60)
print("load() 返回值：")
print(documents)

print("\n返回值的数据类型：")
print(type(documents))

print("\nDocument 数量：")
print(len(documents))


# ==================================================
# 4. 取得第一个 Document
# ==================================================

document = documents[0]


print("\n" + "=" * 60)
print("第一个 Document 的类型：")
print(type(document))

print("\nDocument 正文 page_content：")
print(document.page_content)

print("\n正文的数据类型：")
print(type(document.page_content))

print("\n正文字符数量：")
print(len(document.page_content))


# ==================================================
# 5. 查看 Loader 自动添加的 metadata
# ==================================================

print("\n" + "=" * 60)
print("Document metadata：")
print(document.metadata)

print("\nmetadata 的数据类型：")
print(type(document.metadata))

print("\nsource 字段：")
print(document.metadata.get("source"))


# ==================================================
# 6. 补充自己的 metadata
# ==================================================

# Loader 自动生成 source，
# 我们也可以继续加入标题、分类等业务信息。
document.metadata["title"] = "LangChain 学习资料"
document.metadata["category"] = "编程学习"
document.metadata["file_type"] = "txt"


print("\n" + "=" * 60)
print("补充后的 metadata：")
print(document.metadata)


# ==================================================
# 7. 当前步骤的数据流
# ==================================================

print("\n" + "=" * 60)
print("当前步骤的数据流：")
print(
    "day6_langchain.txt 文件 "
    "→ TextLoader "
    "→ loader.load() "
    "→ list[Document]"
)

print(
    "文件正文 "
    "→ Document.page_content"
)

print(
    "文件路径 "
    "→ Document.metadata['source']"
)
