import warnings
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning
)

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# ============================================================
# 1. 知识库目录
# ============================================================

script_directory = Path(__file__).resolve().parent
day_directory = script_directory.parent
knowledge_directory = day_directory / "documents"


# ============================================================
# 2. 根据扩展名选择不同 Loader
# ============================================================

def load_one_file(file_path: Path) -> list[Document]:
    """
    加载一份文件。

    参数：
        file_path: Path
            需要加载的真实文件路径。

    返回值：
        list[Document]
            TXT、MD 通常返回一个 Document；
            PDF 通常每一页返回一个 Document。
    """

    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        # 当前程序不认识的类型返回空列表。
        return []

    documents = loader.load()

    # 不同 Loader 产生的 metadata 不完全相同，
    # 在这里统一成后续索引程序容易使用的结构。
    relative_source = file_path.relative_to(
        knowledge_directory
    ).as_posix()

    for document_number, document in enumerate(documents, start=1):
        original_metadata = document.metadata.copy()

        # PyPDFLoader 的 page 从 0 开始，因此对用户显示时加 1。
        pdf_page = original_metadata.get("page")
        page_number = pdf_page + 1 if pdf_page is not None else None

        document.metadata = {
            "source": relative_source,
            "file_name": file_path.name,
            "file_type": suffix,
            "title": file_path.stem,
            "page": page_number,
            "document_number": document_number
        }

    return documents


# ============================================================
# 3. 扫描目录中的所有受支持文件
# ============================================================

supported_suffixes = {".txt", ".md", ".pdf"}

knowledge_files = sorted(
    file_path
    for file_path in knowledge_directory.rglob("*")
    if file_path.is_file()
    and file_path.suffix.lower() in supported_suffixes
)

print("知识库目录：", knowledge_directory)
print("支持的文件类型：", sorted(supported_suffixes))
print("发现的文件数量：", len(knowledge_files))


# ============================================================
# 4. 逐个加载，并合并为同一个 Document 列表
# ============================================================

all_documents: list[Document] = []

for file_path in knowledge_files:
    loaded_documents = load_one_file(file_path)
    all_documents.extend(loaded_documents)

    print("\n[加载文件]", file_path.name)
    print("Loader 返回的 Document 数量：", len(loaded_documents))


# ============================================================
# 5. 查看统一后的结果
# ============================================================

print("\n" + "=" * 60)
print("全部 Document 数量：", len(all_documents))

type_counts: dict[str, int] = {}

for document in all_documents:
    file_type = document.metadata["file_type"]
    type_counts[file_type] = type_counts.get(file_type, 0) + 1

print("不同类型产生的 Document 数量：", type_counts)

for index, document in enumerate(all_documents, start=1):
    preview = document.page_content[:100].replace("\n", " ")

    print("\n" + "-" * 60)
    print(f"第 {index} 份 Document")
    print("来源：", document.metadata["source"])
    print("文件类型：", document.metadata["file_type"])
    print("页码：", document.metadata["page"])
    print("正文长度：", len(document.page_content))
    print("正文预览：", preview)


# ============================================================
# 6. 总结数据流
# ============================================================

pdf_count = sum(
    1
    for file_path in knowledge_files
    if file_path.suffix.lower() == ".pdf"
)

print("\n" + "=" * 60)

if pdf_count == 0:
    print("当前 knowledge 目录中没有 PDF 文件。")
    print("可以复制一份带可选中文字的 PDF 到 knowledge 目录后再次运行。")
else:
    print("本次加载的 PDF 文件数量：", pdf_count)

print("\n数据流：")
print("文件路径 Path")
print("  → 根据扩展名选择 TextLoader 或 PyPDFLoader")
print("  → loader.load()")
print("  → list[Document]")
print("  → 统一 metadata")
print("  → all_documents")
