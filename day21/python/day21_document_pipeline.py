"""Day21 第三步：加载 TXT、Markdown、PDF，并切分为带来源的 Document。

直接运行默认测试资料：
    python day21/python/day21_document_pipeline.py

也可以指定一个或多个文件：
    python day21/python/day21_document_pipeline.py 文件1.txt 文件2.pdf
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parents[2]
DAY21_DIR = PROJECT_DIR / "day21"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

DEFAULT_FILES = [
    DAY21_DIR / "documents" / "day21_sample.txt",
    PROJECT_DIR / "day06" / "documents" / "markdown" / "day6_langgraph.md",
    PROJECT_DIR / "day06" / "documents" / "pdf" / "day6_rag_test.pdf",
]


def relative_source(file_path: Path) -> str:
    """优先返回相对于项目根目录的来源路径。"""

    try:
        return str(file_path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(file_path.resolve())


def load_file(file_path: Path) -> list[Document]:
    """根据扩展名选择加载器，并统一文档元数据。"""

    resolved_path = file_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"文件不存在：{resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix}")

    if suffix == ".pdf":
        loaded_documents = PyPDFLoader(str(resolved_path)).load()
    else:
        loaded_documents = TextLoader(
            str(resolved_path),
            encoding="utf-8",
            autodetect_encoding=False,
        ).load()

    source = relative_source(resolved_path)
    normalized_documents: list[Document] = []

    for document_index, document in enumerate(loaded_documents):
        # PyPDFLoader 的 page 从0开始；对用户显示时转换成从1开始。
        raw_page = document.metadata.get("page")
        display_page = int(raw_page) + 1 if raw_page is not None else None

        metadata = {
            "source": source,
            "title": resolved_path.stem,
            "file_type": suffix.removeprefix("."),
            "page": display_page,
            "document_index": document_index,
        }

        normalized_documents.append(
            Document(
                page_content=document.page_content.strip(),
                metadata=metadata,
            )
        )

    return normalized_documents


def load_documents(file_paths: list[Path]) -> list[Document]:
    """加载多个文件，并合并成一个 Document 列表。"""

    documents: list[Document] = []
    for file_path in file_paths:
        documents.extend(load_file(file_path))
    return documents


def make_chunk_id(document: Document) -> str:
    """根据来源、页码、起点和正文生成稳定文本块编号。"""

    raw_key = "|".join(
        [
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", "")),
            str(document.metadata.get("start_index", "")),
            document.page_content,
        ]
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]
    return f"chunk-{digest}"


def split_documents(documents: list[Document]) -> list[Document]:
    """递归切分文档，并为每块增加 start_index 和 chunk_id。"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.id = make_chunk_id(chunk)
    return chunks


def display_location(document: Document) -> str:
    source = str(document.metadata["source"])
    page = document.metadata.get("page")
    return f"{source}，第 {page} 页" if page is not None else source


def main() -> None:
    file_paths = (
        [Path(argument) for argument in sys.argv[1:]]
        if len(sys.argv) > 1
        else DEFAULT_FILES
    )

    print("Day21 多格式 Document 加载与切分")
    print("支持类型：", sorted(SUPPORTED_SUFFIXES))
    print("=" * 78)
    print("准备加载：")
    for file_path in file_paths:
        print("-", file_path)

    documents = load_documents(file_paths)
    print("\n加载后的Document数量：", len(documents))

    for index, document in enumerate(documents, start=1):
        preview = document.page_content[:100].replace("\n", " ")
        print(f"\n原始Document {index}")
        print("位置：", display_location(document))
        print("字符数：", len(document.page_content))
        print("metadata：", document.metadata)
        print("正文预览：", repr(preview))

    chunks = split_documents(documents)
    print("\n" + "=" * 78)
    print("切分后的文本块数量：", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print("\n" + "-" * 78)
        print(f"文本块 {index}")
        print("chunk_id：", chunk.id)
        print("位置：", display_location(chunk))
        print("start_index：", chunk.metadata.get("start_index"))
        print("字符数：", len(chunk.page_content))
        print("正文：", repr(chunk.page_content))

    print("\n" + "=" * 78)
    print("数据流：")
    print("文件路径 → TextLoader/PyPDFLoader → list[Document]")
    print("list[Document] → RecursiveCharacterTextSplitter")
    print("→ metadata + start_index + chunk_id → list[Document]文本块")


if __name__ == "__main__":
    main()
