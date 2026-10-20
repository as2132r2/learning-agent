"""Day21 第二步：文档上传校验、哈希去重与原子保存。

运行：python day21/python/day21_upload_validation_server.py
访问：http://127.0.0.1:8026
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from aiohttp import web


HOST = "127.0.0.1"
PORT = 8026
MAX_FILE_SIZE = 1024 * 1024

ALLOWED_TYPES: dict[str, set[str]] = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {
        "text/markdown",
        "text/plain",
        "application/octet-stream",
    },
}

DAY21_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY21_DIR / "html" / "day21_upload_validation.html"
UPLOAD_DIR = DAY21_DIR / "uploads_validated"


def display_filename(original_name: str) -> str:
    """生成仅用于显示的安全文件名，不用于决定最终磁盘路径。"""

    basename = Path(original_name).name
    cleaned = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]", "_", basename)
    return cleaned.strip("._") or "document.txt"


def validate_text_content(file_bytes: bytes) -> tuple[str, int]:
    """验证UTF-8文本内容，返回解码文本和非空行数量。"""

    if b"\x00" in file_bytes:
        raise web.HTTPBadRequest(text="文本文件中不能包含 NUL 字节。")

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise web.HTTPBadRequest(text="文件必须使用 UTF-8 编码。") from error

    if not text.strip():
        raise web.HTTPBadRequest(text="文件不能只包含空白字符。")

    line_count = sum(1 for line in text.splitlines() if line.strip())
    return text, line_count


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(HTML_FILE)


async def upload_handler(request: web.Request) -> web.Response:
    if not request.content_type.startswith("multipart/"):
        raise web.HTTPBadRequest(text="必须使用 multipart/form-data。")

    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "document" or not field.filename:
        raise web.HTTPBadRequest(text="缺少 document 文件字段。")

    original_name = field.filename
    safe_display_name = display_filename(original_name)
    suffix = Path(safe_display_name).suffix.lower()
    declared_mime = (field.headers.get("Content-Type") or "").lower()

    if suffix not in ALLOWED_TYPES:
        raise web.HTTPUnsupportedMediaType(
            text="只允许 .txt 和 .md 文件。"
        )
    if declared_mime not in ALLOWED_TYPES[suffix]:
        raise web.HTTPUnsupportedMediaType(
            text=f"扩展名 {suffix} 与声明的 MIME {declared_mime!r} 不匹配。"
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    total_size = 0
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="upload-",
            suffix=".tmp",
            dir=UPLOAD_DIR,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            while True:
                chunk = await field.read_chunk(size=64 * 1024)
                if not chunk:
                    break

                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise web.HTTPRequestEntityTooLarge(
                        max_size=MAX_FILE_SIZE,
                        actual_size=total_size,
                    )

                hasher.update(chunk)
                temporary_file.write(chunk)

        if total_size == 0:
            raise web.HTTPBadRequest(text="不能上传空文件。")

        file_bytes = temporary_path.read_bytes()
        decoded_text, non_empty_lines = validate_text_content(file_bytes)
        sha256 = hasher.hexdigest()

        # 最终文件名由内容哈希决定，不信任客户端文件名。
        stored_name = f"sha256-{sha256[:24]}{suffix}"
        final_path = UPLOAD_DIR / stored_name
        duplicate = final_path.exists()

        if duplicate:
            temporary_path.unlink(missing_ok=True)
        else:
            # 同一文件系统内原子替换：不会让其他读取者看到半个文件。
            os.replace(temporary_path, final_path)

        temporary_path = None

        return web.json_response(
            {
                "status": "duplicate" if duplicate else "saved",
                "original_name": original_name,
                "display_name": safe_display_name,
                "stored_name": stored_name,
                "declared_mime": declared_mime,
                "detected_content": "utf-8-text",
                "size_bytes": total_size,
                "character_count": len(decoded_text),
                "non_empty_lines": non_empty_lines,
                "sha256": sha256,
                "duplicate": duplicate,
                "saved_path": str(final_path),
            },
            status=200 if duplicate else 201,
        )

    finally:
        # 校验失败时删除尚未正式保存的临时文件。
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@web.middleware
async def json_error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response(
            {
                "status": "error",
                "http_status": error.status,
                "message": error.text,
            },
            status=error.status,
        )


def create_app() -> web.Application:
    app = web.Application(
        middlewares=[json_error_middleware],
        client_max_size=MAX_FILE_SIZE + 64 * 1024,
    )
    app.router.add_get("/", index_handler)
    app.router.add_post("/upload", upload_handler)
    return app


def main() -> None:
    print("Day21 安全文档上传服务已启动。")
    print(f"页面：http://{HOST}:{PORT}")
    print("保存目录：", UPLOAD_DIR)
    print("校验：扩展名、MIME、大小、UTF-8、NUL、空内容、SHA-256")
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
