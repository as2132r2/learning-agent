"""Day21 第一步：浏览器 multipart/form-data 文档上传。

运行：python day21/python/day21_upload_basic_server.py
访问：http://127.0.0.1:8025
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web


HOST = "127.0.0.1"
PORT = 8025
MAX_FILE_SIZE = 1024 * 1024  # 1 MiB
ALLOWED_SUFFIXES = {".txt", ".md"}

DAY21_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = DAY21_DIR / "html" / "day21_upload_basic.html"
UPLOAD_DIR = DAY21_DIR / "uploads"


def safe_filename(original_name: str) -> str:
    """移除路径信息和危险字符，并加入唯一前缀。"""

    basename = Path(original_name).name
    cleaned = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]", "_", basename)
    cleaned = cleaned.strip("._") or "document.txt"
    return f"{uuid.uuid4().hex[:10]}-{cleaned}"


async def index_handler(request: web.Request) -> web.FileResponse:
    """返回文件上传页面。"""

    return web.FileResponse(HTML_FILE)


async def upload_handler(request: web.Request) -> web.Response:
    """读取 multipart 文件字段，限制大小并保存到 uploads 目录。"""

    if not request.content_type.startswith("multipart/"):
        raise web.HTTPBadRequest(
            text="请求必须使用 multipart/form-data。"
        )

    reader = await request.multipart()
    file_field = await reader.next()

    if file_field is None or file_field.name != "document":
        raise web.HTTPBadRequest(text="缺少名为 document 的文件字段。")

    original_name = file_field.filename or ""
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise web.HTTPUnsupportedMediaType(
            text="当前步骤只允许上传 .txt 和 .md 文件。"
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = safe_filename(original_name)
    target_path = UPLOAD_DIR / stored_name

    total_size = 0
    chunks: list[bytes] = []

    while True:
        chunk = await file_field.read_chunk(size=64 * 1024)
        if not chunk:
            break

        total_size += len(chunk)
        if total_size > MAX_FILE_SIZE:
            raise web.HTTPRequestEntityTooLarge(
                max_size=MAX_FILE_SIZE,
                actual_size=total_size,
            )
        chunks.append(chunk)

    if total_size == 0:
        raise web.HTTPBadRequest(text="不能上传空文件。")

    file_bytes = b"".join(chunks)
    try:
        decoded_text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise web.HTTPBadRequest(
            text="文件必须使用 UTF-8 编码。"
        ) from error

    target_path.write_bytes(file_bytes)

    response_data: dict[str, Any] = {
        "status": "saved",
        "original_name": original_name,
        "stored_name": stored_name,
        "size_bytes": total_size,
        "character_count": len(decoded_text),
        "suffix": suffix,
        "saved_path": str(target_path),
    }

    print(
        f"[上传成功] original={original_name!r} "
        f"stored={stored_name!r} size={total_size}"
    )
    return web.json_response(response_data, status=201)


@web.middleware
async def error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """将 HTTP 异常统一转换成浏览器容易处理的 JSON。"""

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
        middlewares=[error_middleware],
        client_max_size=MAX_FILE_SIZE + 64 * 1024,
    )
    app.router.add_get("/", index_handler)
    app.router.add_post("/upload", upload_handler)
    return app


def main() -> None:
    print("Day21 浏览器文档上传服务已启动。")
    print(f"页面：http://{HOST}:{PORT}")
    print("允许类型：", sorted(ALLOWED_SUFFIXES))
    print("最大文件：", MAX_FILE_SIZE, "bytes")
    print("保存目录：", UPLOAD_DIR)
    print("按 Ctrl+C 停止服务。")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
