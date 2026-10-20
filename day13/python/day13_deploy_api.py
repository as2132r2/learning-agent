"""Day13 第八步：可通过环境变量部署、支持优雅退出的API入口。"""

import os
import signal
from http.server import ThreadingHTTPServer
from threading import Thread

from dotenv import load_dotenv

import day13_http_stream_api as stream_api
import day13_http_session_api as session_api


load_dotenv()


def read_port() -> int:
    """读取并校验HTTP_PORT。"""

    text = os.getenv("HTTP_PORT", "8000").strip()
    try:
        port = int(text)
    except ValueError as error:
        raise RuntimeError("HTTP_PORT必须是整数。") from error

    if not 1 <= port <= 65535:
        raise RuntimeError("HTTP_PORT必须在1到65535之间。")
    return port


HTTP_HOST = os.getenv("HTTP_HOST", "127.0.0.1").strip()
HTTP_PORT = read_port()

if not HTTP_HOST:
    raise RuntimeError("HTTP_HOST不能为空。")


class DeploymentServer(ThreadingHTTPServer):
    """适合练习部署的多线程HTTP服务器配置。"""

    allow_reuse_address = True
    daemon_threads = True


server = DeploymentServer(
    (HTTP_HOST, HTTP_PORT),
    stream_api.StreamingApiHandler
)
shutdown_started = False


def handle_shutdown_signal(signum: int, _frame: object) -> None:
    """收到SIGINT/SIGTERM后，让serve_forever安全退出。"""

    global shutdown_started

    if shutdown_started:
        return

    shutdown_started = True
    signal_name = signal.Signals(signum).name
    print(f"\n收到{signal_name}，正在停止接收新请求……")

    # shutdown不能在运行serve_forever的同一线程直接调用。
    Thread(target=server.shutdown, daemon=True).start()


signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


print("Day13部署版API已启动。")
print(f"监听地址：http://{HTTP_HOST}:{HTTP_PORT}")
print("进程编号PID：", os.getpid())
print("SIGINT和SIGTERM都会触发优雅退出。")

try:
    server.serve_forever(poll_interval=0.5)
finally:
    server.server_close()
    session_api.database_connection.close()
    print("HTTP监听端口已关闭。")
    print("SQLite数据库连接已关闭。")
    print("服务已安全退出。")
