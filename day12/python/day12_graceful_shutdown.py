"""Day12 第九步：演示服务的优雅停机和资源释放。"""

from dataclasses import dataclass


class ServiceStoppingError(RuntimeError):
    """服务已经进入停机阶段，不再接受新请求。"""


@dataclass
class ServiceLifecycle:
    """管理服务是否接受请求，以及当前活跃请求数量。"""

    accepting_requests: bool = False
    active_requests: int = 0
    database_closed: bool = False
    model_client_closed: bool = False

    def start(self) -> None:
        if self.database_closed or self.model_client_closed:
            raise RuntimeError("已关闭的服务对象不能重新启动。")

        self.accepting_requests = True
        print("[生命周期] 服务已启动，开始接受新请求。")

    def begin_request(self, request_id: str) -> None:
        if not self.accepting_requests:
            raise ServiceStoppingError(
                f"服务正在停机，请求{request_id}被拒绝。"
            )

        self.active_requests += 1
        print(
            f"[请求开始] {request_id}，"
            f"活跃请求数={self.active_requests}"
        )

    def finish_request(self, request_id: str) -> None:
        if self.active_requests <= 0:
            raise RuntimeError("没有活跃请求，却尝试结束请求。")

        self.active_requests -= 1
        print(
            f"[请求完成] {request_id}，"
            f"活跃请求数={self.active_requests}"
        )

    def begin_shutdown(self) -> None:
        self.accepting_requests = False
        print("[生命周期] 停止接受新请求。")
        print(
            "[生命周期] 等待活跃请求完成：",
            self.active_requests
        )

    def close_resources(self) -> None:
        if self.active_requests != 0:
            raise RuntimeError(
                "仍有活跃请求，不能提前关闭底层资源。"
            )

        self.database_closed = True
        print("[资源] 数据库连接已关闭。")

        self.model_client_closed = True
        print("[资源] 模型客户端已关闭。")

    def shutdown(self) -> None:
        self.begin_shutdown()
        self.close_resources()
        print("[生命周期] 服务已安全停止。")


def handle_request(
    lifecycle: ServiceLifecycle,
    request_id: str
) -> None:
    try:
        lifecycle.begin_request(request_id)
    except ServiceStoppingError as error:
        print("[请求拒绝]", error)
        return

    try:
        print(f"[业务处理] 正在处理{request_id}。")
    finally:
        # 无论业务成功还是抛错，都必须减少活跃请求数。
        lifecycle.finish_request(request_id)


print("=" * 60)
print("实验一：正常启动、处理请求和关闭")

runtime = ServiceLifecycle()
runtime.start()
handle_request(runtime, "req-001")
runtime.shutdown()

print("\n" + "=" * 60)
print("实验二：停机后拒绝新请求")

runtime = ServiceLifecycle()
runtime.start()
runtime.begin_request("req-active")
runtime.begin_shutdown()

# 停机期间的新请求被拒绝。
handle_request(runtime, "req-new")

# 原有请求完成后才能关闭资源。
runtime.finish_request("req-active")
runtime.close_resources()
print("停机完成：", runtime.database_closed, runtime.model_client_closed)

print("\n" + "=" * 60)
print("实验三：仍有活跃请求时禁止提前关闭")

runtime = ServiceLifecycle()
runtime.start()
runtime.begin_request("req-active-2")
runtime.begin_shutdown()

try:
    runtime.close_resources()
except RuntimeError as error:
    print("正确阻止提前关闭：", error)

runtime.finish_request("req-active-2")
runtime.close_resources()


# ============================================================
# 数据流
# ============================================================
#
# start()
#    ↓ accepting_requests=True
# begin_request()
#    ↓ active_requests + 1
# 处理业务
#    ↓ finish_request()
#    ↓ active_requests - 1
# begin_shutdown()
#    ↓ accepting_requests=False
#    ↓ 新请求拒绝，旧请求继续完成
# close_resources()
#    ↓ active_requests == 0
#    ↓ 关闭数据库和模型客户端
