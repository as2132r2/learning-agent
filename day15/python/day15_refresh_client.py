"""Day15 第九步客户端：演示登录、刷新、重放和退出。"""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"


def post(path: str, data: dict) -> tuple[int, dict]:
    request = Request(
        BASE_URL + path,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except HTTPError as error:
        return error.code, json.loads(error.read().decode())


print("[1] 模拟登录")
status, tokens = post("/auth/demo-login", {"user_id": "user-xiaolin"})
print("HTTP：", status)
old_refresh = tokens["refresh_token"]

print("\n[2] 使用Refresh Token无感刷新")
status, new_tokens = post("/auth/refresh", {"refresh_token": old_refresh})
print("HTTP：", status)
print("是否获得新Access：", "access_token" in new_tokens)
print("Refresh是否已经轮换：", old_refresh != new_tokens["refresh_token"])

print("\n[3] 重放旧Refresh Token")
status, replay = post("/auth/refresh", {"refresh_token": old_refresh})
print("HTTP：", status)
print(json.dumps(replay, ensure_ascii=False, indent=2))

print("\n[4] 使用新Refresh Token退出")
status, logout = post(
    "/auth/logout",
    {"refresh_token": new_tokens["refresh_token"]}
)
print("HTTP：", status)
print(json.dumps(logout, ensure_ascii=False, indent=2))

print("\n[5] 退出后再刷新")
status, after_logout = post(
    "/auth/refresh",
    {"refresh_token": new_tokens["refresh_token"]}
)
print("HTTP：", status)
print(json.dumps(after_logout, ensure_ascii=False, indent=2))
