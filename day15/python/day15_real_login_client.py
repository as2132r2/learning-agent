"""Day15 第十一步客户端：输入密码登录并自动刷新一次。"""

import getpass
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post(path: str, data: dict) -> tuple[int, dict]:
    request = Request(
        "http://127.0.0.1:8000" + path,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode())
    except HTTPError as error:
        return error.code, json.loads(error.read().decode())


user_id = input("用户编号：").strip()
password = getpass.getpass("密码：")

status, result = post(
    "/auth/login",
    {"user_id": user_id, "password": password}
)
print("登录HTTP状态码：", status)

if status != 200:
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1)

print("登录成功，已获得Access Token和Refresh Token。")

status, refreshed = post(
    "/auth/refresh",
    {"refresh_token": result["refresh_token"]}
)
print("刷新HTTP状态码：", status)
print("无感刷新成功：", status == 200)
print("Refresh Token已轮换：", (
    status == 200
    and refreshed["refresh_token"] != result["refresh_token"]
))
