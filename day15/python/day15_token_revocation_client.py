"""Day15 第七步客户端：撤销当前令牌并验证它立即失效。"""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser(description="Day15令牌撤销客户端")
parser.add_argument("token")
args = parser.parse_args()

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"Authorization": f"Bearer {args.token.strip()}"}


def send(request: Request) -> tuple[int, dict]:
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


print("第一步：撤销当前令牌")
status, body = send(Request(
    f"{BASE_URL}/auth/revoke",
    data=b"",
    headers=HEADERS,
    method="POST"
))
print("HTTP状态码：", status)
print(json.dumps(body, ensure_ascii=False, indent=2))

print("\n第二步：使用同一个令牌再次查询任务")
status, body = send(Request(
    f"{BASE_URL}/tasks",
    headers=HEADERS,
    method="GET"
))
print("HTTP状态码：", status)
print(json.dumps(body, ensure_ascii=False, indent=2))
