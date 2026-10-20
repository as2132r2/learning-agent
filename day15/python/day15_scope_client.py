"""Day15 第六步客户端：使用Scope令牌访问审计接口。"""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser(description="Day15 Scope接口客户端")
parser.add_argument("token")
args = parser.parse_args()

request = Request(
    "http://127.0.0.1:8000/audit/tasks",
    headers={"Authorization": f"Bearer {args.token.strip()}"},
    method="GET"
)

try:
    with urlopen(request, timeout=10) as response:
        status = response.status
        body = json.loads(response.read().decode("utf-8"))
except HTTPError as error:
    status = error.code
    body = json.loads(error.read().decode("utf-8"))

print("HTTP状态码：", status)
print(json.dumps(body, ensure_ascii=False, indent=2))
