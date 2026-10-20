"""Day15 第五步客户端：使用角色令牌访问管理员任务接口。"""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser(description="Day15管理员接口客户端")
parser.add_argument("token")
args = parser.parse_args()

request = Request(
    "http://127.0.0.1:8000/admin/tasks",
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
