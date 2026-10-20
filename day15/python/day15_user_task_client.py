"""Day15 第四步客户端：使用某个用户令牌管理自己的任务。"""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser(description="Day15用户任务客户端")
parser.add_argument("token")
parser.add_argument("command", choices=["create", "list", "get"])
parser.add_argument("value", nargs="?", help="create的问题或get的task_id")
args = parser.parse_args()

headers = {
    "Authorization": f"Bearer {args.token.strip()}",
    "Content-Type": "application/json; charset=utf-8"
}
base_url = "http://127.0.0.1:8000"

if args.command == "create":
    if not args.value:
        raise SystemExit("create命令需要提供问题。")
    request = Request(
        f"{base_url}/tasks",
        data=json.dumps(
            {"question": args.value}, ensure_ascii=False
        ).encode("utf-8"),
        headers=headers,
        method="POST"
    )
elif args.command == "list":
    request = Request(f"{base_url}/tasks", headers=headers, method="GET")
else:
    if not args.value:
        raise SystemExit("get命令需要提供task_id。")
    request = Request(
        f"{base_url}/tasks/{args.value}",
        headers=headers,
        method="GET"
    )

try:
    with urlopen(request, timeout=15) as response:
        status = response.status
        data = json.loads(response.read().decode("utf-8"))
except HTTPError as error:
    status = error.code
    data = json.loads(error.read().decode("utf-8"))

print("HTTP状态码：", status)
print(json.dumps(data, ensure_ascii=False, indent=2))
