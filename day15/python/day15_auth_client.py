"""Day15 第三步客户端：携带Bearer Token访问GET /me。"""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser(description="Day15认证API客户端")
parser.add_argument("token", help="day15_token_cli.py签发的完整令牌")
parser.add_argument(
    "--base-url",
    default="http://127.0.0.1:8000"
)
args = parser.parse_args()

request = Request(
    f"{args.base_url}/me",
    headers={
        "Authorization": f"Bearer {args.token.strip()}"
    },
    method="GET"
)

try:
    with urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
        print("HTTP状态码：", response.status)
        print(json.dumps(data, ensure_ascii=False, indent=2))
except HTTPError as error:
    data = json.loads(error.read().decode("utf-8"))
    print("HTTP状态码：", error.code)
    print(json.dumps(data, ensure_ascii=False, indent=2))
