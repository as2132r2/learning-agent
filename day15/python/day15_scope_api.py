"""Day15 第六步：使用Scope执行细粒度接口授权。"""

from aiohttp import web

import day15_role_api as role_api
import day15_user_task_api as base


def require_scope(request: web.Request, required_scope: str) -> web.Response | None:
    """检查签名载荷中的scopes列表。"""

    raw_scopes = request["identity"].get("scopes", [])
    scopes = raw_scopes if isinstance(raw_scopes, list) else []

    if required_scope not in scopes:
        return web.json_response(
            {
                "error": "insufficient_scope",
                "message": f"该接口需要权限：{required_scope}",
                "required_scope": required_scope
            },
            status=403,
            headers={
                "WWW-Authenticate": (
                    f'Bearer error="insufficient_scope", '
                    f'scope="{required_scope}"'
                )
            }
        )

    return None


async def audit_tasks_handler(request: web.Request) -> web.Response:
    """拥有task:read:any的身份可以审计全部任务。"""

    rejection = require_scope(request, "task:read:any")
    if rejection is not None:
        return rejection

    cursor = await base.db().execute(
        """
        SELECT task_id, user_id, question, status,
               created_at, started_at, finished_at
        FROM user_tasks
        ORDER BY created_at DESC
        LIMIT 200
        """
    )
    rows = await cursor.fetchall()
    await cursor.close()

    return web.json_response({
        "requested_by": request["identity"]["sub"],
        "used_scope": "task:read:any",
        "count": len(rows),
        "tasks": [dict(row) for row in rows]
    })


base.app.router.add_get("/audit/tasks", audit_tasks_handler)


if __name__ == "__main__":
    print("Day15 Scope授权API已启动。")
    print("Scope审计接口：GET /audit/tasks")
    print("所需权限：task:read:any")
    web.run_app(base.app, host="127.0.0.1", port=8000)


# Token有效 → Authentication通过
# scopes包含task:read:any → Authorization通过
# role名称不再是该接口的判断条件
