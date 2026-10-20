"""Day15 第五步：在用户隔离API中增加管理员授权接口。"""

from aiohttp import web

import day15_user_task_api as base


def require_role(request: web.Request, required_role: str) -> web.Response | None:
    """角色不足时返回403；通过时返回None。"""

    actual_role = request["identity"].get("role", "user")

    if actual_role != required_role:
        return web.json_response(
            {
                "error": "forbidden",
                "message": f"该接口需要{required_role}角色。"
            },
            status=403
        )

    return None


async def admin_list_tasks_handler(request: web.Request) -> web.Response:
    """管理员可以审计所有用户任务。"""

    rejection = require_role(request, "admin")
    if rejection is not None:
        return rejection

    cursor = await base.db().execute(
        """
        SELECT task_id, user_id, question, status, answer, error,
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
        "required_role": "admin",
        "count": len(rows),
        "tasks": [dict(row) for row in rows]
    })


base.app.router.add_get("/admin/tasks", admin_list_tasks_handler)


if __name__ == "__main__":
    print("Day15角色授权API已启动。")
    print("普通接口：GET/POST /tasks")
    print("管理员接口：GET /admin/tasks")
    web.run_app(base.app, host="127.0.0.1", port=8000)


# authentication：token是否有效、请求者是谁
# authorization：identity.role是否允许执行当前操作
