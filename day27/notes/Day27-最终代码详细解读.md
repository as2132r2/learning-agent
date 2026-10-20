# Day27 最终代码详细解读

本文解释 `day27_final_ops_dashboard.py` 如何组合 Day27 前五步，并通过 HTTP API 和浏览器形成完整生产运维控制台。

## 一、最终文件结构

```text
day27/
├── python/
│   ├── day27_slo_basic.py
│   ├── day27_burn_rate.py
│   ├── day27_alert_manager.py
│   ├── day27_incident_lifecycle.py
│   ├── day27_auto_rollback.py
│   └── day27_final_ops_dashboard.py
├── html/day27_final.html
├── javascript/day27_final.js
├── css/day27_final.css
├── database/day27_operations.db
└── notes/
```

最终 Python 文件不是重新实现所有逻辑，而是导入前五步的函数进行编排。这能避免同一规则出现多份实现。

---

## 二、模块导入与职责

```python
from day27_alert_manager import process_alert
from day27_auto_rollback import execute_rollback
from day27_burn_rate import evaluate_multi_window, measure_window
from day27_incident_lifecycle import create_incident
from day27_slo_basic import open_database, save_events
```

调用关系：

```text
final_ops_dashboard（编排层）
├── slo_basic（事件和基础数据库）
├── burn_rate（窗口测量与判定）
├── alert_manager（告警状态）
├── incident_lifecycle（事故状态）
└── auto_rollback（版本和回滚）
```

最终文件负责“什么时候调用谁”，各子模块负责“自己的规则如何执行”。

---

## 三、常量与路径

```python
HOST, PORT = "127.0.0.1", 8036
DAY27_DIR = Path(__file__).resolve().parents[1]
MONITOR_INTERVAL_SECONDS = 15
DASHBOARD_ALERT_KEY = f"{ALERT_KEY}:dashboard"
```

- `HOST` 只监听本机，适合教学；
- `DAY27_DIR` 从当前 Python 文件计算目录，不依赖终端当前目录；
- 后台每 15 秒产生一次新监控结果；
- dashboard 使用独立 `alert_key`，不会和前面命令行演示的告警混在一起。

HTML、JavaScript 和 CSS 都使用绝对 `Path` 定位，因此从项目根目录运行也不会找错文件。

---

## 四、为什么需要 `monitor_lock`

```python
monitor_lock = asyncio.Lock()
```

监控周期可能由两个入口同时触发：

```text
后台15秒定时器 ─┐
                 ├→ run_monitor_cycle()
用户点击立即运行 ─┘
```

若不加锁，两轮可能同时更新告警次数、创建事故或尝试回滚。`async with monitor_lock` 保证一个进程中同一时间只有一轮监控写操作。

这只是进程内锁。多实例部署时应使用数据库锁、分布式锁或单独的监控调度服务。

---

## 五、`initialize_all()`：初始化但不重置

函数执行：

```text
打开SQLite
→ 开启外键
→ 创建各步骤需要的表
→ 检查service_release_state
→ 只有完全不存在时才写入演示发布状态
```

关键点：

```python
if state is None:
    initialize_demo_release(connection, now)
```

如果每次启动都无条件初始化，就会覆盖上次回滚结果。现在重启服务器后仍能看到真实保存状态。

---

## 六、`monitor_cycle_sync()`：完整监控事务

这是最终系统最重要的同步函数，返回类型是 `dict[str, Any]`。

### 1. 生成当前时间

```python
now = datetime.now().astimezone().replace(microsecond=0)
```

使用带时区时间，并去掉微秒，方便数据库排序和页面显示。

### 2. 保存演示请求事件

```python
save_events(connection, build_six_hour_events(now))
```

`build_six_hour_events()` 生成确定性六小时事件，并把它们移动到当前滚动窗口。`save_events()` 使用稳定请求 ID，所以重复周期更新同一批教学数据，而不会无限插入重复记录。

真实系统这里应读取网关、应用指标系统或遥测平台，不应伪造事件。

### 3. 计算四个窗口

```python
measurements = {
    definition.name: measure_window(connection, definition, now)
    for definition in WINDOWS
}
```

数据类型关系：

```text
WINDOWS: tuple[WindowDefinition, ...]
measure_window(...) → WindowMeasurement
measurements: dict[str, WindowMeasurement]
```

键是 `5m`、`30m`、`1h`、`6h`，值保存总数、失败数、可用率、错误率和燃烧率。

### 4. 多窗口判定并保存快照

```python
decision = evaluate_multi_window(measurements)
snapshot_id = save_snapshot(connection, measurements)
```

`decision` 是 `BurnDecision`，主要字段：

- `level`：normal、warning 或 page；
- `reason`：判断依据；
- `fast_burn_triggered`；
- `slow_burn_triggered`。

快照让以后能够回答“当时看到的指标是什么”。

### 5. 更新告警状态

```python
alert_action = process_alert(
    connection,
    DASHBOARD_ALERT_KEY,
    SERVICE_NAME,
    decision,
    now,
)
```

参数从上到下分别是：数据库连接、告警身份、服务名、本轮判定和发生时间。

返回 `AlertAction`，说明本轮是打开、重复、抑制、恢复还是无变化，以及是否应该发送通知。

### 6. page 时创建事故

```python
if decision.level == "page":
    incident, created = create_incident(...)
```

`create_incident()` 自身具有去重逻辑。相同告警已有未关闭事故时返回旧事故，不会每 15 秒创建一条新事故。

### 7. 判断是否回滚

```python
release = load_release_state(connection)
if release.active_version != release.last_known_good_version:
    rollback = execute_rollback(...)
```

只有当前活跃版本不是可靠版本时才执行回滚。若已经运行可靠版本，就返回“无需重复回滚”的摘要。

这说明：`page` 告警可以持续存在，但危险动作只在满足前置条件时执行。

### 8. 返回可序列化结果

Dataclass 使用：

```python
asdict(alert_action)
asdict(rollback)
```

转成普通字典后才能稳定交给 JSON 响应或日志输出。

---

## 七、同步数据库和异步 HTTP 的连接

```python
async def run_monitor_cycle():
    async with monitor_lock:
        return await asyncio.to_thread(monitor_cycle_sync)
```

SQLite 和当前监控函数是同步代码。`asyncio.to_thread()` 把它放入工作线程，避免阻塞 aiohttp 事件循环。

流程：

```text
aiohttp事件循环
→ 等待asyncio.Lock
→ 把同步监控交给线程
→ 事件循环仍可处理其他HTTP连接
→ 线程完成后恢复协程
```

锁放在 `to_thread()` 外面，确保不会同时启动多个写数据库线程。

---

## 八、`read_overview_sync()`：构造页面唯一事实

该函数只读数据库，组合：

- 当前四窗口指标；
- 当前燃烧判定；
- 当前告警；
- 当前未解决事故；
- 事故时间线；
- 发布状态；
- 版本列表；
- 最近告警事件；
- 最近回滚动作。

返回结构：

```text
overview
├── generated_at
├── service
├── monitor_interval_seconds
├── decision
├── windows
├── alert
├── incident
├── incident_timeline
├── release
├── versions
├── alert_events
└── rollbacks
```

前端不自己推导事故或回滚结果，只渲染后端返回的事实，从而避免浏览器和服务器产生两套判断规则。

---

## 九、HTTP 处理函数

### `overview_handler`

```python
return web.json_response(await asyncio.to_thread(read_overview_sync))
```

读取数据库也放到线程，返回 JSON。

### `run_monitor_handler`

```python
result = await run_monitor_cycle()
return web.json_response({"ok": True, "cycle": result})
```

按钮请求执行真正的监控周期，受到同一个 `monitor_lock` 保护。

### `health_handler`

只返回服务是否存活、服务名和数据库路径，不执行昂贵模型或监控逻辑。

### `file_handler`

根据固定请求路径返回三个固定文件：

```python
{"/": HTML, "/app.js": JS, "/app.css": CSS}[request.path]
```

它不是任意文件下载接口，因此用户不能通过路径参数读取服务器上的其他文件。

---

## 十、统一错误中间件

```python
@web.middleware
async def errors(request, handler):
```

它包裹所有路由：

- HTTP 业务异常返回对应状态码；
- 未预料异常在服务端打印类型；
- 客户端只得到通用 500 信息，不泄露内部堆栈和路径。

生产环境还应把异常写入结构化日志和追踪系统。

---

## 十一、后台监控任务的生命周期

启动：

```python
app["monitor_task"] = asyncio.create_task(
    monitor_worker(app), name="day27-slo-monitor"
)
```

循环：

```text
执行监控
→ 输出简短日志
→ sleep 15秒
→ 再执行
```

关闭：

```python
task.cancel()
with contextlib.suppress(asyncio.CancelledError):
    await task
```

仅调用 `cancel()` 不够，还要 `await task` 回收任务。`CancelledError` 是预期关闭信号，因此在清理阶段抑制。

普通监控异常只记录后继续下一轮；取消异常必须重新抛出，否则服务器无法正常停止后台任务。

---

## 十二、路由注册

```text
/                 → HTML
/app.js           → JavaScript
/app.css          → CSS
/health           → 健康检查JSON
/api/overview     → 全景状态JSON
/api/monitor/run  → 手动监控POST
```

`POST` 用于立即运行，是因为它会更新事件、快照、告警和事故状态；不能把有副作用的操作设计成普通 `GET`。

---

## 十三、前端 JavaScript 数据流

页面启动：

```text
DOMContentLoaded
→ loadOverview()
→ fetch('/api/overview')
→ response.json()
→ renderOverview(data)
→ setInterval每5秒重复
```

点击按钮：

```text
click
→ POST /api/monitor/run
→ 等待监控周期完成
→ 再GET /api/overview
→ 页面显示新状态
```

15 秒监控间隔和 5 秒页面刷新并不冲突：页面可能连续三次读到相同事实，但能在新一轮完成后较快显示结果。

前端渲染前对文本进行转义，避免数据库或返回值中的特殊文本直接变成 HTML。

---

## 十四、完整端到端流程图

```text
后台定时器/立即运行按钮
          ↓
run_monitor_cycle()
          ↓ asyncio.Lock
asyncio.to_thread(monitor_cycle_sync)
          ↓
请求事件 → 四个滚动窗口 → BurnDecision
          ↓
      保存快照
          ↓
process_alert() → alert_states + alert_events
          ↓ page
create_incident() → incidents + incident_timeline
          ↓ 当前版本不等于LKG
execute_rollback() → release state + rollback_actions
          ↓
      SQLite提交

浏览器每5秒
          ↓
GET /api/overview
          ↓
read_overview_sync()
          ↓
SQLite各状态表和事件表
          ↓
统一JSON → DOM渲染
```

---

## 十五、关键数据类型

| 变量 | 类型 | 含义 |
|---|---|---|
| `WINDOWS` | `tuple[WindowDefinition, ...]` | 监控窗口定义 |
| `measurements` | `dict[str, WindowMeasurement]` | 各窗口测量结果 |
| `decision` | `BurnDecision` | 多窗口判断 |
| `alert_action` | `AlertAction` | 本轮告警动作 |
| `incident` | `Incident` | 当前事故对象 |
| `release` | `ReleaseState` | 当前发布状态 |
| `monitor_lock` | `asyncio.Lock` | 防止监控周期并发写入 |
| `monitor_task` | `asyncio.Task[None]` | 后台定时任务 |
| `overview` | `dict[str, Any]` | 提供给浏览器的全景数据 |

---

## 十六、当前教学版本的边界

1. 请求事件是确定性模拟数据，不是真实遥测。
2. SQLite 适合本地学习，不适合大量多实例指标写入。
3. `asyncio.Lock` 只能协调单进程。
4. 告警通知只记录动作，没有连接短信、电话或值班平台。
5. 回滚是数据库状态模拟，没有调用 Kubernetes 或发布平台。
6. 没有身份认证，不能直接暴露到公网。
7. 指标、日志和分布式追踪尚未通过 trace ID 关联。

这些不是代码错误，而是教学边界。Day27 的目标是先掌握生产运维闭环的结构。

---

## 十七、运行与验证

```bash
python day27/python/day27_final_ops_dashboard.py
```

访问：

```text
http://127.0.0.1:8036
```

可验证：

1. 页面显示四个窗口；
2. 快速燃烧产生 page 告警；
3. 重复刷新不会创建无限事故；
4. 已处于可靠版本时不会重复回滚；
5. 点击“立即运行”后页面状态更新；
6. `Ctrl+C` 后后台任务被正确取消。

最终要记住：

```text
监控不是一张图表，
而是从用户结果到安全恢复、全程有状态且可审计的控制闭环。
```
