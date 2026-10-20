# LangChain Day29：标准化遥测与 OpenTelemetry 思想

## 一、今天的主线

Day28 自己实现了日志与 Trace；Day29 进一步建立跨服务可互操作的标准化遥测：

```text
W3C traceparent → 标准Span语义 → 一致采样
→ 有界批量导出 → 隐私治理与保留期 → 可视化查询
```

## 二、固定七步回顾

### 1. W3C `traceparent`

文件：`day29_traceparent_basic.py`

```text
version-trace_id-parent_id-trace_flags
00-32位十六进制-16位十六进制-2位十六进制
```

跨服务时保持 trace_id，每一跳创建新 span_id，flags 最低位表示 sampled。全零 ID、长度错误、大写十六进制和不支持的版本不能继续使用；非法外部值应丢弃并创建新 Trace。

### 2. Span Kind、Status 与语义属性

文件：`day29_span_semantics.py`

| Kind | 含义 |
|---|---|
| SERVER | 接收同步远程请求 |
| CLIENT | 发起同步远程请求 |
| PRODUCER | 发布异步工作 |
| CONSUMER | 处理异步工作 |
| INTERNAL | 当前进程内部操作 |

HTTP 稳定字段使用 `http.request.method`、`http.response.status_code`、`server.address`、`url.full` 等名称。正常 HTTP Span 通常保持 `UNSET`，错误设置 `ERROR` 与低基数 `error.type`。RAG 字段使用课程自己的 `rag.*` 命名空间。

### 3. Head Sampling 与 Parent-Based Sampling

文件：`day29_sampling_strategies.py`

实现 AlwaysOn、AlwaysOff、TraceIdRatioBased 和 ParentBased。比例采样必须根据 trace_id 稳定判断，不能让不同服务重新随机。存在父上下文时继承 flags；根请求才使用根采样器。

Head Sampling 在请求开始时不知道未来是否失败。若必须保留错误或慢请求，需要错误日志、指标、更高入口采样率或 Tail Sampling。

### 4. 批量导出、背压与关闭

文件：`day29_batch_exporter.py`

```text
业务产生Span → 非阻塞有界队列 → 后台Worker
→ 按数量或时间组批 → 遥测后端
```

队列满时本练习丢弃最新 Span 并增加 dropped 指标，优先保护业务。关闭时停止接收、放入哨兵并等待 `queue.join()`，冲刷已接受数据。

### 5. 脱敏、限制与保留期

文件：`day29_telemetry_governance.py`

- 敏感键整字段遮盖；
- URL 敏感查询参数遮盖；
- 邮箱和 Bearer 模式脱敏；
- user.id 假名化；
- 字符串、列表、属性、事件数量限制；
- 30 天保留期清理。

实际发现：宽泛匹配 `token` 会误伤 `input_tokens` 和 `output_tokens`，说明脱敏规则必须同时测试漏报与误报。

### 6. 最终标准化遥测 RAG 服务

文件：`day29_final_telemetry_service.py` 及 HTML、CSS、JavaScript。

浏览器产生标准 traceparent，服务端执行 Parent-Based 采样，创建 SERVER/INTERNAL/CLIENT Span，治理后通过有界队列批量写入 SQLite 和 JSONL，再由页面查询。

### 7. 笔记与最终代码解读

当前文档总结概念；`Day29-最终代码详细解读.md` 解释最终实现。

## 三、标准化的价值

```text
W3C traceparent负责跨服务传播
Span Kind描述通信关系
语义属性统一字段含义
Exporter适配不同后端
```

标准不是为了字段整齐，而是为了让不同语言、框架和平台互操作。

## 四、采样、记录和业务执行是三件事

```text
业务执行：请求是否完成
记录：进程是否创建Span数据
采样导出：是否把完整Trace送到后端
```

`sampled=0` 不会拒绝业务，也不能用于权限判断。

## 五、遥测系统的保护原则

1. 使用有界队列，不能无限占用内存。
2. 后端变慢时优先保护业务。
3. dropped 与 export_failures 必须可观测。
4. 关闭时尽量冲刷已接受数据。
5. 敏感数据必须在入队前治理。
6. 设置保留期，不能默认永久保存。
7. 排除健康检查、静态文件和内部轮询噪声。

## 六、最终数据流

```text
浏览器traceparent
→ 校验 + ParentBased采样
→ SERVER根Span
→ INTERNAL检索
→ CLIENT模型调用
→ INTERNAL回答组织
→ 脱敏与数量限制
→ 有界队列 + 批量Exporter
├→ SQLite
└→ JSONL
→ Web控制台
```

## 七、两个真实陷阱

### 误脱敏

任意 `token` 子串会错误遮盖 Token 用量。应匹配完整敏感键段，并建立正反测试样本。

### 自观测噪声

统一中间件也会追踪控制台轮询和静态资源。生产环境需要排除规则或独立采样率，否则观察工具本身会制造大量 Trace。

## 八、复习检查题

1. traceparent 四部分是什么？
2. CLIENT 与 PRODUCER 有什么区别？
3. 为什么 HTTP 200 常保持 UNSET？
4. 为什么比例采样必须按 trace_id 稳定决定？
5. Head Sampling 为什么可能漏掉错误 Trace？
6. `put_nowait()` 如何保护业务？
7. `queue.join()` 为什么依赖 `task_done()`？
8. 假名化为什么不等于匿名化？
9. 为什么脱敏必须发生在入队前？
10. 控制台为什么也可能制造观测噪声？

## 九、Day30 预告

下一天建议进入生产化收尾：配置分层、密钥管理、数据库迁移、健康检查、备份恢复、发布检查清单和完整项目验收。
