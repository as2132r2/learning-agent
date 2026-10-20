# LangChain / LangGraph Day 4 学习笔记

## 1. 今天的主题

Day 4 学习的是 Agent 安全与 Human-in-the-loop（人在回路中）。

核心目标不是让 Agent “更自主”，而是让它在执行有副作用的操作前受到约束：

```text
模型提出操作
→ 程序判断风险
→ 高风险操作暂停
→ 人工批准、拒绝或修改
→ Python 强制校验
→ 执行工具
→ 保存审计记录
→ 防止重复执行
```

---

## 2. 今天完成的代码

### 2.1 手动审批

文件：`day4_manual_approval.py`

学习内容：

- 模型的 `tool_call` 只是操作提案。
- Python 程序决定是否真正调用工具。
- 只读工具可以自动执行。
- 有副作用的工具必须询问用户。

关键区别：

```text
模型请求 transfer_money
≠
transfer_money 已经执行
```

只有程序运行下面的调用以后，工具才真正执行：

```python
tool.invoke(tool_args)
```

---

### 2.2 interrupt 基础

文件：`day4_interrupt_basic.py`

学习内容：

- `interrupt()` 暂停 LangGraph 工作流。
- 暂停时返回审批问题和业务参数。
- `Command(resume=...)` 把人工决定送回工作流。

基本结构：

```python
decision = interrupt({
    "question": "是否批准？",
    "amount": 100
})
```

恢复：

```python
graph.invoke(
    Command(resume="yes"),
    config=config
)
```

重要规律：包含 `interrupt()` 的节点恢复时，会从节点开头重新运行。

因此，有副作用的代码必须放在 `interrupt()` 后面，或者放进单独的执行节点。

错误顺序：

```text
执行转账 → interrupt
```

正确顺序：

```text
interrupt → 收到批准 → 执行转账
```

---

### 2.3 跨进程暂停与恢复

文件：`day4_interrupt_persistent.py`

学习内容：

- `InMemorySaver` 只能在当前程序进程中保存状态。
- `SqliteSaver` 可以把暂停状态写入数据库。
- 程序关闭后，仍然可以使用相同 `thread_id` 恢复。

启动和恢复必须使用同一个流程编号：

```bash
python day04/python/day4_interrupt_persistent.py start transfer-001
python day04/python/day4_interrupt_persistent.py resume transfer-001 yes
```

`thread_id` 的作用：

```text
thread_id
→ 找到对应的工作流检查点
→ 知道暂停在哪个节点
→ 恢复相应流程
```

如果编号不同，就找不到原来暂停的工作流。

---

### 2.4 LLM Agent 与 interrupt 集成

文件：`day4_agent_interrupt.py`

学习内容：

- 用户使用自然语言提出请求。
- 模型判断是否需要工具。
- Python 根据工具名称判断风险等级。
- 只读工具直接执行。
- 高风险工具进入人工审批。

示例：

```text
查询余额
→ check_balance
→ 只读工具
→ 自动执行
```

```text
请求转账
→ transfer_money
→ 高风险工具
→ interrupt 暂停
```

风险分类集合：

```python
approval_required_tools = {
    "transfer_money"
}
```

这个判断由 Python 完成，不能只依赖模型自行决定。

---

### 2.5 批准、拒绝和修改

文件：`day4_agent_edit.py`

人工审批不再只有 yes/no，而是支持三种决定：

```text
yes  → 批准原参数
no   → 拒绝执行
edit → 修改参数后批准
```

恢复数据使用结构化字典：

```python
{
    "action": "edit",
    "new_args": {
        "to_account": "account-003",
        "amount": 500.0
    }
}
```

结构化数据比只传一个字符串更适合复杂审批流程。

---

### 2.6 确定性的业务校验

文件：`day4_agent_validation.py`

模型可能主动拒绝明显错误，例如负数转账，但不能把模型当作最终安全规则。

最终规则必须由 Python 强制执行：

```text
账户必须符合 account-数字 格式
收款账户不能等于付款账户
金额必须是数字
金额必须大于 0
金额不能超过单笔限额
```

两层防线：

```text
第一层：模型的语义判断
第二层：Python 的确定性校验
```

即使模型提出了错误工具调用，Python 校验也必须阻止执行。

审批也不能绕过校验：

```text
人工批准
≠
参数一定合法
```

正确顺序是：

```text
人工批准 → 参数校验 → 执行
```

---

### 2.7 审计日志

文件：`day4_agent_audit.py`

审计日志回答五个问题：

1. 谁发起了什么请求？
2. 模型提出了什么工具参数？
3. 人工做了什么决定？
4. 最终使用了什么参数？
5. 工具是否成功执行？

主要事件：

```text
workflow_started
approval_requested
approved
rejected
edited_and_approved
auto_approved
validation_failed
execution_succeeded
execution_failed
```

LangGraph 检查点和审计日志作用不同：

```text
检查点：让工作流以后可以继续运行
审计日志：让人以后可以追踪发生过什么
```

它们可以放在同一个 SQLite 文件里，但应使用不同的数据表。

#### 为什么审批请求需要防重复记录？

`interrupt()` 恢复后节点会从头执行，所以 `interrupt()` 前面的日志代码也会再次运行。

我们使用 `tool_call_id` 检查是否已经记录过：

```text
相同 thread_id
+ 相同 tool_call_id
+ approval_requested 已存在
→ 不重复写入
```

---

## 3. ToolMessage 必须包含完整事实

在测试参数修改时，模型一度报告“执行参数和原始参数不一致”。

原因是模型只看到了：

```text
原始提案：account-002，6000元
执行结果：account-003，500元
```

模型无法自动读取审计数据库，也不知道中间发生了人工修改。

修复方法是让 `ToolMessage` 明确包含：

```text
用户修改并批准了参数
原始提案参数
最终授权参数
真实执行结果
```

因此要记住：

```text
数据库中保存了事实
≠
模型自动知道这些事实
```

需要模型理解的信息，必须通过消息上下文传给模型。

---

## 4. 幂等保护

最终文件：`day4_safe_agent.py`

幂等的含义：同一个操作重复请求多次，最终只产生一次业务效果。

危险情况：

```text
第一次转账已经成功
→ 网络超时
→ 程序误以为失败并重试
→ 用户被重复扣款
```

我们使用唯一的 `tool_call_id` 保存执行凭证：

```text
执行前查询 tool_call_id
├─ 不存在：执行工具并保存结果
└─ 已存在：不再执行，复用原结果
```

相关审计事件：

```text
duplicate_execution_prevented
execution_result_reused
```

学习版本只能保护本地程序中的重复执行。

真实支付接口通常需要支持：

```text
idempotency_key = tool_call_id
```

最终幂等保障应由真正产生副作用的外部系统提供。

---

## 5. Day 4 最终完整数据流

```text
用户自然语言请求
        ↓
HumanMessage
        ↓
assistant 节点调用 LLM
        ↓
AIMessage
        ↓
是否包含 tool_calls？
   ├─ 否 → 直接回答 → END
   └─ 是
        ↓
execute_tools 节点
        ↓
工具是否属于高风险集合？
   ├─ 否 → 自动批准
   └─ 是 → 写入审批审计 → interrupt
                              ↓
                    SQLite 保存检查点
                              ↓
                    Command(resume=决定)
                              ↓
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
          approve           reject             edit
            ↓                 ↓                 ↓
        使用原参数        不执行工具        使用新参数
            └─────────────────┬─────────────────┘
                              ↓
                       Python 业务校验
                       ├─ 失败 → 不执行
                       └─ 通过
                              ↓
                     查询执行凭证 tool_call_id
                       ├─ 已执行 → 复用结果
                       └─ 未执行 → 调用工具
                                      ↓
                                保存执行凭证
                              ↓
                         写入审计日志
                              ↓
                           ToolMessage
                              ↓
                       assistant 再次调用 LLM
                              ↓
                          最终自然语言回答
```

---

## 6. 三种数据不要混淆

### 消息状态

典型数据：

```python
HumanMessage
AIMessage
ToolMessage
```

用途：给模型提供当前对话和工具结果。

### 工作流检查点

典型数据：

```text
当前节点
MessagesState
interrupt 暂停信息
下一步节点
```

用途：关闭程序以后还能恢复流程。

### 审计数据

典型数据：

```text
原始参数
最终参数
审批决定
校验结果
执行结果
发生时间
```

用途：安全追踪、排错和责任确认。

---

## 7. 今天最重要的结论

1. 模型只能提出工具调用，程序控制真正执行。
2. 高风险工具必须经过人工审批。
3. `interrupt()` 必须配合 checkpointer 和固定 `thread_id`。
4. 恢复后节点会从开头重跑，副作用必须放在暂停之后。
5. 人工可以批准、拒绝或修改参数。
6. 人工批准不能代替 Python 业务校验。
7. 审计日志和工作流检查点解决的是不同问题。
8. 模型不会自动读取数据库，关键事实要通过消息传递。
9. 有副作用的工具必须考虑幂等，避免重复执行。

---

## 8. 常用测试命令

启动最终安全 Agent：

```bash
python day04/python/day4_safe_agent.py start safe-001
```

批准：

```bash
python day04/python/day4_safe_agent.py resume safe-001 yes
```

拒绝：

```bash
python day04/python/day4_safe_agent.py resume safe-001 no
```

修改后批准：

```bash
python day04/python/day4_safe_agent.py resume safe-001 edit
```

查看审计日志：

```bash
python day04/python/day4_safe_agent.py logs safe-001
```

每次独立测试建议使用新的 `thread_id`：

```text
safe-001
safe-002
safe-003
```

---

## 9. 明天 Day 5 预览

Day 5 建议开始学习 RAG（检索增强生成）：让模型根据自己的文档回答问题。

预计顺序：

1. 什么是 Document。
2. 文本切分器如何产生 chunks。
3. Embedding 如何把文本转换为向量。
4. 向量数据库如何保存和检索文本块。
5. Retriever 如何找到相关资料。
6. 把检索结果和问题一起交给模型。
7. 返回带来源的回答。
8. 最后组合成一个带记忆、工具和 RAG 的小型 Agent。

Day 5 的核心数据流：

```text
自己的文档
→ 文本切块
→ Embedding
→ 向量库
→ 用户提问
→ 检索相关文本块
→ 文本块 + 问题交给模型
→ 基于资料生成回答
```
