# Day34：毕业项目的人工审批与可靠写入

今日完成固定7步：笔记模型、暂停恢复、批准写入、校验与异常、隔离测试、完整交互、总结解读。Day35按原计划做毕业验收。

## 1. 待审批笔记的数据约定

文件：day34_note_contract.py。

NoteDraft是用户提供的会话编号、标题和正文。PendingNote增加系统笔记编号、快照哈希和pending状态。

会话编号只允许字母、数字、短横线和下划线。标题与正文限制长度，不接受空白或危险控制字符。extra=forbid拒绝用户额外传入approved等字段。

快照绑定note_id、session_id、title和content。正文变化时必须重新准备审批，不能继续使用旧快照。

SHA-256用于一致性检查，不是身份认证；能同时修改快照和程序的人不受此机制限制。

## 2. interrupt暂停与恢复

文件：day34_note_interrupt.py。

```text
prepare生成快照并保存checkpoint
→ approval调用interrupt
→ 程序返回待审批内容
→ 人工yes/no
→ Command(resume=decision)
→ 核对会话、笔记编号和哈希
```

恢复时审批节点从开头重新执行。因此随机编号放在prepare节点生成，文件写入放在审批后的节点，避免恢复时重复生成对象或提前执行副作用。

flow_id定位一笔审批；session_id标明所属会话。一个session可以拥有多个flow，不能把两者当成同一个编号。

## 3. 批准写入与拒绝隔离

文件：day34_note_writer.py。

写入前检查approval严格等于True，草稿与快照一致，快照哈希有效，目标路径符合规则。

文件名由系统note_id生成，标题不能决定任意路径。

写入过程：

```text
确定性Markdown
→ 同目录临时文件
→ flush + fsync
→ os.link原子发布且不覆盖
→ 清理临时文件
```

重复执行相同note_id时：相同字节复用；不同字节拒绝。拒绝分支不调用写入函数，也不创建笔记目录。

这处理了“文件已写入，但checkpoint还没确认完成”的重试窗口，不代表整个系统实现了分布式exactly-once事务。

## 4. 工具参数与异常分类

文件：day34_guarded_operations.py。

CalculationArgs采用严格类型，拒绝数字字符串、布尔值、非有限数、额外字段及绝对值超过一万亿的参数。工具执行失败时必须说明，不能让模型虚构成功结果。

笔记保存失败记录status=failed、error_type、retryable、retry_count。

| 情况 | 行为 |
|---|---|
| EBUSY、EAGAIN等临时错误 | 允许人工重试，最多2次 |
| 权限不足、磁盘满等其他OS错误 | 先排查，不直接放行重试 |
| 快照不一致、内容冲突 | 阻断，不能重复执行解决 |

retry_flow保持原笔记编号、快照和批准决定，通过update_state(as_node="approval")重新调度save，然后invoke(None)继续运行。它没有重新调用interrupt，也没有重新准备草稿。

## 5. 会话归属与失败验证

文件：day34_session_checks.py。

require_flow_session读取目标流程的最新快照并检查所属会话。checked_resume和checked_retry先做这项检查，再调用恢复函数。

9项测试覆盖跨会话、独立目录、拒绝不写、重复审批、错误哈希、重试上限、权限失败、草稿变化和不存在流程，已全部实际通过。

当前是本机可信调用环境。会话归属不是身份认证，用户仍能通过/session切换编号，不能直接作为多用户公网权限机制。

## 6. 完整学习助手

文件：day34_learning_assistant.py。

CLI协调两张图：聊天图处理交流、工具和检索；审批图处理快照、interrupt、保存和拒绝。

正常模式运行目录：

```text
day34/runtime/
├── chat.db
├── approvals.db
└── saved_notes/会话编号/笔记编号.md
```

demo使用demo_前缀文件与目录。check使用临时目录，完成后清理。

| 命令 | 功能 |
|---|---|
| /note | 输入标题正文，生成待审批提案 |
| /flows | 列出当前会话的流程 |
| /approve 编号 | 批准指定快照 |
| /reject 编号 | 拒绝 |
| /retry 编号 | 重试符合条件的保存失败 |
| /history | 最近聊天消息 |
| /status | 聊天图的状态，不是审批列表 |
| /session 编号 | 切换会话 |
| exit | 关闭数据库并退出 |

审批命令可用latest表示当前会话最近更新的流程，并非“最新待审批流程”；它可能已经完成。实际操作优先明确指定流程编号。

## 7. 消息与审批真相

创建提案时，聊天历史只记录“等待审批，尚未保存”。审批完成后，CLI根据审批图实际返回值写入保存成功、拒绝或失败结果。

这个步骤不调用LLM，让后续聊天可以读取真实执行结果。

/help、/flows、/status、/history不进入历史；/note和成功返回结果的审批命令会以HumanMessage和AIMessage记录。

两份数据库没有跨库事务。如果笔记已保存但聊天事件写入失败，应以审批数据库和实际文件为准；聊天记录不作为唯一保存凭证。

## 当前验收证据与边界

已验证：严格工具参数、真实Python计算72、Fake检索回答、重开数据库后批准/拒绝、写入幂等、冲突拦截、重试分类和会话归属。

尚未在本次调用：真实DeepSeek与真实检索整条链路。明天毕业验收会安排真实模型演示，不把Fake通过等同于真实质量合格。

笔记草稿当前由用户输入，正文留空可采用最近助手回答；没有让模型自动修改用户已经批准的内容。当前一次输入一行正文，模型只支持本项目已有的检索与计算范围。

## 复习命令

```bash
python day34/python/day34_note_contract.py --demo
python day34/python/day34_note_interrupt.py demo
python day34/python/day34_note_writer.py demo
python day34/python/day34_guarded_operations.py demo
python day34/python/day34_session_checks.py
python day34/python/day34_learning_assistant.py --check

python day34/python/day34_learning_assistant.py xiaolin-capstone-001
python day34/python/day34_learning_assistant.py --demo
```

前几步单独脚本的数据库与最终runtime数据库分别存放，避免把不同版本工作流的checkpoint混用。最终入口依赖day33与day34同目录模块，需保留这些代码文件。
