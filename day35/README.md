# 编程学习助手：毕业项目运行说明

固定入口：day35/python/day35_final_assistant.py。

它复用Day34完整交互程序；验收代码和报告集中在Day35。请保留项目的day21、day22、day33、day34、day35模块关系，单独复制入口文件不能独立运行。

## 1. 已实现能力

- 普通聊天、同会话姓名与历史计算结果追问。
- add、multiply真实工具执行与严格参数检查。
- 本地课程资料的混合检索、RRF、模型精排和来源引用。
- 学习笔记提案、人工批准/拒绝、受控文件写入。
- SQLite聊天及审批checkpoint，支持重新打开后恢复。
- 临时写入故障的有限人工重试、内容冲突拒绝覆盖。

## 2. 运行环境

当前验证环境是Ubuntu、Python 3.12和项目.venv。先进入项目根目录并激活：

```bash
cd /path/to/learning-agent
source .venv/bin/activate
```

已有环境可以直接运行，不必重新安装。需要建立新环境时，在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r day35/requirements.txt
```

requirements记录当前已安装且完成验收的直接依赖版本，不是包含全部传递依赖与哈希的完整锁文件。新环境仍应重跑验收。

## 3. 配置

模板位于day35/config/.env.example，只有字段名和普通配置，没有真实密钥。

真实模式读取项目根目录.env中的：

```dotenv
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=填写你自己的密钥
```

已有根.env时只补充缺失字段，不要用模板覆盖。实际模型以运行环境配置为准，模型名与API地址需要与你的供应商账号匹配。load_dotenv默认不覆盖已经设置的系统环境变量。

模板不会自动加载；不需要把密钥写入Python代码或聊天消息。

## 4. 三种启动方式

正常模式，会调用真实模型：

```bash
python day35/python/day35_final_assistant.py xiaolin-capstone-001
```

离线固定样例，无模型API调用：

```bash
python day35/python/day35_final_assistant.py --demo
```

交互接线检查，使用临时数据库与Fake：

```bash
python day35/python/day35_final_assistant.py --check
```

demo的计算样例固定为(20+4)*3，知识样例固定为RAG，不能用它评估通用模型能力。

## 5. 建议演示顺序

输入：

```text
我叫小林。
请计算(20+4)*3。
刚才的结果是多少？
根据课程资料解释RAG。
/note
```

随后输入笔记标题和正文。正文留空采用最近助手回答，批准前请核对显示的具体内容。

使用程序显示的完整流程编号：

```text
/approve note-flow-实际编号
```

拒绝：

```text
/reject note-flow-实际编号
```

同一流程不能批准两次。只有符合条件的临时写入故障才能执行/retry。

## 6. 本地命令

| 命令 | 说明 |
|---|---|
| /help | 帮助 |
| /history | 最近12条聊天消息和工具消息 |
| /status | 当前聊天图的消息数、路线、待执行节点 |
| /session session-b | 切换会话，不删除旧记录 |
| /note | 输入笔记标题和正文 |
| /flows | 当前会话的审批流程及状态 |
| /approve 编号 | 批准指定提案 |
| /reject 编号 | 拒绝指定提案 |
| /retry 编号 | 有限重试原已批准写入 |
| exit | 关闭数据库并退出 |

审批命令支持latest作为当前会话最近更新流程的快捷选择。它不保证该流程仍待审批，实际验收推荐指定完整编号。

## 7. 数据目录

最终入口沿用Day34数据，不会把数据库改放Day35：

```text
day34/runtime/
├── chat.db
├── approvals.db
├── saved_notes/会话编号/笔记编号.md
├── demo_chat.db
├── demo_approvals.db
└── demo_saved_notes/会话编号/笔记编号.md
```

真实验收另用day35/runtime/live-运行编号下的独立数据库。Fake验收使用临时目录，清理后证据保留在JSON报告。

笔记文件和会话数据库都可能包含个人数据，分享代码时不应一并提交运行目录。

## 8. 课程索引

真实检索沿用：

- day21/database/day21_vector.db：向量索引；
- day22/database/day22_fts.db：关键词索引；
- .fastembed_cache：Embedding缓存。

已有索引可直接使用。需要重新初始化时：

```bash
python day21/python/day21_vector_index.py sync
python day22/python/day22_fts_basic.py "什么是RAG？"
```

sync会同步课程文档与已校验上传，可能修改现有索引。首次加载Embedding可能下载模型；索引内容、模型或阈值变化后应重新验证检索结果。

RAG精排与回答会把命中资料正文发送到配置的模型API。

## 9. 验收命令与报告

```bash
# 生成固定计划，不执行验收
python day35/python/day35_acceptance_plan.py

# G01-G12，真实图与存储，Fake模型和检索
python day35/python/day35_fake_acceptance.py

# G01/G02/G03/G04/G08，真实模型与课程索引，会产生费用
python day35/python/day35_live_acceptance.py

# 核查报告，补充两个独立子进程的恢复测试
python day35/python/day35_reliability_audit.py
```

结果位于day35/reports。passed、failed、blocked、not_run分别表示通过、不满足判据、环境阻断、未执行。blocked不能记为通过。

重新运行真实验收会生成新的独立运行目录并更新最新报告。随后应重跑reliability_audit更新源报告哈希。

## 10. 常见问题

配置缺失：核对根.env和系统环境变量，勿打印密钥。

网络或API不可用：真实验收记blocked；检查连接、账号和供应商配置后重跑，不切换Fake冒充通过。

资料不足：检查实际课程内容、索引和精排结果，不应强行让模型编造答案。

聊天提示有未完成节点：/status查看，保留原记录排查或切换新会话。当前未实现通用聊天失败任务自动恢复。

笔记未保存：/flows核对pending、rejected、failed或saved。聊天和审批数据库没有跨库事务，应结合审批状态与实际文件判断。

文件冲突：同note_id不同内容拒绝覆盖，重新发起提案；不要通过修改哈希绕过审批。

## 11. 能力范围

这是本机学习项目。关键词路由、固定相关性阈值、引用编号检查、会话归属约束都有边界：不提供公网用户认证，不进行完整事实审计，也不保证任意数学表达式都能计算。

验收通过代表固定场景在当前环境得到验证，不代表所有可能输入都正确。
