# LangGraph 学习资料

LangGraph 使用图结构组织大语言模型应用。

一个图通常包含节点、边和状态：

- 节点负责读取状态并返回状态更新。
- 边负责连接节点并决定流程方向。
- 条件边可以根据状态选择不同分支。
- Checkpointer 可以按 thread_id 保存工作流状态。

当流程需要人工审批时，可以使用 interrupt 暂停，再使用 Command 恢复。
