# 贡献

欢迎通过 Issue 提交问题或通过 Pull Request 提交改进。

1. 安装 `pip install -e .`。
2. 使用临时目录和 Fake 模型编写可离线复现的回归测试。
3. 运行 `python -m unittest discover -s tests -v` 与 `python -m learning_agent --check`。
4. 明确真实模型、Fake 和故障注入的边界，不夸大测试覆盖。

不要提交 `.env`、运行数据库、真实用户对话、上传文档、模型缓存或生成的私人笔记。
