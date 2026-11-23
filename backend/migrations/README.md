# 数据表初始化

本版通过 `src.models.base.Base.metadata.create_all` 创建首次运行所需的六张表。模型在 `src/models/workflow.py`。`observations` 仍以 JSON `payload` 存每条版本，`source_key` 由曝光编号、修订号和各测光字段组成的规范内容派生；同一编号的不同修订各占一行，收到时间不参与身份，因此原样重试不会新增行。`python -m scripts.seed_demo` 按观测内容键补全演示样本，保留现有分析及签核。数据库端口只发布到本机。
