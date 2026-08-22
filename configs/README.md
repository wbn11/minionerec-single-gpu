# 配置目录

训练与评测参数必须写入 YAML，而不是依赖脚本中的隐式默认值。

计划分为：

- `data/`：数据路径、类别、历史长度和切分协议。
- `semantic_id/`：embedding、RQ-VAE 层数与 codebook 参数。
- `sft/`：模型、batch、学习率和 checkpoint 策略。
- `sasrec/`：协同教师配置。
- `grpo/`：官方奖励、rollout group 和 KL 配置。
- `evaluation/`：beam、过滤协议和指标列表。

这些目录当前只占位；每个阶段获确认后才添加实际配置。
