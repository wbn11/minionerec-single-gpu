# 官方多卡与单 A6000 差异

本文件只记录允许的系统实现差异。未列出的算法变更默认不允许进入复现基线。

## 必须保持一致

- 数据切分与最大历史长度；
- item 与 SID 映射；
- SFT 输入输出语义；
- 官方奖励公式；
- GRPO 的组内 advantage 语义；
- Trie 合法目录；
- HR/NDCG 定义及评测过滤协议。

## 允许的单卡替代

| 官方多卡实现 | 单 A6000 复现 |
|---|---|
| 多进程 `torchrun` | 单 Python 进程 |
| 多 GPU 数据并行 | 梯度累积保持有效 batch |
| ZeRO-2 | 单卡不启用 ZeRO |
| 默认激活保存 | Gradient checkpointing |
| 低效或关闭的 attention backend | PyTorch SDPA |
| 多份完整模型常驻 | 共享 SFT base，并用 adapter 表示可训练 policy |
| 多 GPU 分片评测 | 单卡小 batch、顺序写入预测 |

## 需要实测后才能决定

- SFT 的精确 micro-batch；
- GRPO rollout group 16 是否低于 44GB 峰值；
- 是否需要 CPU offload；
- 1.5B 是否作为复现后的规模验证。

这些项目必须在对应步骤先解释资源模型，再由项目所有者确认，不能在本阶段提前决定。

## 不属于复现基线

- CGRF 或其他新奖励；
- 修改 SID 层数或 codebook 大小；
- 更改官方 target 定义；
- 使用测试集选择超参数；
- 用随机商品替换非法生成；
- 将后续 GPR、HEPO 或 TS-Rec 当作原始 MiniOneRec 主链路。
