# MiniOneRec：单张 RTX A6000 复现

本项目在单张 NVIDIA RTX A6000 48GB 上复现 MiniOneRec 的主流程。代码参考固定 upstream commit `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`，当前使用 Amazon18 `Industrial_and_Scientific`、Qwen3-Embedding-4B 和 Qwen2.5-1.5B-Instruct。

本地 `E:\minionerec` 是代码与实验产物的完整副本；服务器 `/home/user/wbn/minionerec` 只负责运行 GPU 实验。

## 主流程

```text
Amazon18 数据处理
  → 标题+描述商品 embedding
  → RQ-VAE 三层 Semantic ID
  → SFT 数据转换与词表扩展
  → Qwen2.5-1.5B 全参数 BF16 SFT
  → ranking GRPO
  → 约束 beam search 与 HR/NDCG 评测
```

当前已完成整个流程。GRPO 在 4,533 条测试样本上取得 HR@3 `0.093316`、NDCG@3 `0.079940`、HR@10 `0.139643`、NDCG@10 `0.096283`，所有已统计的 Top-K 指标均高于同一实验的 SFT 模型。

## 代码入口

| 顺序 | 入口 | 作用 |
| ---: | --- | --- |
| 1 | `scripts/prepare_amazon.py` | Amazon18 清洗、K-core、ID 映射和时间切分 |
| 2 | `scripts/generate_data_statistics.py` | 生成数据规模与历史长度统计 |
| 3 | `scripts/generate_item_embeddings.py` | 生成商品文本 embedding |
| 4 | `scripts/train_rqvae.py` | 训练 RQ-VAE |
| 5 | `scripts/generate_semantic_ids.py` | 生成并消解碰撞后的三层 SID |
| 6 | `scripts/convert_dataset.py` | 转换 SFT/GRPO 使用的 CSV |
| 7 | `scripts/train_sft.py` | 全参数 BF16 SFT |
| 8 | `scripts/evaluate_sft.py` | 统一执行 SFT 或 GRPO 的 Top-K 评测 |
| 9 | `scripts/train_grpo.py` | 全参数 ranking GRPO |

命令行入口只解析参数；数据、量化、训练、生成、奖励和评测实现位于 `src/minionerec/`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `scripts/` | 可直接在服务器执行的主流程入口 |
| `src/minionerec/` | 可复用的核心实现 |
| `artifacts/data/` | 原始、处理后和最终训练数据 |
| `artifacts/models/` | Qwen 基础模型 |
| `results/rqvae/` | RQ-VAE 权重与训练统计 |
| `results/sft/` | SFT 最终模型与统计 |
| `results/grpo/` | GRPO 最终模型与统计 |
| `results/evaluation/` | SFT/GRPO 的最终 HR/NDCG |
| `environment/` | A6000 环境说明和依赖锁 |

## 文档

- [完整运行命令](docs/reproduction_guide.md)
- [实验配置与结果](docs/reproduction_results.md)
- [服务器环境](environment/README.md)
- [MiniOneRec 上游许可证](LICENSE-MiniOneRec.txt)

当前范围只包含官方主流程，不实现 CGRF，不修改 ranking reward，也不下载额外大模型或大数据。
