# MiniOneRec 单张 RTX A6000 复现结果

## 1. 结论

本项目已经在单张 NVIDIA RTX A6000 48GB 上跑通 MiniOneRec 的第一阶段完整流程：

`Amazon18 数据处理 → 商品文本 embedding → RQ-VAE/SID → SFT → GRPO → Top-K 评测`

本次实验使用 `Industrial_and_Scientific` 数据集和 Qwen2.5-1.5B-Instruct。最终 GRPO 模型在 4,533 条测试样本上达到：

- HR@3：0.093316
- NDCG@3：0.079940
- HR@5：0.110743
- NDCG@5：0.087088
- HR@10：0.139643
- NDCG@10：0.096283

GRPO 相比同一实验中的 SFT 模型，在所有已统计的 HR 和 NDCG 指标上均有提升。其中 HR@3 相对提升 7.63%，NDCG@3 相对提升 6.53%。这说明在相同模型、数据和 SID 空间下，ranking GRPO 确实进一步改善了推荐排序效果。

## 2. 实验范围与可比性

- 官方代码参考固定在 upstream commit `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`。
- 只复现论文的 MiniOneRec 主流程，不包含 CGRF，不修改官方 ranking reward。
- 只使用 Amazon18 的 `Industrial_and_Scientific`，暂未运行 `Office_Products`。
- 商品文本编码器使用 Qwen3-Embedding-4B。
- 推荐模型使用 Qwen2.5-1.5B-Instruct，并进行全参数 BF16 SFT 和 GRPO。
- 所有训练与评测均在单张 RTX A6000 上完成。

论文实验覆盖 0.5B 至 7B 的 Qwen2.5-Instruct，并使用 8 张 H100 进行 SFT；但论文表 1 没有在表内注明 MiniOneRec 这一行对应的具体参数规模。论文表 1 的结果不能被视为本次 1.5B 单卡实验的同配置结果，因此本文只将它作为论文参考值，不把差距解释为复现失败。

参考资料：

- [MiniOneRec 官方论文](https://arxiv.org/pdf/2510.24431)
- [固定 upstream commit](https://github.com/AkaliKong/MiniOneRec/tree/0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed)

## 3. 运行环境

| 项目 | 实际环境 |
| --- | --- |
| GPU | NVIDIA RTX A6000，47.43 GiB |
| NVIDIA Driver | 565.57.01 |
| Driver 支持的最高 CUDA | 12.7 |
| Python | 3.11.5 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA Runtime | 12.4 |
| Transformers | 4.57.1 |
| TRL | 0.24.0 |
| Accelerate | 1.10.1 |
| BitsAndBytes | 0.48.1 |
| NumPy | 1.26.3 |

服务器项目目录为 `/home/user/wbn/minionerec`，Python 环境为项目内的 `.venv-a6000`。模型、数据和训练过程均可离线运行。

## 4. 数据处理结果

### 4.1 处理规则

1. 读取 Amazon18 Industrial_and_Scientific 的交互和商品元数据。
2. 过滤交互次数少于 5 的用户和商品。
3. 按时间排列用户交互。
4. 将每个用户历史截断为最近 10 个商品。
5. 由交互序列构造 next-item 样本。
6. 将全部样本按目标交互时间统一排序，再按 8:1:1 划分训练、验证和测试集。

这里采用的是全局时间切分，不是对每个用户分别留出最后两个商品。因此测试集中允许出现验证集中没有出现过的用户，但该样本自身仍带有预测目标之前的历史交互。

### 4.2 数据规模

| 指标 | 数量 |
| --- | ---: |
| 用户数 | 7,694 |
| 商品数 | 3,686 |
| 过滤后交互数 | 53,018 |
| next-item 样本数 | 45,324 |
| 训练样本 | 36,259 |
| 验证样本 | 4,532 |
| 测试样本 | 4,533 |
| 最大历史长度 | 10 |

过滤后交互数大于 next-item 样本数是正常的。一个拥有 `n` 次交互的用户最多产生 `n-1` 个 next-item 样本，因为第一次交互前没有历史可作为输入。

论文表 4 写的是 3,685 个 Industrial 商品，而本次由官方预处理链实际得到 3,686 行商品数据；训练、验证、测试样本数与论文表 4 一致。本次没有为了匹配表格数字而额外删除商品。

最终 CSV 的字段含义如下：

| 字段 | 含义 | 后续用途 |
| --- | --- | --- |
| `user_id` | 映射后的用户 ID | 样本标识和数据分析 |
| `history_item_title` | 历史商品标题序列 | 文本历史相关的 SFT/GRPO 任务 |
| `item_title` | 目标商品标题 | 标题生成与 SID/标题对齐 |
| `history_item_id` | 历史商品内部 ID 序列 | 数据一致性检查 |
| `item_id` | 目标商品内部 ID | 查找目标商品信息 |
| `history_item_sid` | 历史商品 SID 序列 | 主推荐任务的模型输入 |
| `item_sid` | 目标商品 SID | 训练答案、奖励计算和评测真值 |

## 5. 商品 embedding

### 5.1 原理与配置

每个商品将 `title` 和 `description` 用一个空格拼接，输入冻结的 Qwen3-Embedding-4B。Tokenizer 将文本截断到最多 2,048 个 token；不同商品不需要具有相同的有效 token 数，批内 padding 只用于并行计算。模型输出每个 token 的 2,560 维隐藏状态，再使用 attention mask 做加权平均池化，得到固定的 2,560 维商品向量。

模型推理使用 FP16，以降低显存和提升速度；保存时转成 Float32，使后续 RQ-VAE 使用 Float32 计算。转存不能恢复 FP16 推理已经损失的精度，但能避免在后续阶段继续使用更低精度存储和计算。2,048 是输入 token 上限，不是 embedding 维度。

### 5.2 实际结果

| 指标 | 结果 |
| --- | --- |
| 商品数 | 3,686 |
| embedding 形状 | `(3686, 2560)` |
| 保存类型 | `float32` |
| NaN/Inf | 0 |
| 全零向量 | 0 |
| L2 范数最小值 | 84.8837 |
| L2 范数平均值 | 97.3637 |
| L2 范数最大值 | 110.2058 |
| batch size | 8 |
| 运行时间 | 2 分 23 秒 |

`finite=True` 表示矩阵中所有数值都是有限实数，没有 NaN、正无穷或负无穷。

## 6. RQ-VAE 与 Semantic ID

### 6.1 配置

| 参数 | 数值 |
| --- | ---: |
| 输入维度 | 2,560 |
| latent 维度 | 32 |
| 残差量化层数 | 3 |
| 每层码本大小 | 256 |
| batch size | 20,480 |
| learning rate | 1e-3 |
| epochs | 10,000 |
| warmup epochs | 50 |
| commitment beta | 0.25 |
| KMeans 初始化迭代 | 100 |
| 指标记录间隔 | 50 epochs |

RQ-VAE 先把 2,560 维语义向量压缩到 32 维 latent，再由三层码本逐层量化残差。每个商品最终表示为三个离散码，例如 `<a_12><b_34><c_56>`。低维 latent 使距离计算和聚类更稳定，也降低了在仅 3,686 个样本上学习高维码本的难度。

### 6.2 训练结果

| 指标 | 结果 |
| --- | ---: |
| 最低 total loss | 0.520868（epoch 2,438） |
| 最低训练期 collision rate | 0.128052（epoch 9,950） |
| epoch 10,000 total loss | 0.558625 |
| epoch 10,000 reconstruction loss | 0.125462 |
| epoch 10,000 collision rate | 0.135648 |
| epoch 10,000 各层使用码数 | `[28, 256, 256]` |

生成 SID 时采用 epoch 9,950 的 `best_collision` 权重。第一层只使用 28 个码，说明粗粒度聚类存在明显集中；后两层使用全部 256 个码，继续细分残差。三层组合仍提供足够的商品区分能力。

### 6.3 碰撞消解结果

两个或多个商品得到完全相同的三层 SID 时称为碰撞。按商品数定义：

`collision_rate = (item_count - unique_sid_count) / item_count`

| 指标 | 消解前 | 20 轮消解后 |
| --- | ---: | ---: |
| 商品数 | 3,686 | 3,686 |
| 唯一 SID 数 | 3,214 | 3,673 |
| 碰撞商品冗余数 | 472 | 13 |
| collision rate | 12.8052% | 0.3527% |
| 碰撞组数 | 314 | 12 |
| 最大碰撞组大小 | 19 | 3 |
| 各层使用码数 | `[28, 256, 256]` | `[28, 256, 256]` |

消解阶段只对碰撞商品重新分配最后一层码，使用 Sinkhorn epsilon 0.003 让分配更均衡。它没有增加第四个“冲突位”，因此最终仍有 13 个碰撞冗余；评测目录也按 3,673 个唯一 SID 建立。

## 7. SFT

### 7.1 词表与任务

原始 tokenizer 有 151,665 个 token。实际出现的三层 SID 码共 540 个，因此向 tokenizer 增加 540 个不可再切分的 SID token，并将模型输入/输出 embedding 扩展到 152,205 行。三个 SID token 联合表示一个商品；没有出现的码不加入词表。

SFT 使用三类数据：

| 任务 | 样本数 | 学习目标 |
| --- | ---: | --- |
| SID 历史 → 下一个 SID | 36,259 | 学习主推荐任务 |
| SID ↔ 商品标题对齐 | 7,319 | 建立 SID 与标题的双向语义联系 |
| SID 历史 → 下一个标题 | 36,259 | 将 SID 序列偏好映射回自然语言标题 |
| 合计 | 79,837 | 三类数据拼接训练 |
| 验证集 | 4,532 | 验证 SID next-item 预测 |

训练输入包含 prompt 和正确 answer，标签与输入等长。prompt 和 padding 位置的 label 设为 `-100`，交叉熵只监督 answer token。因果注意力保证模型预测某个 answer token 时只能看到 prompt 和此前的 answer token，不能看到当前位置或未来答案。

论文附录描述的是把三层完整码本的 `3 × 256 = 768` 个 token 加入词表；本次固定 commit 实现根据实际 `index.json` 收集出现过的码，只加入 540 个，未使用的码不参与训练。

### 7.2 实际配置

| 参数 | 数值 |
| --- | ---: |
| 基座模型 | Qwen2.5-1.5B-Instruct |
| 参数量（扩词表后） | 1,544,127,488 |
| 训练精度 | BF16，全参数 |
| micro batch size | 4 |
| 梯度累积 | 32 |
| effective batch size | 128 |
| 最大长度 | 512 |
| learning rate | 3e-4 |
| warmup steps | 20 |
| scheduler | Transformers 默认 linear |
| 计划 epochs | 10 |
| 验证与保存间隔 | 总进度每 5% |
| checkpoint 保留数 | 1 |
| 早停 patience | 3 次验证，即约 1.5 epochs |

### 7.3 训练结果

| 指标 | 结果 |
| --- | ---: |
| 实际停止 epoch | 4.5002 |
| global step | 2,808 |
| 平均 training loss | 0.526101 |
| 最佳 checkpoint | epoch 3.0，step 1,872 |
| 最佳 eval loss | 1.526703 |
| 训练耗时 | 10,757 秒，约 2 小时 59 分 |

训练在第 4.5 个 epoch 触发早停，不是异常中断。由于启用了 `load_best_model_at_end=True`，写入 `final_model` 的是 step 1,872 的最低验证损失权重，而不是第 4.5 个 epoch 的最后一次权重。

最终模型验证通过：tokenizer、模型配置、输入 embedding 和输出 embedding 的词表大小均为 152,205；输入输出 embedding 权重共享；单条验证样本 forward loss 为 2.808593，loss 和 logits 均为有限值。

## 8. GRPO

### 8.1 与 SFT 的区别

SFT 为每个 prompt 提供一个正确答案，并最小化答案 token 的交叉熵。GRPO 则从 SFT 模型出发，为同一个 prompt 生成一组候选 SID，按照是否命中目标和负样本排名计算奖励，再用组内标准化后的相对优势更新模型。

GRPO 训练仍然包含推荐与 SID/文本对齐任务，但只保留能够用封闭 SID 空间验证和打分的方向：

| 任务 | 样本数 |
| --- | ---: |
| SID 历史 → 下一个 SID | 36,259 |
| 标题 → SID | 3,646 |
| 描述 → SID | 2,870 |
| 标题序列 → SID | 10,000 |
| 合计 | 52,775 |
| 验证集 | 4,532 |

### 8.2 实际配置

| 参数 | 数值 |
| --- | ---: |
| 初始模型 | SFT `final_model` |
| policy/reference 模型加载数 | 2 |
| 训练精度 | BF16，全参数 |
| 每个 prompt 的候选数 G | 16 |
| micro batch size | 16 条候选，即 1 个唯一 prompt |
| gradient accumulation | 64 |
| 每次更新的候选总数 | 1,024 |
| 每次更新的唯一 prompt 数 | 64 |
| epochs | 2 |
| global steps | 1,650 |
| learning rate | 1e-5 |
| scheduler | cosine，warmup ratio 0.03 |
| KL beta | 0.001 |
| temperature | 1.0 |
| 最大 prompt/completion 长度 | 512 / 128 |
| constrained beam width | 16 |
| reference 同步 | 每 512 step，mixup alpha 0.6 |
| gradient checkpointing | 开启 |
| optimizer | paged AdamW 32-bit |

奖励由两部分组成：正确 SID 的 exact reward，以及对高置信错误候选施加更强惩罚的 ranking reward。约束解码只允许生成目录中合法的 SID，因此正式训练和测试中的合法候选率均为 100%。

### 8.3 训练结果

| 指标 | 结果 |
| --- | ---: |
| 完成 epoch | 2.0 |
| global step | 1,650 |
| training loss | 0.003975 |
| 最终验证 loss | 0.000165 |
| 最终验证 reward | 0.001261 |
| 最终验证 exact reward | 0.011060 |
| 最终验证 KL | 0.164816 |
| GPU peak allocated | 10.88 GiB |
| GPU peak reserved | 12.65 GiB |
| 保留 checkpoint | `checkpoint-1650` |

GRPO loss 是策略目标与 KL 约束形成的优化量，不能和 SFT 的 token 交叉熵直接比较。判断 GRPO 是否有效应以同一测试集上的 HR/NDCG 为主。

## 9. 最终推荐效果

评测对每条测试样本执行 50-beam 约束解码，返回 50 个候选 SID。测试集共 4,533 条样本、226,650 个候选，SFT 和 GRPO 的合法候选率均为 100%。

### 9.1 SFT 与 GRPO 对比

绝对提升按百分点计算：`(GRPO - SFT) × 100`。相对提升按 `((GRPO - SFT) / SFT) × 100%` 计算。

| 指标 | SFT | GRPO | 绝对提升 | 相对提升 |
| --- | ---: | ---: | ---: | ---: |
| HR@1 | 0.059343 | 0.061769 | +0.243 个百分点 | +4.09% |
| NDCG@1 | 0.059343 | 0.061769 | +0.243 个百分点 | +4.09% |
| HR@3 | 0.086698 | 0.093316 | +0.662 个百分点 | +7.63% |
| NDCG@3 | 0.075042 | 0.079940 | +0.490 个百分点 | +6.53% |
| HR@5 | 0.104787 | 0.110743 | +0.596 个百分点 | +5.68% |
| NDCG@5 | 0.082456 | 0.087088 | +0.463 个百分点 | +5.62% |
| HR@10 | 0.137216 | 0.139643 | +0.243 个百分点 | +1.77% |
| NDCG@10 | 0.092869 | 0.096283 | +0.341 个百分点 | +3.68% |
| HR@20 | 0.179351 | 0.181337 | +0.199 个百分点 | +1.11% |
| NDCG@20 | 0.103357 | 0.106690 | +0.333 个百分点 | +3.23% |
| HR@50 | 0.237591 | 0.240238 | +0.265 个百分点 | +1.11% |
| NDCG@50 | 0.114912 | 0.118380 | +0.347 个百分点 | +3.02% |

GRPO 测试耗时为 1,049.5 秒，吞吐为 4.319 samples/s，评测峰值显存为 7.12 GiB。

### 9.2 与论文表 1 对比

| 指标 | 本次 1.5B 单卡 GRPO | 论文 MiniOneRec | 绝对差值 |
| --- | ---: | ---: | ---: |
| HR@3 | 0.093316 | 0.1143 | -2.098 个百分点 |
| NDCG@3 | 0.079940 | 0.1011 | -2.116 个百分点 |
| HR@5 | 0.110743 | 0.1321 | -2.136 个百分点 |
| NDCG@5 | 0.087088 | 0.1084 | -2.131 个百分点 |
| HR@10 | 0.139643 | 0.1586 | -1.896 个百分点 |
| NDCG@10 | 0.096283 | 0.1167 | -2.042 个百分点 |

该表用于说明当前单卡结果所处的位置，不是严格消融实验。主要不可比因素包括：本次模型明确为 1.5B，而论文表 1 未标注该行的具体参数规模；论文 SFT 使用 8 张 H100；本次 SFT 的早停验证频率与 patience 按单卡实现调整；本次只扩展实际出现的 540 个 SID token；本次 SID 仍有 13 个碰撞冗余。

从结果上看，本次 1.5B 模型已经高于论文表 1 的多个传统和生成式基线。例如本次 HR@10 为 0.1396，高于论文中的 SASRec 0.1088、HSTU 0.1163、TIGER 0.1321、LCRec 0.1332 和 BIGRec 0.1370，但仍低于 D3、S-DPO 和论文完整 MiniOneRec。由于这些数值来自论文中各方法的既有实验，该比较同样只作为位置参考。

## 10. 主要产物

| 阶段 | 产物位置 |
| --- | --- |
| 商品 embedding | `artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy` |
| RQ-VAE 模型 | `results/rqvae/Industrial_and_Scientific/rqvae_model.pth` |
| RQ-VAE 统计 | `results/rqvae/Industrial_and_Scientific/rqvae_training_stats.json` |
| 商品 SID | `artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index.json` |
| SID 统计 | `artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index.stats.json` |
| 最终训练 CSV | `artifacts/data/final/amazon18/{train,valid,test}/Industrial_and_Scientific_5_2016-10-2018-11.csv` |
| 数据目录信息 | `artifacts/data/final/amazon18/info/Industrial_and_Scientific_5_2016-10-2018-11.txt` |
| SFT 最终模型 | `results/sft/Industrial_and_Scientific/final_model/` |
| SFT 训练统计 | `results/sft/Industrial_and_Scientific/training_stats.json` |
| SFT 测试指标 | `results/evaluation/Industrial_and_Scientific/sft_metrics.json` |
| GRPO 最终模型 | `results/grpo/Industrial_and_Scientific/final_model/` |
| GRPO 训练统计 | `results/grpo/Industrial_and_Scientific/training_stats.json` |
| GRPO 测试指标 | `results/evaluation/Industrial_and_Scientific/grpo_metrics.json` |

`artifacts/data` 保存后续阶段会继续消费的数据产物；`results` 保存训练模型、训练统计和最终实验指标。

## 11. 面试版总结

这个项目把推荐问题转成了受约束的语言生成问题。首先用 Qwen3-Embedding-4B 把商品标题和描述编码成 2,560 维语义向量，再用三层 RQ-VAE 将商品压缩成三个离散 SID token。然后扩展 Qwen2.5-1.5B-Instruct 的词表，通过多任务 SFT 同时学习用户序列推荐以及 SID 与商品文本之间的语义对齐。最后使用 GRPO 对同一 prompt 生成 16 个合法候选，根据命中和排序难度构造组内相对奖励，继续优化模型。

单张 A6000 完成了从数据处理、SID 构造、全参数 SFT、全参数 GRPO 到 50-beam 测试的完整闭环。GRPO 后 HR@3 从 0.0867 提升到 0.0933，NDCG@3 从 0.0750 提升到 0.0799，且全部 Top-K 指标均优于 SFT，证明强化学习阶段在当前实验中带来了稳定但幅度有限的排序收益。

当前局限是只跑了一个较小数据集、推荐模型规模为 1.5B、SID 仍有 13 个碰撞冗余，尚未复现 Office 数据集和论文的多卡规模实验。下一阶段若要增强说服力，应优先在不改变算法的前提下补跑 Office 或更大数据集，并保持相同评测协议。
