# MiniOneRec 单卡复现与 CGRF-H 奖励优化

本项目在一张 NVIDIA RTX A6000 48GB 上复现 MiniOneRec 的完整推荐流程，并在不改变 Semantic ID、SFT 数据和官方 ranking reward 的前提下，实现了 CGRF-H（Confidence-Gated Collaborative and Hierarchical Reward）奖励增强。

主线已经完整跑通：

```text
Amazon18 原始数据
    ↓ 5-core 过滤、序列构造、全局时间切分
商品标题 + 描述
    ↓ Qwen3-Embedding-4B
2,560 维商品向量
    ↓ RQ-VAE（32 维 latent，3 × 256 码本）
Semantic ID
    ↓ Qwen2.5-1.5B-Instruct 全参数 BF16 SFT
SFT 推荐模型
    ├── 官方 ranking GRPO ───────────────→ Baseline GRPO
    └── SASRec 教师 + CGRF-H 奖励 GRPO ─→ CGRF-H
                                              ↓
                                同一测试集、同一 Beam-50 SID 评测
```

固定参考的上游版本为 [`0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`](https://github.com/AkaliKong/MiniOneRec/tree/0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed)。当前正式结果只包含 `Industrial_and_Scientific` 数据集；Revival 语义码实验暂不纳入主线、结果表或 Git。

## 1. 系统概览

### 1.1 复现目标

MiniOneRec 将推荐问题改写成语言模型的受约束生成问题：先把每件商品量化成由三个离散 token 组成的 Semantic ID（SID），再让语言模型根据用户历史 SID 生成下一件商品的 SID。这样可以统一利用商品文本语义、序列偏好和大语言模型的生成能力。

本项目分为两个层次：

1. **Baseline 复现**：数据处理、商品 embedding、RQ-VAE/SID、SFT、官方 ranking GRPO 和 Top-K 评测。
2. **CGRF-H 创新**：训练轻量 SASRec 协同教师，在 GRPO 中按教师置信度融合协同排序奖励与 SID 层级前缀奖励，缓解组内候选奖励全部相同的问题。

### 1.2 运行环境

| 项目 | 已验证配置 |
| --- | --- |
| 服务器目录 | `/home/user/wbn/minionerec` |
| 虚拟环境 | `/home/user/wbn/minionerec/.venv-a6000` |
| GPU | NVIDIA RTX A6000，49,140 MiB（约 47.43 GiB） |
| Driver / CUDA 上限 | 565.57.01 / 12.7 |
| Python | 3.11.5 |
| PyTorch | 2.6.0+cu124 |
| Transformers / TRL | 4.57.1 / 0.24.0 |
| Embedding 模型 | Qwen3-Embedding-4B |
| 推荐模型 | Qwen2.5-1.5B-Instruct |

完整服务器依赖快照见 [`requirements-a6000.txt`](requirements-a6000.txt)。本地是唯一代码源，服务器只负责实验；同步代码时不要覆盖服务器的 `.venv-a6000/`、`artifacts/` 或 `results/`。

服务器启动检查：

```bash
cd /home/user/wbn/minionerec
source .venv-a6000/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
python -m pip check
```

模型已经离线保存在：

```text
artifacts/models/Qwen3-Embedding-4B/
artifacts/models/Qwen2.5-1.5B-Instruct/
```

### 1.3 项目目录

```text
minionerec/
├── scripts/                         # Baseline 各阶段命令行入口
├── src/minionerec/
│   ├── data/                        # Amazon 预处理、CSV 转换、SFT/GRPO 数据集
│   ├── semantic_id/                 # embedding、RQ-VAE、SID 生成
│   ├── training/                    # SFT、GRPO、词表与损失
│   ├── generation/                  # 受约束生成与 Beam Search
│   ├── rewards/                     # 官方 ranking reward
│   └── evaluation/                  # HR/NDCG 与补充的 Item-level CCE
├── innovations/
│   └── cgrf_hierarchical_grpo/      # SASRec 教师、奖励回放、CGRF-H 训练
├── artifacts/                       # 本地数据、模型、embedding（不提交 Git）
├── results/                         # Baseline 模型和结果
├── requirements-a6000.txt           # A6000 环境锁
└── README.md
```

### 1.4 复现范围与官方差异

本项目复现固定 commit 的核心方法和数据流，但受单卡硬件、模型规模和实验成本限制，不应将它表述为论文全部实验配置的逐项复刻。

| 对比项 | 论文或上游方案 | 本项目实际设置 |
| --- | --- | --- |
| 代码参考 | MiniOneRec 官方实现 | 固定 commit `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed` |
| 数据范围 | 论文包含多个 Amazon 数据集 | 只完成 `Industrial_and_Scientific` |
| 推荐模型规模 | 论文实验覆盖多个 Qwen2.5-Instruct 规模 | Qwen2.5-1.5B-Instruct |
| 训练硬件 | 论文使用多卡高端 GPU | 单张 RTX A6000 48GB |
| SID token | 论文按完整三层码本描述为 `3 × 256 = 768` | 根据实际 `index.json` 加入出现过的 540 个 token |
| SFT 轮数 | 配置上限依实验而定 | 配置 10 epochs，验证早停于约 4.5 epochs，并回载最佳权重 |
| GRPO 奖励 | exact + ranking reward | baseline 保持不变；CGRF-H 只在创新实验中额外加入稠密奖励 |
| 主评测 | SID-level Top-K 推荐 | 同口径 SID-level HR/NDCG；Item-level CCE 只作补充 |
| 重复实验 | 论文级实验设置 | 当前每种方法一个随机种子、一次正式训练 |

因此，“复现完成”表示整条算法链路、训练任务、奖励和评测已在单卡环境跑通；最终数值应视为当前 1.5B 单卡配置的实验结果，而不是与论文多模型、多硬件结果严格等价的复刻值。

## 2. 模块功能、输入与输出

| 阶段 | 入口 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| Amazon18 预处理 | `scripts/prepare_amazon.py` | reviews、metadata JSONL | `.inter`、`item.json`、`review.json`、`user2id`、`item2id` |
| 数据统计 | `scripts/generate_data_statistics.py` | processed 目录 | `.data_stats.json` |
| 商品 embedding | `scripts/generate_item_embeddings.py` | `item.json`、`item2id`、Qwen3 | `.npy`、manifest |
| RQ-VAE | `scripts/train_rqvae.py` | 商品 embedding | `rqvae_model.pth`、训练统计 |
| SID 生成 | `scripts/generate_semantic_ids.py` | embedding、RQ-VAE checkpoint | `index.json`、SID 统计 |
| 最终数据转换 | `scripts/convert_dataset.py` | 交互切分、商品、SID | train/valid/test CSV、info 文件 |
| SFT | `scripts/train_sft.py` | Qwen2.5、train/valid、商品与 SID | `final_model/`、训练统计 |
| Baseline GRPO | `scripts/train_grpo.py` | SFT 模型、GRPO 数据 | `final_model/`、训练统计 |
| 统一评测 | `scripts/evaluate_sft.py` | 任一最终模型、test、info | HR/NDCG JSON |
| SASRec 教师 | `innovations/.../train_sasrec.py` | `history_item_id → item_id` | SASRec checkpoint、统计 |
| 离线奖励回放 | `innovations/.../analyze_rewards.py` | SFT、SASRec、GRPO 候选 | 候选缓存、奖励分析 |
| CGRF-H GRPO | `innovations/.../train_cgrf_grpo.py` | SFT、SASRec、原 GRPO 数据 | CGRF-H `final_model/`、统计 |
| 结果汇总 | `innovations/.../summarize_results.py` | baseline/CGRF-H JSON | `experiment_summary.json` |

`scripts/` 只负责参数解析和调用；主要实现位于 `src/minionerec/`。创新代码单独放在 `innovations/cgrf_hierarchical_grpo/`，通过 `src` 路径复用 baseline 数据、生成、奖励和评测实现。

## 3. Baseline 执行流程

以下命令均在服务器仓库根目录执行。先定义公共路径：

```bash
cd /home/user/wbn/minionerec
source .venv-a6000/bin/activate

MINIONE_CATEGORY=Industrial_and_Scientific
MINIONE_PROCESSED=artifacts/data/processed/amazon18/Industrial_and_Scientific
MINIONE_FINAL=artifacts/data/final/amazon18
MINIONE_FINAL_NAME=Industrial_and_Scientific_5_1996-10-2018-11
```

### 3.1 数据预处理

```bash
python scripts/prepare_amazon.py \
  --dataset "$MINIONE_CATEGORY" \
  --metadata-file artifacts/data/raw/amazon18/Industrial_and_Scientific/meta_Industrial_and_Scientific.json \
  --reviews-file artifacts/data/raw/amazon18/Industrial_and_Scientific/Industrial_and_Scientific_5.json \
  --output-root artifacts/data/processed/amazon18

python scripts/generate_data_statistics.py \
  --data-dir "$MINIONE_PROCESSED" \
  --dataset-name "$MINIONE_CATEGORY"
```

处理规则：用户和商品均进行 5-core 过滤；每个 next-item 样本最多保留最近 10 条历史；所有样本按目标交互时间统一稳定排序，再按 80%/10%/10% 划分。

### 3.2 商品 embedding

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/generate_item_embeddings.py \
  --item-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.item.json" \
  --item-mapping-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.item2id" \
  --model-path artifacts/models/Qwen3-Embedding-4B \
  --output-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.emb-qwen-td.npy" \
  --batch-size 8 \
  --max-length 2048 \
  --device cuda:0 \
  --torch-dtype float16
```

商品的 `title` 与 `description` 用空格连接。Qwen3 输出 2,560 维 token 隐状态，程序用 attention mask 做均值池化；模型推理用 FP16，最终矩阵保存为 Float32。

### 3.3 RQ-VAE 与 Semantic ID

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_rqvae.py \
  --embedding-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.emb-qwen-td.npy" \
  --output-dir results/rqvae/Industrial_and_Scientific \
  --epochs 10000 \
  --batch-size 20480 \
  --eval-step 50 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 python scripts/generate_semantic_ids.py \
  --embedding-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.emb-qwen-td.npy" \
  --checkpoint-file results/rqvae/Industrial_and_Scientific/rqvae_model.pth \
  --output-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.index.json" \
  --statistics-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.index.stats.json" \
  --device cuda:0
```

RQ-VAE 把 2,560 维商品向量压缩到 32 维 latent，再用三层、每层 256 个向量的残差码本量化。每件商品最终表示成三个 SID token。SID 生成阶段使用 Sinkhorn 对碰撞商品的最后一层分配进行迭代调整。

### 3.4 转换最终训练数据

```bash
python scripts/convert_dataset.py \
  --data-dir "$MINIONE_PROCESSED" \
  --dataset-name "$MINIONE_CATEGORY" \
  --output-dir "$MINIONE_FINAL" \
  --output-name "$MINIONE_FINAL_NAME"
```

CSV 包含 `user_id`、历史/目标标题、历史/目标 Item ID、历史/目标 SID。Item ID 用于目录映射和补充分析，SID 是 SFT、GRPO 与主评测的生成目标。

### 3.5 SFT

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u scripts/train_sft.py \
  --model-path artifacts/models/Qwen2.5-1.5B-Instruct \
  --train-file "$MINIONE_FINAL/train/$MINIONE_FINAL_NAME.csv" \
  --valid-file "$MINIONE_FINAL/valid/$MINIONE_FINAL_NAME.csv" \
  --item-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.item.json" \
  --index-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.index.json" \
  --output-dir results/sft/Industrial_and_Scientific \
  --num-epochs 10 \
  --batch-size 128 \
  --micro-batch-size 4 \
  --learning-rate 3e-4 \
  --max-length 512
```

SFT 包含三类任务：SID 历史预测下一 SID、SID 与商品标题双向对齐、SID 历史预测下一标题。训练时只对 answer token 计算交叉熵，prompt 和 padding 的 label 为 `-100`。

### 3.6 官方 ranking GRPO

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u scripts/train_grpo.py \
  --model-path results/sft/Industrial_and_Scientific/final_model \
  --train-file "$MINIONE_FINAL/train/$MINIONE_FINAL_NAME.csv" \
  --valid-file "$MINIONE_FINAL/valid/$MINIONE_FINAL_NAME.csv" \
  --item-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.item.json" \
  --index-file "$MINIONE_PROCESSED/$MINIONE_CATEGORY.index.json" \
  --info-file "$MINIONE_FINAL/info/$MINIONE_FINAL_NAME.txt" \
  --output-dir results/grpo/Industrial_and_Scientific \
  --micro-batch-size 16 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 64 \
  --num-epochs 2 \
  --learning-rate 1e-5 \
  --num-generations 16 \
  --max-prompt-length 512 \
  --max-completion-length 128 \
  --temperature 1.0 \
  --beta 0.001
```

每个 prompt 约束生成 16 个合法 SID，使用 exact reward 与 ranking reward 计算组内相对优势。`micro_batch_size=16` 表示每个 micro batch 含一个 prompt 的 16 个候选；累积 64 次后更新一次，因此每次参数更新覆盖 64 个唯一 prompt、1,024 个候选。

### 3.7 统一评测

SFT、baseline GRPO 与 CGRF-H 均使用同一个评测入口，只替换 `--model-path` 和 `--output-file`：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/evaluate_sft.py \
  --model-path results/grpo/Industrial_and_Scientific/final_model \
  --test-file "$MINIONE_FINAL/test/$MINIONE_FINAL_NAME.csv" \
  --info-file "$MINIONE_FINAL/info/$MINIONE_FINAL_NAME.txt" \
  --output-file results/evaluation/Industrial_and_Scientific/grpo_metrics.json \
  --batch-size 8 \
  --num-beams 50 \
  --max-new-tokens 256 \
  --length-penalty 0.0 \
  --device cuda:0
```

正式对比采用评估 JSON 的 `metrics` 字段，即固定 commit 对应的 **SID-level HR/NDCG**。文件中的 `item_metrics` 是碰撞修正 Item-level CCE 补充结果，不与主表混用。

## 4. CGRF-H 创新流程

官方奖励在真实目标没有出现在 16 个候选时，经常给整组候选相同或几乎相同的分数，组内标准化后 advantage 为零。CGRF-H 保留官方奖励并增加稠密项：

```text
R = R_official + λ × [g × R_collaborative + (1 - g) × R_hierarchical]
```

- `R_official`：不修改的 exact + ranking reward。
- `R_collaborative`：冻结 SASRec 对候选 Item 的组内排序百分位奖励。
- `R_hierarchical`：候选 SID 与目标 SID 共享 0/1/2/3 层前缀时，奖励为 0/0.2/0.5/1.0。
- `g`：根据真实目标在 SASRec 候选排序中的位置计算的置信门控；教师越可信，越依赖协同奖励，否则更依赖 SID 层级奖励。
- `λ=0.1`：离线奖励回放后选定的保守权重。

流程为：

1. 用真实 `history_item_id → item_id` 训练 14.38 万参数的 SASRec 教师。
2. 冻结 SFT 与 SASRec，在 2,000 个 prompt 上回放同样的 16 候选奖励。
3. 确认稠密奖励减少零 advantage，同时精确目标仍保持最高奖励。
4. 从同一 SFT checkpoint 训练 CGRF-H GRPO；其余超参数与 baseline 完全一致。
5. 评测时只加载最终 Qwen 模型，不需要 SASRec，因此不增加线上推理成本。

SASRec 读取的是最终训练/验证 CSV，而不是预处理阶段的 `.train.inter`；实际只使用 `history_item_id` 和 `item_id` 两列。`index.json` 会从 `Item ID → SID` 反转成 `SID → [Item ID, ...]`。若一个候选 SID 只对应一个 Item，直接使用该 Item 的 SASRec logit；若对应多个 Item，则用下式得到一个对 Item 排列顺序无关的 SID 分数：

```text
SID_score = logsumexp(item_logits) - log(item_count)
```

减去 `log(item_count)` 可以避免碰撞组仅因包含更多 Item 而获得更高分，同时保留组内高分 Item 的影响。该处理服务于当前 SID-level 训练与评测，不能区分同一 SID 内的具体商品；Item-level 部署仍需要二阶段重排。

完整命令与实现说明见 [`innovations/cgrf_hierarchical_grpo/README.md`](innovations/cgrf_hierarchical_grpo/README.md)。

## 5. 关键实验参数

### 5.1 Baseline

| 阶段 | 关键设置 |
| --- | --- |
| 数据 | 5-core；history ≤ 10；全局时间 8:1:1 |
| Embedding | Qwen3-Embedding-4B；max length 2,048；batch 8；FP16 推理、Float32 保存 |
| RQ-VAE | 2560→32；3 × 256 码本；10,000 epochs；batch 20,480；lr 1e-3；warmup 50 |
| SFT | Qwen2.5-1.5B；全参数 BF16；micro 4；累积 32；effective 128；lr 3e-4；max 512 |
| GRPO | G=16；micro 16；累积 64；2 epochs；lr 1e-5；β=1e-3；temperature 1.0 |
| 评测 | 4,533 条测试样本；constrained Beam-50；batch 8；max new tokens 256 |

### 5.2 CGRF-H

| 模块 | 关键设置 |
| --- | --- |
| SASRec | max sequence 10；hidden 32；2 layers；2 heads；dropout 0.3；batch 256；lr 1e-3 |
| 教师选择 | 最多 100 epochs；patience 10；按验证 NDCG@10 选最优 |
| 奖励回放 | 2,000 groups；16 candidates/group；比较 λ=0.1/0.2/0.3 |
| CGRF-H | λ=0.1；其余训练参数与 baseline GRPO 一致 |

## 6. 实验结果

### 6.1 数据与 Semantic ID

| 指标 | 结果 |
| --- | ---: |
| 用户 / 商品 / 过滤后交互 | 7,694 / 3,686 / 53,018 |
| next-item 样本 | 45,324 |
| train / valid / test | 36,259 / 4,532 / 4,533 |
| embedding | `(3686, 2560)`，Float32，有限值，无全零行 |
| RQ-VAE best loss | 0.520868（epoch 2,438） |
| 选用的 best collision checkpoint | epoch 9,950 |
| Sinkhorn 前碰撞 | 472 个冗余，12.8052% |
| Sinkhorn 后碰撞 | 13 个冗余，0.3527% |
| 最终唯一 SID | 3,673 |
| 三层码本使用数 | `[28, 256, 256]` |

序列长度统计：

| 统计对象 | 最短 | 平均 | 最长 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 过滤后每个用户的完整交互序列 | 5 | 6.890824 | 88 | 5-core 后、构造样本前 |
| next-item 样本历史（截断前） | 1 | 4.376820 | 87 | 目标商品之前的全部历史 |
| next-item 样本历史（截断后） | 1 | 3.975907 | 10 | 只保留最近 10 个商品 |

共有 2,664 条样本发生历史截断，占全部 45,324 条 next-item 样本的 5.8777%。截断保持时间顺序，只删除更早的交互。

用户平均只有约 6.89 次交互，但最长序列达到 88，说明该数据集具有明显的序列长度长尾：大多数用户行为较短，少数活跃用户贡献了很长的历史。限制 history 为 10 可以统一模型输入规模、降低显存和长序列噪声，并让模型更关注近期兴趣。由于只有 5.8777% 的样本发生截断，这个限制对绝大多数样本没有信息损失；代价是活跃用户的长期偏好不能完整进入当前样本。

选用的 epoch 9,950 checkpoint 的码本统计：

| RQ-VAE 层 | 已使用 code | 总 code | 利用率 | 作用 |
| --- | ---: | ---: | ---: | --- |
| 第 1 层 | 28 | 256 | 10.9375% | 粗粒度语义簇 |
| 第 2 层 | 256 | 256 | 100% | 对第一层残差继续细分 |
| 第 3 层 | 256 | 256 | 100% | 对第二层残差继续细分 |

第一层只有 10.9375% 的 code 被使用，存在明显的码本集中。可能原因包括：

1. **第一层承担粗粒度聚类。** RQ-VAE 逐层量化残差，第一层主要划分大的语义类别，不一定需要使用全部 256 个 code；第二、三层负责继续区分同一粗粒度簇中的商品。
2. **数据规模相对较小。** 当前只有 3,686 件商品，却为第一层提供了 256 个可选 code，部分粗粒度中心可能没有足够独立的商品簇支撑。
3. **训练目标不直接鼓励均衡使用。** 当前损失主要优化重构误差、量化误差和 commitment loss，没有额外的码本熵或利用率正则项。只要少量第一层 code 配合后两层能够降低重构损失，优化器就没有动力主动激活死码。
4. **初始化与早期分配会产生路径依赖。** KMeans 初始化后，获得较多样本的中心会持续收到更多更新；早期未被选择的中心可能逐渐成为死码。

低利用率会把区分商品的压力推到第二、三层，降低第一层作为“语义大类”的表达丰富度，并可能增加完整 SID 碰撞风险。但它不等于 RQ-VAE 训练失败：本实验第二、三层利用率均为 100%，三层组合在 Sinkhorn 处理后得到 3,673 个唯一 SID，最终碰撞冗余只有 13。因而当前模型可以支撑后续流程，不过第一层利用率仍是后续可优化的明确方向。

这里的利用率只表示“至少被一个商品选择的 code 比例”，不能说明已使用 code 之间是否均衡。例如 28 个 code 全部被使用，仍可能存在某一个 code 容纳大量商品的情况。若要判断码本是否真正健康，还应同时观察每层 occupancy 分布、perplexity、最大/最小占用数和完整 SID collision rate；baseline 现有统计只完整记录了使用数与碰撞，因此 README 不额外推测未记录的分布指标。

### 6.2 训练结果与资源

| 阶段 | 结果 | 耗时 | 峰值显存 |
| --- | --- | ---: | ---: |
| 商品 embedding | 3,686 件完成 | 143 秒 | 未记录 |
| SFT | 2,808 steps；best eval loss 1.526703 | 10,757 秒（2.99 h） | 未记录 |
| Baseline GRPO | 1,650 steps；2 epochs | 49,640.6 秒（13.79 h） | 10.88 GiB allocated |
| SASRec | best epoch 66；NDCG@10 0.110354 | 30.7 秒 | 0.045 GiB |
| CGRF-H GRPO | 1,650 steps；2 epochs | 49,745.0 秒（13.82 h） | 10.88 GiB allocated |

CGRF-H 比 baseline GRPO 多 104.3 秒，仅增加约 0.21% 的训练时间；两者峰值显存相同。SASRec 不参与最终推理。

### 6.3 离线奖励回放

| 指标 | Baseline reward | CGRF-H（λ=0.1） |
| --- | ---: | ---: |
| 零 advantage 组比例 | 70.4% | 2.1% |
| 可提供非零组内学习信号的组比例 | 29.6% | 97.9% |
| 精确目标保持组内最高奖励 | 100% | 100% |

零 advantage 率绝对下降 68.3 个百分点，相对减少 97.02%。这验证了 CGRF-H 的直接目的：不替换官方目标，只为稀疏候选组补充可区分的学习信号。

### 6.4 最终 SID-level 推荐指标

所有模型使用同一 4,533 条测试集和 constrained Beam-50。

| 指标 | SFT | Baseline GRPO | CGRF-H | CGRF-H 相对 GRPO |
| --- | ---: | ---: | ---: | ---: |
| HR@1 | 0.059343 | 0.061769 | 0.060887 | -1.429% |
| NDCG@1 | 0.059343 | 0.061769 | 0.060887 | -1.429% |
| HR@3 | 0.086698 | 0.093316 | 0.092654 | -0.709% |
| NDCG@3 | 0.075042 | 0.079940 | 0.079428 | -0.641% |
| HR@5 | 0.104787 | 0.110743 | **0.111405** | **+0.598%** |
| NDCG@5 | 0.082456 | 0.087088 | **0.087204** | **+0.133%** |
| HR@10 | 0.137216 | 0.139643 | **0.142290** | **+1.896%** |
| NDCG@10 | 0.092869 | 0.096283 | **0.097147** | **+0.897%** |
| HR@20 | 0.179351 | 0.181337 | **0.184867** | **+1.946%** |
| NDCG@20 | 0.103357 | 0.106690 | **0.107928** | **+1.160%** |
| HR@50 | 0.237591 | 0.240238 | **0.249724** | **+3.949%** |
| NDCG@50 | 0.114912 | 0.118380 | **0.120805** | **+2.048%** |

相对 baseline GRPO，CGRF-H 在测试集中使 HR@10、HR@20、HR@50 分别多命中 12、16、43 条；收益随 K 增大而更明显。代价是 HR@1 少 4 条、HR@3 少 3 条，说明当前奖励更偏向扩大高质量候选覆盖，而不是进一步集中 Top-1 概率。

### 6.5 结论边界

- 当前结果证明的是：在同一固定配置的一次实验中，CGRF-H 明显改善奖励密度，并提升 K≥5 的 SID-level 指标。
- CGRF-H 验证 KL 为 0.8447，baseline 为 0.1648，策略偏移更强；后续可增加 λ/β 消融来平衡 Top-1 与候选覆盖。
- 目前只有一个数据集、一个随机种子，尚未进行显著性检验，因此不声称统计显著。
- λ=0.1/0.2/0.3 只完成了离线奖励回放，只有 λ=0.1 完成下游训练；不能据此声称 λ=0.1 是全局最优。
- SID 碰撞相关的 Item-level CCE 仅作补充。主表保持与 baseline 一致的 SID-level 口径。

## 7. 结果文件

大型数据、模型、checkpoint、训练日志和创新原始结果由 `.gitignore` 排除。可审查的 baseline JSON 保存在 `results/`，CGRF-H 的统一紧凑汇总保存在：

```text
innovations/cgrf_hierarchical_grpo/experiment_summary.json
```

从原始统计文件重新生成汇总：

```bash
python innovations/cgrf_hierarchical_grpo/scripts/summarize_results.py
```

正式模型仍可直接评测；已清理的 Trainer 中间 checkpoint 不再用于断点续训。重新训练某阶段前，应先备份或移动该阶段现有输出目录，避免覆盖正式结果。

## 8. 参考

- [MiniOneRec 论文](https://arxiv.org/abs/2510.24431)
- [MiniOneRec 固定上游 commit](https://github.com/AkaliKong/MiniOneRec/tree/0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed)
- [CGRF-H 详细说明](innovations/cgrf_hierarchical_grpo/README.md)
- [MiniOneRec License](LICENSE-MiniOneRec.txt)
