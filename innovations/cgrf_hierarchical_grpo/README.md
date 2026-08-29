# CGRF-H：置信门控的协同—层级 GRPO 奖励

本目录实现 MiniOneRec 的 CGRF-H（Confidence-Gated Collaborative and Hierarchical Reward）创新实验。它复用 baseline 的 Semantic ID、SFT checkpoint、GRPO 数据、受约束生成和 SID-level 评测，只修改 GRPO 的奖励计算。

## 1. 动机与方法

官方 ranking GRPO 对每个 prompt 生成 16 个 SID 候选。当真实目标不在候选中、错误候选又缺少可区分关系时，一整组奖励可能完全相同；组内标准化后 advantage 为零，这个 prompt 不产生有效策略梯度。

CGRF-H 在不替换官方奖励的前提下增加稠密项：

```text
R = R_official + λ × [g × R_collaborative + (1 - g) × R_hierarchical]
```

其中：

- `R_official` 是固定 commit 中的 exact reward 与 ranking reward 之和。
- `R_hierarchical` 根据候选 SID 和目标 SID 的公共前缀层数给出 `0 / 0.2 / 0.5 / 1.0`。
- `R_collaborative` 将冻结 SASRec 对组内候选的分数转换成 `[0, 1]` 排名百分位。
- `g` 由真实目标在 SASRec 排序中的名次计算。教师越确信真实目标，`g` 越接近 1；否则退回更稳定的 SID 层级奖励。
- `λ` 控制稠密奖励强度。本次离线回放比较 `0.1 / 0.2 / 0.3`，正式训练采用 `0.1`。

SASRec 只在训练阶段提供奖励，不会加入最终 Qwen 模型，也不会增加评测和线上推理开销。

## 2. 文件关系

```text
scripts/train_sasrec.py
    └── src/.../sasrec.py + sasrec_training.py
            ↓ sasrec_model.pth
scripts/analyze_rewards.py
    └── src/.../reward_replay.py + reward_fusion.py
            ↓ reward_analysis.json
scripts/train_cgrf_grpo.py
    └── src/.../cgrf_training.py + reward_fusion.py
            ↓ CGRF-H final_model + training_stats.json
baseline scripts/evaluate_sft.py
            ↓ 同口径 sid_metrics.json
scripts/summarize_results.py
            ↓ experiment_summary.json
```

| 文件 | 功能 |
| --- | --- |
| `src/.../sasrec.py` | 轻量单向 Transformer 序列推荐模型 |
| `src/.../sasrec_training.py` | Item ID 数据集、训练、验证和早停 |
| `src/.../reward_fusion.py` | 官方、层级、协同奖励与置信门控 |
| `src/.../reward_replay.py` | 冻结模型生成候选并离线比较奖励 |
| `src/.../cgrf_training.py` | 在 baseline GRPO 上接入 CGRF-H 奖励 |
| `scripts/train_sasrec.py` | SASRec 命令行入口 |
| `scripts/analyze_rewards.py` | 离线奖励回放入口 |
| `scripts/train_cgrf_grpo.py` | CGRF-H 正式训练入口 |
| `scripts/summarize_results.py` | 聚合 baseline 与创新结果 |

## 3. 实验流程

所有命令从仓库根目录 `/home/user/wbn/minionerec` 执行，并要求 baseline SFT `final_model` 和正式数据已经存在。

### 3.1 训练 SASRec 教师

```bash
CUDA_VISIBLE_DEVICES=0 python innovations/cgrf_hierarchical_grpo/scripts/train_sasrec.py \
  --train-file artifacts/data/final/amazon18/train/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --valid-file artifacts/data/final/amazon18/valid/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --num-items 3686 \
  --output-dir innovations/cgrf_hierarchical_grpo/results/sasrec \
  --device cuda:0
```

输入仅使用真实 Item ID：`history_item_id → item_id`。CSV 中的 Item ID `0` 是正常商品，模型内部统一加一并保留内部 ID `0` 作为 padding。只有训练集更新参数，验证集按 NDCG@10 选择最佳 checkpoint。

这里的 `--train-file` 和 `--valid-file` 都是 `artifacts/data/final/amazon18/` 下的最终 CSV，不是 processed 目录中的 `.train.inter` 中间文件。SASRec 只读取 CSV 的 `history_item_id` 和 `item_id` 两列，标题、描述和 SID 不参与教师训练。

实际配置：

| 参数 | 数值 |
| --- | ---: |
| 参数量 | 143,776 |
| 最大序列长度 | 10 |
| hidden size | 32 |
| Transformer layers / heads | 2 / 2 |
| dropout | 0.3 |
| train / eval batch | 256 / 512 |
| learning rate | 1e-3 |
| 最大 epoch / patience | 100 / 10 |
| seed / workers | 42 / 4 |

#### SID 映射与碰撞组聚合

SASRec 预测的是真实 Item ID，而 Qwen/GRPO 生成的是 SID。程序从 baseline `index.json` 读取 `Item ID → SID`，然后建立完整的反向映射：

```text
SID → [Item ID 1, Item ID 2, ...]
```

对唯一 SID，协同分数就是对应 Item 的 SASRec logit。对碰撞 SID，程序不会随机选择 Item，也不会只选择映射中的第一项，而是计算：

```text
SID_score = logsumexp(item_logits) - log(item_count)
```

这等价于 Item 指数分数平均值的对数。它比普通最大值更平滑，又通过减去 `log(item_count)` 消除碰撞组大小造成的天然加分。真实 `target_item_id` 用于检查目标 Item 确实属于目标 SID；目标 SID 的协同分数仍按整个碰撞组聚合，从而保持训练奖励与 SID-level 评测一致。

该设计不能解决同一个 SID 内部的 Item 区分问题。如果系统最终需要输出真实 Item，应在 SID 生成后使用 SASRec 分数或业务排序模型对碰撞组进行二阶段重排。

### 3.2 离线奖励回放

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python innovations/cgrf_hierarchical_grpo/scripts/analyze_rewards.py \
  --model-path results/sft/Industrial_and_Scientific/final_model \
  --sasrec-checkpoint innovations/cgrf_hierarchical_grpo/results/sasrec/sasrec_model.pth \
  --train-file artifacts/data/final/amazon18/train/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --valid-file artifacts/data/final/amazon18/valid/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --item-file artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item.json \
  --index-file artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index.json \
  --info-file artifacts/data/final/amazon18/info/Industrial_and_Scientific_5_1996-10-2018-11.txt \
  --output-dir innovations/cgrf_hierarchical_grpo/results/reward_replay \
  --sample 2000 \
  --num-generations 16 \
  --lambdas 0.1 0.2 0.3 \
  --device cuda:0
```

该阶段冻结 SFT 和 SASRec，不创建 optimizer 或 reference model。它缓存与 GRPO 相同的 16 个受约束候选，并比较不同奖励公式是否满足：数值有限、降低零 advantage、精确目标仍获得最高奖励。

### 3.3 两步冒烟测试

```bash
CGRF_SMOKE_DIR=$(mktemp -d /tmp/minionerec-cgrf-h-smoke.XXXXXX)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u innovations/cgrf_hierarchical_grpo/scripts/train_cgrf_grpo.py \
  --model-path results/sft/Industrial_and_Scientific/final_model \
  --sasrec-checkpoint innovations/cgrf_hierarchical_grpo/results/sasrec/sasrec_model.pth \
  --train-file artifacts/data/final/amazon18/train/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --valid-file artifacts/data/final/amazon18/valid/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --item-file artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item.json \
  --index-file artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index.json \
  --info-file artifacts/data/final/amazon18/info/Industrial_and_Scientific_5_1996-10-2018-11.txt \
  --output-dir "$CGRF_SMOKE_DIR" \
  --dense-weight 0.1 \
  --smoke-test
```

冒烟测试只执行两个 optimizer step，不验证、不保留中间 checkpoint，输出位于 `/tmp`。

### 3.4 正式 CGRF-H GRPO

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u innovations/cgrf_hierarchical_grpo/scripts/train_cgrf_grpo.py \
  --model-path results/sft/Industrial_and_Scientific/final_model \
  --sasrec-checkpoint innovations/cgrf_hierarchical_grpo/results/sasrec/sasrec_model.pth \
  --train-file artifacts/data/final/amazon18/train/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --valid-file artifacts/data/final/amazon18/valid/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --item-file artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item.json \
  --index-file artifacts/data/processed/amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index.json \
  --info-file artifacts/data/final/amazon18/info/Industrial_and_Scientific_5_1996-10-2018-11.txt \
  --output-dir innovations/cgrf_hierarchical_grpo/results/grpo/cgrf_h_lambda_01 \
  --dense-weight 0.1 \
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

除 `dense_weight=0.1` 和冻结 SASRec 教师外，数据、初始化模型、batch、学习率、生成、KL、scheduler 与 baseline GRPO 相同。正式训练完成 2 epochs、1,650 steps。

### 3.5 同口径评测

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/evaluate_sft.py \
  --model-path innovations/cgrf_hierarchical_grpo/results/grpo/cgrf_h_lambda_01/final_model \
  --test-file artifacts/data/final/amazon18/test/Industrial_and_Scientific_5_1996-10-2018-11.csv \
  --info-file artifacts/data/final/amazon18/info/Industrial_and_Scientific_5_1996-10-2018-11.txt \
  --output-file innovations/cgrf_hierarchical_grpo/results/evaluation/cgrf_h_lambda_01/sid_metrics.json \
  --batch-size 8 \
  --num-beams 50 \
  --max-new-tokens 256 \
  --length-penalty 0.0 \
  --device cuda:0
```

主实验读取输出 JSON 的 `metrics` 字段（SID-level），与 baseline 完全一致。`item_metrics` 是额外的碰撞修正 Item-level CCE，不作为本创新主表。

## 4. 实验结果

### 4.1 SASRec

| 指标 | 结果 |
| --- | ---: |
| best epoch / executed epochs | 66 / 76 |
| HR@10 | 0.146293 |
| NDCG@10 | 0.110354 |
| 训练耗时 | 30.736 秒 |
| 峰值 allocated 显存 | 0.045 GiB |

### 4.2 奖励回放

2,000 个候选组、每组 16 个不同合法 SID：

| 指标 | Baseline | CGRF-H λ=0.1 |
| --- | ---: | ---: |
| 真实目标进入候选率 | 29.6% | 29.6% |
| 零 reward 组率 | 70.4% | 1.9% |
| 零 advantage 组率 | 70.4% | 2.1% |
| 平均 reward std | 0.07872 | 0.09326 |
| 精确目标保持最高奖励率 | 100% | 100% |

零 advantage 率绝对下降 68.3 个百分点，相对减少 97.02%。教师门控均值为 0.492，中位数为 0.342；真实目标的教师平均排名为 4.164，中位数为 3。

### 4.3 正式训练与最终推荐指标

| 项目 | Baseline GRPO | CGRF-H |
| --- | ---: | ---: |
| epochs / steps | 2 / 1,650 | 2 / 1,650 |
| 训练耗时 | 49,640.639 秒 | 49,744.952 秒 |
| peak allocated | 10.88 GiB | 10.88 GiB |
| peak reserved | 12.65 GiB | 12.65 GiB |
| 最终验证 KL | 0.164816 | 0.844738 |

| 指标 | Baseline GRPO | CGRF-H | 相对变化 |
| --- | ---: | ---: | ---: |
| HR@1 | **0.061769** | 0.060887 | -1.429% |
| NDCG@1 | **0.061769** | 0.060887 | -1.429% |
| HR@3 | **0.093316** | 0.092654 | -0.709% |
| NDCG@3 | **0.079940** | 0.079428 | -0.641% |
| HR@5 | 0.110743 | **0.111405** | +0.598% |
| NDCG@5 | 0.087088 | **0.087204** | +0.133% |
| HR@10 | 0.139643 | **0.142290** | +1.896% |
| NDCG@10 | 0.096283 | **0.097147** | +0.897% |
| HR@20 | 0.181337 | **0.184867** | +1.946% |
| NDCG@20 | 0.106690 | **0.107928** | +1.160% |
| HR@50 | 0.240238 | **0.249724** | +3.949% |
| NDCG@50 | 0.118380 | **0.120805** | +2.048% |

CGRF-H 的收益主要体现在 K≥5，且随 K 增大。HR@10、HR@20、HR@50 分别比 baseline 多命中 12、16、43 条测试样本；HR@1 和 HR@3 分别少命中 4、3 条。当前实现改善了候选覆盖，但更高 KL 和轻微 Top-1/Top-3 回落说明 λ 与 β 仍有优化空间。

## 5. 结果汇总与边界

原始模型、日志和 `results/` 不提交 Git。将本地原始 JSON 汇总为可审查文件：

```bash
python innovations/cgrf_hierarchical_grpo/scripts/summarize_results.py
```

输出：

```text
innovations/cgrf_hierarchical_grpo/experiment_summary.json
```

当前结论来自一个数据集、一个随机种子的单次正式训练，没有显著性检验。λ=0.2 和 0.3 只做过离线回放，未进行完整下游训练；因此本实验支持“λ=0.1 在当前固定配置下有效”，不支持“λ=0.1 是全局最优”。
