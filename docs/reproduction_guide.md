# MiniOneRec 单卡复现命令

本文只记录已经实际跑通的 `Industrial_and_Scientific` 主流程。所有命令均在服务器 `/home/user/wbn/minionerec` 执行。

## 1. 环境与路径

```bash
cd /home/user/wbn/minionerec
source .venv-a6000/bin/activate

MINIONE_CATEGORY=Industrial_and_Scientific
MINIONE_PROCESSED=artifacts/data/processed/amazon18/Industrial_and_Scientific
MINIONE_FINAL=artifacts/data/final/amazon18
MINIONE_FINAL_NAME=Industrial_and_Scientific_5_2016-10-2018-11
```

基础模型必须已经存在：

```text
artifacts/models/Qwen3-Embedding-4B/
artifacts/models/Qwen2.5-1.5B-Instruct/
```

## 2. Amazon18 数据处理

输入是解压后的 review 和 metadata JSONL：

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

验收：7,694 个用户、3,686 个商品、53,018 条过滤后交互；训练/验证/测试样本分别为 36,259、4,532、4,533。

## 3. 商品 embedding

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

验收：输出形状 `(3686, 2560)`、类型 `float32`，没有 NaN、Inf 或全零行。

## 4. RQ-VAE 与 Semantic ID

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

RQ-VAE 使用 32 维 latent 和三层 256 码本。碰撞消解后得到 3,673 个唯一 SID，剩余碰撞冗余为 13。

## 5. 转换最终 CSV

```bash
python scripts/convert_dataset.py \
  --data-dir "$MINIONE_PROCESSED" \
  --dataset-name "$MINIONE_CATEGORY" \
  --output-dir "$MINIONE_FINAL"
```

输出位于 `$MINIONE_FINAL/{train,valid,test,info}/`。

## 6. SFT

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

实际训练在 epoch 4.5 早停并回载 step 1,872 的最佳权重。最终模型位于 `results/sft/Industrial_and_Scientific/final_model/`。

## 7. SFT 评测

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/evaluate_sft.py \
  --model-path results/sft/Industrial_and_Scientific/final_model \
  --test-file "$MINIONE_FINAL/test/$MINIONE_FINAL_NAME.csv" \
  --info-file "$MINIONE_FINAL/info/$MINIONE_FINAL_NAME.txt" \
  --output-file results/evaluation/Industrial_and_Scientific/sft_metrics.json \
  --batch-size 8 \
  --num-beams 50 \
  --device cuda:0
```

## 8. ranking GRPO

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

验收：完成 2 epochs、1,650 steps，合法候选率为 100%。最终模型位于 `results/grpo/Industrial_and_Scientific/final_model/`。

## 9. GRPO 评测

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/evaluate_sft.py \
  --model-path results/grpo/Industrial_and_Scientific/final_model \
  --test-file "$MINIONE_FINAL/test/$MINIONE_FINAL_NAME.csv" \
  --info-file "$MINIONE_FINAL/info/$MINIONE_FINAL_NAME.txt" \
  --output-file results/evaluation/Industrial_and_Scientific/grpo_metrics.json \
  --batch-size 8 \
  --num-beams 50 \
  --device cuda:0
```

最终参数、资源占用和 SFT/GRPO 指标对比见 [reproduction_results.md](reproduction_results.md)。重新运行某个训练阶段前，应先明确备份或移走该阶段已有输出目录，程序不会静默覆盖正式结果。
