# 运行脚本

这里仅放薄命令入口。脚本应当读取 YAML、调用 `src/minionerec/` 中的实现并保存 manifest，不得包含大段模型、奖励或指标逻辑。

预计按复现顺序增加：

1. `inspect_upstream.py`
2. `prepare_data.py`
3. `train_rqvae.py`
4. `train_sft.py`
5. `evaluate.py`
6. `train_sasrec.py`
7. `train_grpo.py`
