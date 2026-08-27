# RTX A6000 服务器环境

服务器项目目录为 `/home/user/wbn/minionerec`，虚拟环境为 `/home/user/wbn/minionerec/.venv-a6000`。虚拟环境只在服务器维护，不从本地复制或覆盖。

## 已验证配置

| 项目 | 版本或容量 |
| --- | --- |
| GPU | NVIDIA RTX A6000，49140 MiB，compute capability 8.6 |
| Driver | 565.57.01 |
| `nvidia-smi` CUDA 上限 | 12.7 |
| Python | 3.11.5 |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA runtime | 12.4 |
| Transformers | 4.57.1 |
| TRL | 0.24.0 |
| Accelerate | 1.10.1 |
| BitsAndBytes | 0.48.1 |
| NumPy | 1.26.3 |

Driver 的 CUDA 12.7 表示驱动兼容上限；PyTorch 自带 CUDA 12.4 runtime，两者可以正常配合。

完整依赖快照保存在 [locks/a6000-py311-cu124.txt](locks/a6000-py311-cu124.txt)。

## 启动与验证

在服务器项目根目录执行：

```bash
source .venv-a6000/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
python -m pip check
```

预期 GPU 为 `NVIDIA RTX A6000`，`python -m pip check` 不报告损坏依赖。正式训练前使用 `nvidia-smi` 确认 GPU 0 的空闲显存。

运行时应显式指定：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ...
```

两个 Qwen 模型已经放在 `artifacts/models/`，训练过程不依赖网络。

## 同步规则

- 本地修改 `scripts/`、`src/` 和文档后，再手动同步到服务器。
- 不覆盖服务器 `.venv-a6000/`。
- 数据、模型和实验结果在 `artifacts/`、`results/` 中按相同相对路径保存。
- 所有运行都限制在 `/home/user/wbn/minionerec`，不使用 `sudo`。
