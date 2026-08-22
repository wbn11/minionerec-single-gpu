# MiniOneRec 单 A6000 复现

本仓库用于参考 MiniOneRec 官方论文、代码与公开数据格式，独立完成一套可在单张 NVIDIA A6000 上运行的生成式推荐复现。

当前阶段只建立工程骨架和复现契约。尚未安装训练依赖、下载数据、运行模型或加入 CGRF 创新。

## 项目原则

- 先完成官方主链路复现，再开始创新。
- 保持数据切分、Semantic ID、SFT 目标、官方奖励和评测协议的算法语义。
- 单卡改造仅解决并行方式、显存占用、batch 组织和运行效率问题。
- 每个实现或实验步骤先说明原理、输入输出、配置和验收标准，再由项目所有者确认执行。
- 不在 README 中填写未经实际运行验证的指标。

## 计划主链路

```text
Amazon 序列与商品文本
        |
        +--> Semantic ID / RQ-VAE --> SID Catalog --> Trie
        |
        +--> history -> target --> Qwen2.5 SFT --> 官方 GRPO
        |                                      |
        +--> SASRec 教师 ----------------------+
                                               |
                                               v
                                      约束生成与统一评测
```

## 目录

- `src/minionerec/`：可复用的核心实现。
- `configs/`：按阶段版本化的配置。
- `scripts/`：只负责读取配置并调用核心包的薄入口。
- `tests/`：合成数据单元测试与小型集成测试。
- `docs/reproduction/`：逐步复现说明、范围和单卡差异。
- `third_party/UPSTREAM.md`：官方来源及固定版本。
- `environment/`：A6000 项目私有环境的说明、创建脚本与依赖锁文件。
- `.venv-a6000/`：A6000 上实际安装 Python 包的位置，不提交 Git。
- `artifacts/`：本地数据、模型和完整预测，不提交 Git。
- `results/`：可以审阅和提交的小型指标、表格与 manifest。

详细模块边界见 `docs/architecture.md`。

## 当前状态

- [x] 建立复现范围与工程骨架
- [ ] 固定并审计官方源代码
- [ ] 建立单卡环境
- [ ] 数据与 SID 协议
- [ ] RQ-VAE / Semantic ID
- [ ] 0.5B SFT
- [ ] 约束生成与官方评测
- [ ] SASRec 与官方 GRPO
- [ ] 复现验收
- [ ] 创新阶段
