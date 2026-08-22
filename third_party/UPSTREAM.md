# 上游来源

本项目参考以下公开资料，但项目根目录不直接复制官方仓库代码。

## MiniOneRec

- 论文：https://arxiv.org/abs/2510.24431
- 官方仓库：https://github.com/AkaliKong/MiniOneRec
- 模型与预处理数据：https://huggingface.co/kkknight/MiniOneRec
- 审计基线 commit：`0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`
- 审计日期：2026-08-22

## OneRec 背景

- 初版：https://arxiv.org/abs/2502.18965
- 生产技术报告：https://arxiv.org/abs/2506.13695
- OneRec-V2：https://arxiv.org/abs/2508.20900

## 使用原则

- 上游代码作为算法行为和数据格式的参考。
- 每个移植模块必须在后续文档中记录官方文件、我们的模块和行为差异。
- 保留原项目的版权和许可证要求；在审查上游许可证前不复制大段源码。
- 上游 commit 变动不会自动进入复现基线。
