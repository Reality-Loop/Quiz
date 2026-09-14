# 研究参考与接口依据

核查日期：2026-09-14。以下仅保留本题用到、能够定位到作者/官方来源的项目；不是完整 Related Work，也不声称某模型为当前最优。

## FineBio

- 作者论文：[FineBio: A Fine-Grained Video Dataset of Biological Experiments with Hierarchical Annotation](https://arxiv.org/abs/2402.00293)
- 官方仓库：[aistairc/FineBio](https://github.com/aistairc/FineBio)
- 关系：分层协议/步骤/原子操作、物体位置及操作状态；作者摘要描述 32 位参与者、14.5 小时实验时长的多视角数据。与本题层级化理解有关，不为本题合成数据背书。
- 数据入口：官方 README 要求提交签署的许可协议，获批后取得视频/元数据/标注；视频使用限制与仓库 MIT 代码许可不同。README 于 2026-09-10 公布协议更新，限非商业研究/开发用途；商业使用需另行沟通。不可把 README 可见理解为视频可随意打包公开。
- 本次读取的 README blob SHA：`2ee45cd603b96c5b32f95a464c1d3c7f006bc540`。许可应以实际申请时最新版为准。

## ProBio

- 官方项目：[ProBio](https://probio-dataset.github.io/)
- 官方数据说明：[ProBio Dataset](https://probio-dataset.github.io/dataset.html)
- 论文：[NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81c7202dbd3cd3006b35a58a076195c0-Abstract-Datasets_and_Benchmarks.html)
- 关系：协议引导的分层多模态实验理解；官方数据页列出 13 类实验及 180.6 小时数据，但不同论文时长统计口径不可未经对齐直接比较。
- 项目页面标示数据使用 CC BY-NC-SA；不得因此推断可用于任意商业模型或重新授权。当前仓库没有打包 ProBio 视频。

## ExpVid

- 作者论文：[ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](https://arxiv.org/abs/2510.11606)
- 关系：作者将任务组织为细粒度感知、过程理解、科学推理，并使用视觉驱动的标注与多学科专家验证。本题借鉴分层评估思路，但重点另加因果时间限制与及时提醒，不复制其数据或声称复现其结果。

## 本地模型接口

- [Ollama Chat API](https://docs.ollama.com/api/chat)：`POST /api/chat`，显式 `stream=false`。
- [Ollama Vision](https://docs.ollama.com/capabilities/vision)：REST 图片输入使用 base64 编码。
- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)：`format` 接收 JSON Schema。接口支持不等于每个视觉模型都兼容、或语义输出准确；需要分别实测。

## 标准库测试

- [Python unittest](https://docs.python.org/3/library/unittest.html)：基础题单元测试无需 pytest。实际执行环境与结果见根目录 VALIDATION.md。

本仓库未提供 FineBio、ProBio、ExpVid 或其他第三方视频的镜像、账户凭证或下载绕过手段。候选人的回答不需要获取这些受限资产。
