# 本地验证记录

验证日期：2026-09-14。此文件记录实际执行，不代表已在 GitHub 远端发布或完成真实模型测试。

## 执行环境

- Python：`3.13.5`。
- 平台：`Linux-6.18.44-x86_64-with-glibc2.41`。
- 核心依赖：仅 Python 标准库。
- 可选解码实测：OpenCV 4.13.0（本环境已有安装）。

## 已实际执行并通过

| 检查 | 实测结果 |
|---|---|
| `python -m unittest discover -s tests -v` | 44 项全部通过，包含本地 HTTP mock 与真实 MP4 解码测试 |
| `python -S -m unittest discover -s tests -v` | 禁用 site-packages，42 项通过，2 项可选 OpenCV 测试跳过 |
| `python -S -m wetlab_challenge demo` | 无第三方包环境下完成输入审计、基线推理、两组评测、报告输出 |
| `python -m wetlab_challenge demo --agent submission` | 未修改 starter 可直接运行，输出契约有效 |
| `--views first` 与 `--latency-s 2` | 两个消融运行入口均已实际完成 |
| 分开的 `run` 和 `evaluate` | 与 demo 的非运行耗时指标完全一致 |
| 三个额外随机种子 17/42/2027 | 额外六个开发集的审计、推理、评测全部完成 |
| 视频 `--dry-run` | 顺序解码原创 4 秒 MP4，得到 t=0/1/2/3 秒四个采样点，写出四张 JPEG 和诊断文件 |
| 本地 Ollama 协议测试 | 本地 mock HTTP 服务收到了正确的 `/api/chat`、base64 images、stream=false 请求 |
| Python 3.10 语法解析 | 通过 AST feature_version=(3,10) 检查；不是 Python 3.10 运行时验证 |
| 文档相对链接 | 未发现断链 |

附带固定模拟数据：train=24、dev_iid=8、dev_ood=8，共 40 个模拟 trial，1145 个观察时间点。不是 40 段真实湿实验视频。

## 基线实际数值：仅合成样例

| 划分 | Action Macro-F1 | Step Segment F1@0.5 | Error F1 | Timely Recall |
|---|---:|---:|---:|---:|
| dev_iid | 0.7189 | 0.2742 | 0.4000 | 0.0000 |
| dev_ood | 0.6235 | 0.2047 | 0.4706 | 0.5000 |

完整分母、事件数、逐例结果见 `examples/baseline_*.metrics.json`。小样本的及时召回波动很大；不得据此宣称泛化性能。面试官参考实现与内部结果在独立私有交付包，不包含在本公开仓库。

## 没有执行 / 没有提供

没有运行真实 GPU 视觉大模型、没有下载模型权重、没有执行 SFT/LoRA/GRPO、没有复现论文结果、没有采集或专家标注 50–100 小时真实实验数据、没有真实专家 Hidden Test、没有远端 GitHub push 或 Actions 运行。

没有在所有支持的 Python/OS 组合上做本地验证；CI 配置计划覆盖 Python 3.10–3.13。可选 OpenCV 依赖在本地使用已有安装，未验证每种全新机器上的下载与安装。

`scientific_benchmark_valid=false` 是有意保留的保守标志。即使接入声明过专家复核的真实标签，软件也不会自行认证数据足以成为正式科研 Benchmark。

## 运行日志

测试原始输出保存在 `examples/test_results_full.txt` 与 `examples/test_results_stdlib.txt`。基础流程可不联网重跑，结果中的 processing_seconds 取决于执行机器。
