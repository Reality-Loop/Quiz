# RealityLoop WetLab-Causal Mini Challenge

## 科研实习考核：从多视角视觉观测流到实验步骤理解与及时偏差提醒

**研究问题：在只能使用当前及历史信息的情况下，如何利用有噪声的多视角视觉观测，理解实验执行过程，并在有证据、仍来得及的时候提醒偏差？**

面向优秀本科生、硕士生、博士生。建议净投入 **6–8 小时，上限 10 小时**；收到题目后 7 个自然日内提交即可。达到时间上限可以提交部分完成版，清楚解释卡点。无需付费 API、私人 GPU、下载完整数据集或训练大模型。读题、实验和报告均计入净投入时间。

> **重要边界**：本仓库提供的是 40 个**程序生成的、有噪声的结构化观测流样例**，用于招聘与软件测试；不是 40 段真实实验视频，不是 VLM 实测输出，也不是专家标注的科研 Benchmark。`dev_ood` 只是模拟条件变化。随仓库附带的 MP4 仅测试解码器，不具有实验语义。真实湿实验视频接入属于可选扩展，不是基础题录取门槛。

### 研究背景

RealityLoop 正在探索面向生命科学、化学、环境、材料湿实验室的视频理解与过程评估：从物体、原子动作和状态到 SOP 步骤、偏差发现、错误预判与及时提醒。长期计划结合公开数据与多实验室一手数据开展基准建设和模型后训练；最终测试集以专家标注和复核为主。

本次只选取其中一个可以在有限时间完成的小问题：**观测有噪声、机位会缺失时，如何做可靠的因果时序判断？**不是要求你完成公司整个研究项目。

## 1. 一分钟了解你需要做什么

在 `submission/agent.py` 中实现 `Agent.reset(context)` 和 `Agent.update(observation)`。系统按时间顺序逐次提供观测，每次返回：

- 当前原子动作、可观察状态代理和 SOP 步骤，允许输出 `unknown`；
- 是否发生 `wrong_target`（目标容器偏差）或 `missing_mix`（要求混匀却在未完成时结束）；
- 对判断和告警给出已看到的证据 ID。告警时间由运行器填写，不由候选人回填。

完成一个有依据的改进，与基线比较，做两组消融，写一份 2–3 页报告。**只运行 starter 不构成完成答卷；无提升但分析可信的结果也有价值。**

完整题面见 [docs/TASK.md](docs/TASK.md)，研究报告模板见 [docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md)。

## 2. 直接运行：无需安装第三方依赖

要求 Python 3.10 或以上。在仓库根目录运行；系统将 Python 命名为 `python3` 时替换命令中的 `python`。

```bash
python -m unittest discover -s tests -v
python -m wetlab_challenge demo
```

第二条命令使用仓库附带的数据，运行 baseline、验证输入、评测 `dev_iid` 与 `dev_ood`，生成：

```text
outputs/
  REPORT.md
  audit.json
  run_environment.json
  dev_iid.predictions.jsonl
  dev_iid.metrics.json
  dev_ood.predictions.jsonl
  dev_ood.metrics.json
```

结果中明确标记 `synthetic_fixture`；不得将该结果描述为真实实验识别准确率。`processing_seconds` 是本机实测运行时间，时效指标默认使用视频时间，不代表真实端到端告警延迟。

## 3. 实现与评估自己的方法

```bash
# 运行你在 submission/agent.py 中的实现
python -m wetlab_challenge demo --agent submission --out outputs/submission

# 单视角消融；runner 会真正移除 cam_2，而不只是改配置描述
python -m wetlab_challenge demo --agent submission --views first --out outputs/submission_first

# 模拟固定额外延迟：它是显式假设，不是测得的部署延迟
python -m wetlab_challenge demo --agent submission --latency-s 2 --out outputs/submission_delay2
```

也可以分别运行推理与评测。**推理命令不加载 labels；评测单独执行。**

```bash
python -m wetlab_challenge run --agent submission --split dev_iid --out outputs/my_predictions.jsonl
python -m wetlab_challenge evaluate --split dev_iid --predictions outputs/my_predictions.jsonl --out outputs/my_metrics.json
python -m wetlab_challenge audit
```

重新生成或使用新随机种子时写入独立目录，避免改动统一开发集：

```bash
python -m wetlab_challenge generate --data scratch_data --seed 8123
python -m wetlab_challenge demo --data scratch_data --out outputs/scratch
```

主结果必须使用随仓库提供的固定开发集。新的随机种子只是额外测试，**不等于真实的未知实验室评估**。

## 4. 数据与输入

| 划分 | 数量 | 用途 |
|---|---:|---|
| train | 24 个模拟 trial | 了解任务、调参、检查协议规则 |
| dev_iid | 8 个模拟 trial | 固定开发评测 |
| dev_ood | 8 个模拟 trial | 模拟机位噪声变化、站点和操作员 ID 变化 |

每个 trial 将两个机位放在同一个 episode 中。按 trial 分组，不把同一次操作的两路视角拆到不同划分。输入与标签分目录，所有公开样例均为开发数据；**不附带真正的 Hidden Test**。

`action_scores / state_scores / target_scores` 是有噪声的模拟视觉模块输出，不是干净标注。`protocol` 只描述应当做什么，不说明实际上做了什么。观测缺失不等于动作没发生；模型置信分数也不是校准后的真实性概率。

详细字段、样例、指标定义与边界见 [docs/DATA_AND_METRICS.md](docs/DATA_AND_METRICS.md)。

## 5. 评测内容

自动评测输出动作 Macro-F1、步骤准确率与 Step Segment F1@0.5、已知状态标签上的准确率与覆盖率、事件级 Precision/Recall/F1、每小时误报、可干预事件中的及时发现率、提前预警率，以及配对成功事件的延迟与提前量。

**事件检测与及时提醒分开计算。**晚于不可逆点但仍在事件窗口内的报警，可以算发现偏差，但不算及时提醒。`t_pnr=null` 不强行补时间；`t_detectable >= t_pnr` 的事件单独报告，不能用来惩罚“本来就无法及时看到”的模型。重复报警不能刷高召回。

所有时间边界与空分母处理在文档和单元测试中给出。脚本只检查证据引用存在且来自历史，**不自动证明证据在语义上支持结论**，研究报告必须人工复核失败案例。

## 6. 评分与提交

| 维度 | 分值 | 重点 |
|---|---:|---|
| 问题定义与研究判断 | 25 | 能区分观察、推断、未知与可验证结论 |
| 方法实现 | 25 | 因果时序、多视角、状态与事件处理是否合理 |
| 实验与评测 | 25 | 公平对比、两组消融、误差分析、指标边界 |
| 可复现与工程质量 | 20 | 一键运行、记录环境、测试、明确失败行为 |
| 性能与成本意识 | 5 | 结果有解释，报告延迟/算力/资源成本 |

不按学校排名或学历给分，不要求已有论文。博士或有较强研究经历的候选人，可以用更深入的假设和实验解释体现能力，不要求完成更多免费开发工作。

提交修改后的仓库或私有仓库访问方式、固定 commit SHA、原始预测与指标、报告、投入时间和 AI 工具使用说明。**不要在公开 Issue/PR 上传简历、联系方式、未授权视频、客户 SOP 或任何密钥。**投递通过原招聘私信渠道完成，详见 [docs/SUBMISSION.md](docs/SUBMISSION.md)。

## 7. 真实视频扩展（可选）

仓库另有顺序抽帧与本地 Ollama 视觉接口；不附带大模型权重、不保证任意模型兼容、不把模型输出写成 Ground Truth。基础题完全不依赖此扩展。

```bash
python -m pip install -r requirements-video.txt
python -m wetlab_challenge.video --video examples/decoder_fixture.mp4 --out outputs/video_smoke.jsonl --dry-run --dataset-kind synthetic_fixture
```

这一条仅测试真实 MP4 解码、证据保存和输入格式，**没有运行视觉大模型**。接入已授权真实视频的方法与限制见 [docs/REAL_VIDEO.md](docs/REAL_VIDEO.md)。

## 8. 文件结构

```text
wetlab_challenge/  数据生成、契约校验、弱基线、流式运行器、指标、可选视频接口
submission/       你需要实现的 Agent；初始状态继承弱基线，可直接运行
tests/            单元/回归测试；无 OpenCV 时跳过可选视频测试
data/inputs/      公开模拟观测流
data/labels/      公开模拟开发标签；不是隐藏答案
examples/         解码测试视频、固定基线结果
.github/workflows/ci.yml  CPU CI 配置（远端执行状态另行查看）
docs/             题目、标注与评测规范、真实视频接入、报告与提交模板
VALIDATION.md     交付环境实测记录及未实测项
```

[研究参考与数据授权边界](docs/REFERENCES.md) · [完整题面](docs/TASK.md) · [研究路线背景](docs/PROJECT.md)

本仓库原创代码和合成测试数据采用 MIT 许可；第三方数据、视频、模型及论文不包含在此授权内。招聘结果依据完整材料与交流判断，本考核不构成录用、论文接收或升学承诺。
