# 可选：从真实视频接入视觉模型

## 已提供与未提供的内容

已提供顺序视频解码、按时间采样、有限历史窗口、JPEG 证据保存、Ollama 本地视觉 HTTP 接口、结构校验、失败诊断，以及转为基础题输入格式的代码。基础题不依赖这些功能。

**未附带真实实验视频、模型权重、专家标签或大模型训练结果。**默认 MP4 是程序绘制的解码器测试动画。`--dry-run` 明确不运行模型；HTTP 单测使用本地 mock 服务，不证明某个真实模型已经调通。

## 1. 仅验证视频解码

在单独环境安装可选依赖：

```bash
python -m pip install -r requirements-video.txt
python -m wetlab_challenge.video --video examples/decoder_fixture.mp4 --out outputs/video_smoke.jsonl --dry-run --dataset-kind synthetic_fixture
```

应生成 4 个抽样时间点、4 张 JPEG 证据和 diagnostics JSON。其 action/state/target 全部为 unknown，且明确记录 `ground_truth_created=false`。

## 2. 本地视觉模型

先安装并启动你选择的 Ollama 本地服务，加载一个**已验证支持图片输入与结构化输出**的模型。此仓库不绑定所谓“当前最优模型”，也不自动拉取可能很大的权重。模型名通过 `--model` 显式传入。

下面的 `YOUR_INSTALLED_VISION_MODEL` 是需要替换的配置占位符，不是实际模型名；`local_authorized_video.mp4` 也需要替换为已授权本地文件。

```bash
python -m wetlab_challenge.video --video local_authorized_video.mp4 --model YOUR_INSTALLED_VISION_MODEL --out local_real/inputs/pilot.jsonl --sample-hz 1 --max-seconds 30 --history-frames 3
```

默认仅连接 `http://127.0.0.1:11434/api/chat`，不需要商业 API Key。使用其他机器作为服务需要明确设置 `--endpoint`；非本地主机还需要 `--allow-remote`，因为这会传输图像。不要未经授权把实验室数据上传到外部服务。

协议可通过 `--protocol my_protocol.json` 传入，格式沿用 `fixtures.protocol()` 输出。当前词表只适用于题目中的单轮转移流程，不支持任意 SOP 的自动词表映射。任意真实视频上跑出 JSON，不等于已经理解了该实验。

## 3. 流式因果性与时间

逐帧顺序解码，不把未来图片传给模型。每次只发送最近 `history_frames` 张已采样图片，默认 3 张；不是整段视频推理。

默认使用解码器给出的时间戳。遇到非单调/无效时间戳直接报错；仅在确认输入是恒定帧率时使用 `--assume-cfr` 回退到 frame_index/fps。当前实测覆盖附带的 CFR 测试 MP4，不保证所有容器、VFR 视频及 OpenCV 后端一致。

该脚本是离线构造因果视觉特征缓存，不是已经部署的实时监控服务。视频先完成缓存再跑下游评测时，VLM 运行耗时不会自动成为基础题指标中的延迟。真实时效需要端到端在线测量。

`duration_s` 使用最后一个采样点加采样间隔，是回放网格约定；真实标注导入前必须核对完整媒体时长、时间零点、裁剪偏移与最后采样窗口。

仅提供单视频适配器。多视角原始视频扩展需自行提供时间同步关系并合并成 `observations.views`；不能未经验证假设两个文件的第 100 帧是同一时刻。

## 4. 可观察性、失败与标签

模型返回 JSON 不满足词表、概率范围/归一化、响应格式等要求时，该采样点保留为 unknown，并在 diagnostics 记录失败。不可忽略失败帧，只在成功输出上报告分数。自报 confidence 不是经过校准的概率。

模型输出**只能作为观测或待复核标注建议**，不能直接保存为 GT。接入真实评测时另行提供 `local_real/labels/pilot.jsonl`，用专家标注和复核补齐 action、step、state、event。真实标签需要 `annotation_source=expert_reviewed`、`reviewer_id`、`review_record`；有 PNR 的事件还需要专家依据。不得伪造这些字段。

默认 episode 的 `operator_id/lab_id=unassigned` 是为了推理接口可用，不可据此做跨操作员或跨实验室泛化评估。正式数据整理时必须绑定真实但脱敏的分组 ID。

## 5. 对真实视觉研究能力的进一步评价

基础题通过后，建议招聘方提供一份所有候选人一致的授权小样本、统一算力和专家标签，再开展付费或正式合作形式的视觉模型验证。控制模型、采样预算和训练数据后，比较 raw-video baseline、视觉特征+时序方法，以及必要时的小规模适配。

不应要求候选人为笔试自行获取付费视频、授权、显卡或商业 API，也不应把未经专家确认的 VLM 标注作为录取评分标准。

官方接口依据见 [REFERENCES.md](REFERENCES.md)。
