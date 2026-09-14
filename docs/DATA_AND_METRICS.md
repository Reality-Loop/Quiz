# 数据契约、因果性与指标

## 1. 基础题的范围

默认数据是程序生成的结构化观测流，**没有真实画面、专家标签或真实实验室来源**。每个 episode 是一轮模拟液体转移，两路摄像机的抽象输出已经按秒对齐。动作与状态分数含随机错误。

`state` 仅是可观察操作状态的代理：`empty / loaded / transferred / mixed / unknown`。它不是浓度、纯度、无菌性、反应完成度或最终实验结果。真实任务必须区分直接可见状态、依据 SOP 的推断状态、不可判定状态。

正常动作词表：`background, attach_tip, aspirate, dispense, mix, eject_tip`。预测可加 `unknown`。步骤：准备 S1、转移 S2、混匀 S3、结束 S4，另有 background/unknown。S3 是否必需由 SOP 的 `mix_required` 决定。`target_id` 是预期目标，不是实际转移结果。

## 2. 输入与输出

每个 JSONL 行是一整个 episode，但运行器逐帧调用 agent；不会把整个行传入 agent。

`reset(context)` 只接收 `episode_id` 和 `protocol`。不会传入标签、未来观察、总时长、实验室/操作员 ID 或数据划分。

`update(observation)` 输入示意：

```json
{
  "t": 12.0,
  "views": [{
    "camera_id": "cam_1", "source_t": 12.0,
    "evidence_id": "opaque_trial:cam_1:0012", "visible": true,
    "action_scores": {"aspirate": 0.7, "dispense": 0.2, "unknown": 0.1},
    "state_scores": {"loaded": 0.8, "unknown": 0.2},
    "target_scores": {"tube_a": 0.2, "tube_b": 0.7, "unknown": 0.1}
  }]
}
```

返回值示意（是否足以报警必须由你的方法判断）：

```json
{
  "action": "aspirate", "state": "loaded", "step_id": "S2",
  "evidence_ids": ["opaque_trial:cam_1:0012"],
  "alerts": [{
    "type": "wrong_target", "scope_id": "tube_b",
    "evidence_ids": ["opaque_trial:cam_1:0012"]
  }]
}
```

分数对象允许稀疏键，值须为 [0,1] 内有限数且总和为 1；没有观察可用 `unknown:1`。证据 ID 在 episode 内唯一。`source_t <= t`；空 views 合法。支持返回多个不同类型告警。输出结构故意不允许候选人设置告警时间。

## 3. GT 事件与时间

事件字段：

```text
type, scope_id, t_occur, t_detectable, t_pnr, t_end, severity, evidence_ids
```

`t_occur`：偏差开始发生的时间。`t_detectable`：依据允许的输入，第一次存在充分可见证据的时间，可早于或晚于发生时间。`t_pnr`：在指定干预动作及任务条件下，无法再避免该错误后果的时点。`t_end`：允许将报警关联到该事件的最终时点，而不必等同所有后果消失的时点。

`t_detectable` 与 `t_pnr` 均允许显式 `null`；不能用 0、视频结尾或模型猜测代替未知值。真实标签需要专家复核，PNR 必须附 `pnr_rationale`。脚本检查声明和字段，不能认证专家资质或复核真实性。

**默认数据中的 PNR 是人为设置的软件测试时点，不能引用为移液实验的科学结论。**`missing_mix` 的 PNR 默认未知：结束当前操作不自动意味着任何科学后果已经不可逆。

真实任务还需要记录场景、对象、干预方式、标注人、复核人、分歧处理、时间容差及可观察性。定义依赖输入模态：只看视频与同时读取设备日志的 `t_detectable` 不能混为一谈。

## 4. 视频时间与实际延迟

默认 `t_emitted = t_observed`，用于比较时序决策需要多少视频证据。设置 `--latency-s L` 后为 `t_emitted = t_observed + L`。L 是显式固定附加延迟假设，**没有模拟队列、并发、解码、网络或真实用户收到通知的时间**。

`processing_seconds` 是单次 agent 调用的本机耗时，单独统计，未算入上述视频时间指标。通过视频适配器预计算的 VLM 特征尤其不能被当成“零延迟视觉推理”。真实系统验收应在在线运行中测量图像采集→排队→推理→决策→通知完整链路，并包含积压和失败。

## 5. 帧/片段指标

**动作 Macro-F1**：对固定六类动作分别计算 F1，再对定义有效的类求平均；无该类真值且无该类预测时该类为 null，并报告有效类别数量。`unknown` 预测在有标签帧上记为未识别，不能靠拒答提高准确率。GT action 为 unknown 的帧不计入此指标，另报忽略数量。

**步骤准确率**：按抽样时间点比较 step_id。该值是采样点准确率，不自动等同连续时间加权准确率。

**步骤 Segment F1@0.5**：同类连续片段合并，采用 [start,end) 区间，以 tIoU≥0.5 建立边，做一对一最大匹配。background/unknown 预测不作为正片段；不会把被背景隔开的同类片段错误合并。当前实现要求 GT 步骤完整，无 unknown 区间；真实数据应选择完整标注的评测子集，不能悄悄把未知区间当背景。

**状态**：在已知 GT 状态的时间点上同时报告准确率与回答覆盖率。模型输出 unknown 算未回答，不因省略不确定样本获得虚假高准确率。

## 6. 事件指标

预测按事件类型和 scope_id 匹配，报警须落在 `[t_detectable, t_end]`；`t_detectable` 未知时检测指标暂用 `[t_occur, t_end]`。同一预测最多匹配一个 GT，同一 GT 最多被一个预测计为 TP。其余报警为 FP，未匹配 GT 为 FN。允许发生前报警，但不能通过在尚无可见证据时“猜中未来”获分。

重复告警属于 FP。告警合并、冷却、同一事件的生命周期管理是方法的一部分，不能由评测器替候选人消除重复误报。基础题每轮每类偏差至多一次；多轮真实流程需新增稳定的 incident 身份与事件生命周期。

`Error F1 = 2TP / (2TP+FP+FN)`。

`False alarms/hour = FP / (sum episode durations / 3600)`。

仅定义以下干预机会集合：

```text
G_opportunity = {g | t_detectable != null AND t_pnr != null AND t_detectable < t_pnr}
Timely Recall = 最大一对一及时匹配数 / |G_opportunity|
```

及时边要求 `t_detectable <= t_emitted < t_pnr`，且满足检测事件匹配规则。**报警恰好在 PNR 时不算及时。**使用独立的及时匹配，避免普通匹配分配改变及时结果。

提前预警只在 `t_detectable < t_occur` 的事件上计算，要求 `t_emitted < t_occur`。及时发现并不一定是提前预警。

延迟 `t_emitted - t_detectable` 只在匹配成功且 detectable 已知的事件上求均值；提前量 `t_pnr - t_emitted` 只在及时匹配成功事件上求均值。两者有成功样本选择偏差，必须和召回率、分母、漏报一起阅读。

所有零分母返回 null，而非 0 或 1。没有正例的样本不能因此获得“100% 错误检测能力”。提供的样本很小，每小时误报只是归一化计数，不是可靠的部署误报率估计。

## 7. 防泄漏与证据边界

按完整 trial 分组，同一试验的全部视角同属一个划分。审计检查 trial 重复、去除 ID 后的完全重复观测、模拟 unseen-lab 与训练 lab ID 是否重叠。它不做视频近重复检测，也不证明操作员、设备、协议等其他属性完全独立。

`dev_ood` 同时改变模拟操作员和站点，是组合扰动，不是分别控制变量的真实 Unseen Lab/Unseen Operator 测试。正式项目应构建独立的 Leave-One-Lab-Out、Unseen Operator、Unseen Protocol/Equipment 等划分，并报告混杂因素。

Prefix API 能减少误用未来信息，但 Python 进程不是安全沙箱。候选代码理论上仍能自行打开本地文件。真正的隐藏测评必须在隔离环境中仅挂载输入、禁网、限制文件访问，标签与评分器留在另一个受信进程；不能把公开生成器换一个 seed 就叫不可破解的 Hidden Test。

验证证据 ID 存在、时间合法，只能说明引用正确，不能证明动作确实可见、主体关联正确或实验判断合理。真实结果必须人工复核。
