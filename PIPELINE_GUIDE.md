# OSS Ego Pipeline 当前实现说明

## 1. 项目目标

本项目面向 `oss://ccm-ego/` 中结构不统一的 Ego 数据。输入一个 recording 或 episode 的 OSS 路径，在线读取视频和元数据，抽取带时间戳的视觉证据，生成动作链、对象/场景摘要、难度初判和人工复核队列。

`recording` 是难度判定主单位。长 recording 不自动拆成多个难度样本；原始 manifest 明确给出的 `episode` 边界才可作为独立任务单元。同一次执行的多机位只计一次时长，并记录机位数量。

## 2. 当前运行环境

- OSS 桶挂载：`oss://ccm-ego/` → `/mnt/ego/`
- GPU：PPU-ZW810E，约 96 GiB 显存
- Python：3.12
- PyTorch：2.9.0
- Transformers：5.17.0
- Qwen3-VL 权重：`/mnt/workspace/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master`
- 视频读取：PyAV/FFmpeg

原始视频只从挂载路径流式读取。脚本可以保存选中的帧和窗口检查点，但不保存原视频、OSS 凭证或原始传感器文件。

## 3. 当前代码入口

### `ego_pipeline_mvp.py`

消费 JSON manifest，按扩展名建立对象索引，检查 `meta.json` 和 `calibration.json`，输出 MVP 报告骨架。它还没有接入视频和 VLM。

### `qwen_vl_pipeline.py`

当前可运行的视频语义验证入口：

1. 读取视频容器头，取得时长、帧率和视频流。
2. 以 2 秒为默认采样目标。
3. 单文件超过 300 帧时，在 300 帧预算内均匀覆盖全程；短视频默认至少取 4 帧并包含尾帧。
4. 每 16 帧组成一个 VLM 窗口，默认重叠 2 帧后逐窗调用 Qwen3-VL-4B-Instruct。
5. 每张图片附带真实视频时间戳；模型不能使用帧序号代替秒数。
6. 合并所有窗口的摘要、对象、动作段、unknown 和高难候选。
7. 对重叠窗口的重复动作做 recording 级合并；若模型提供 `task_stage`，同时区分任务阶段数 `N` 与原子动作数。
8. 将无效 JSON、无效时间、候选高难事件和聚合待复核事项写入 `review_queue`。
9. 输出 `stage_timings_s`、逐窗口耗时、显存快照和失败原因，便于定位性能瓶颈。
10. 自动写出 JSONL 事件日志以及 Markdown/HTML 可视化报告；报告包含路径、时长、场景、任务适用性、语义摘要、T/N/H、level、判断依据、动作证据和复核队列。

运行示例：

```bash
python3 qwen_vl_pipeline.py \
  --model /mnt/workspace/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master \
  --video /mnt/ego/Deepreach/dr-3camera-deliverable/lerobot_v2/compose_flashlight/videos/chunk-000/observation.images.head/episode_000000.mp4 \
  --out outputs/flashlight.json \
  --interval 2 --max-frames 300 --window-frames 16
```

批量处理时使用 `scripts/run_qwen_batch.py`；它只加载一次模型，并在每条结果中记录共享模型加载耗时：

```bash
python3 scripts/run_qwen_batch.py \
  --manifest demo_sample_list.json \
  --model /mnt/workspace/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master \
  --out outputs/qwen_batch.json
```

参数：

- `--interval`：目标采样间隔，默认 2 秒。
- `--max-frames`：单 recording 的视觉帧预算，默认 300。
- `--window-frames`：单次 VLM 请求的帧数，默认 16。
- `--window-overlap`：相邻 VLM 窗口重叠帧数，默认 2。
- `--min-frames`：短视频最少均匀采样帧数，默认 4。
- `--max-duration`：超过该秒数直接跳过，默认 600 秒；临时测试可显式增大。
- `--frames-cache`：可选，只缓存选中帧和窗口 JSON，默认不启用。
- `--log`：可选，JSONL 事件日志；默认 `<out>.log.jsonl`。
- `--report-md` / `--report-html`：单视频报告路径；默认 `<out>.report.md` 和 `<out>.report.html`。
- `--attn-implementation`：可选 `eager`、`sdpa` 或 `flash_attention_2`；当前 PPU 环境的 SDPA 路径不稳定时使用 `eager`。

时间字段说明：`end_to_end_s` 是从脚本开始到结束的墙钟时间；`model_inference_s` 是所有 VLM 窗口推理时间之和；`script_runtime_excluding_inference_s` 是前者减去推理时间；`script_overhead_excluding_model_s` 进一步扣除模型加载。相应的 `*_to_video_ratio` 都以视频实际时长为分母。批量模式另记录一次性的 `shared_model_load_s`。

批量模式默认在 `<out stem>_reports/` 为每条视频生成 `.report.md` 和 `.report.html`，也可用 `--report-dir` 指定目录；`--log` 同样默认为 `<out>.log.jsonl`。

覆盖判定采用脚本计算的 `semantic.coverage`：`sampling_complete`、`window_success_ratio`、`schema_valid_ratio`、`evidence_grounded_ratio` 和综合 `score`。模型窗口只输出 `window_coverage`，不再声称看到了整条视频。动作的开始/结束时间由 `start_frame_id`、`end_frame_id` 和真实抽帧时间派生，避免窗口局部时间漂移。批量缓存目录使用序号加文件名，避免多个 `left_cam_left.mp4` 互相复用缓存；CUDA OOM 时会清空缓存并用更小窗口重试一次。

## 4. VLM 输出

### 原子动作词表（来自需求 PDF）

需求文档将原子动作分为 8 类。附件正文实际列出了 61 个英文动作名，虽然文档标题写作“85 类”；因此当前实现只把这 61 个正文动作作为受控词表，另设 `others` 兜底，不自行补齐未出现的 24 类。

```text
基础运动：Move Reach Retract Lift Lower Stop Drag
抓取释放：Grasp Pick Release Drop Hold Handover
放置定位：Place Position Rotate Tilt Flip Stack
施力接触：Push Pull Press Squeeze Touch Tap Strike Hammer Shake Rub Crush Snap
组装连接：Insert Screw Unscrew Attach Detach Switch Open Close
柔性液体：Pour Stir Fold Unfold Braid Tie Wrap Thread Peel Spread
工具与状态：Cut Wipe Clean Sweep Scrub Dispose Paint
认知管理：Identify Verify Group Organize Wait
兜底：others
```

`canonical_action` 必须来自此表；`raw_action` 保留模型原始短语。词表之外的动作不能悄悄改写成近似动作。需求文档还要求动作时间边界、示能框、2D 轨迹和导航 3D 框；当前视频 MVP 只实现动作候选和时间证据，尚未实现这些标注结构。

### 场景与任务词表

需求文档新增的场景/任务覆盖包括：开放式厨房、餐厅/用餐区、卧室、书房/办公桌、洗手间/卫浴区、洗衣房、玄关/门厅、阳台/室内绿植区、实验室工作台、工具操作区、超市货架区、便利店结账台、购物车存放区、办公室/会议室、储物间/仓库，以及多房间导航和动态空间跟踪。

交互专项包括具身导航、主动澄清、异常恢复和约束遵循；这些应作为数据集标签和高难候选类型，不能仅凭普通物体接触确认。

每个窗口要求输出：`summary`、`scene`、`objects`、`action_segments`、`unknown`、`coverage_complete` 和 `high_difficulty_candidates`。

动作段至少应包含：

```json
{
  "start_s": 0.0,
  "end_s": 3.5,
  "action": "抓取",
  "object": "杯子",
  "task_stage": "拿起杯子",
  "state_before": "杯子在桌面",
  "state_after": "杯子离开桌面",
  "confidence": 0.82,
  "evidence_times": [1.5]
}
```

动作词表是受控主统计词表：观察、接近、抓取、拿起、放下、移动、放置、打开、关闭、倒入、擦拭、折叠、装配、拆卸、按压、旋拧、交互、等待、恢复、未知。VLM 原始对象和动作可保留，但开放词不能直接改变主统计。

## 5. 难度规则

- `T` 是视频真实时长，只做时长统计。
- `N` 优先使用连续 `task_stage` 计算任务阶段数；没有阶段标签时退回动作+对象合并，并同时保留 `atomic_action_count`。
- `H` 只统计人工确认的高难事件；模型候选数和规则通过数分别记录为 `model_candidate_count`、`rule_passed_count`。
- 交互需要反馈/响应/调整；条件决策需要条件/判断/决定；多线程协调需要切换/恢复/协调/返回。
- 人工确认 `H >= 1` → 高；只有模型候选或规则通过但未人工确认 → 待判定。
- `H = 0`、链路完整且 `N <= 3` → 低。
- `H = 0`、链路完整且 `N >= 4` → 中。
- 时间轴、任务边界、动作链或证据不足 → 待判定。

模型置信度、目录名称、文件名称和“没有候选”都不能单独确认或排除高难事件。高难候选默认进入人工复核。

## 6. 已知限制

1. 300 帧预算意味着超过约 10 分钟的视频会降采样，不能保证捕捉短暂动作。
2. 窗口重叠和动作去重已接入，但 recording 级动作合并仍是启发式。
3. `task_stage` 依赖模型和人工校准；缺少阶段标签时 N 使用动作+对象回退算法。
4. VLM 能够生成候选语义，但不能替代外部反馈、状态变化和任务边界证据。
5. 同一 recording 的多机位融合、YOLO/ByteTrack、OSS 短时签名 URL 和 HTML 报告仍未接入；P2 的评测、回归对比、常驻批处理和耗时/显存观测已提供脚本/API。

## 7. 推荐后续工作

1. 对超过 10 分钟的 recording 先跳过，或使用本地变化检测筛选关键帧后再送 VLM。
2. 读取 episode manifest 的任务边界和原始时间戳作为独立证据。
4. 接入 YOLO/ByteTrack，提供对象轨迹和变化帧。
5. 用人工标注样本校准 N/H 和低中高比例，再将规则用于正式验收。

## 8. P2 评测与回归

准备独立 gold set 后，可计算动作/对象匹配、时间 IoU、遗漏率、高难 precision/recall、level 混淆矩阵、bootstrap 置信区间及来源分层指标：

```bash
python3 scripts/evaluate_gold.py \
  --gold gold_set.json --predictions predictions.json \
  --out outputs/gold_eval.json
```

固定回归集的新旧结果可用 `scripts/compare_regression.py` 对比 schema、证据、失败状态和关键难度字段；它会在工程指标退化时返回非零退出码。
