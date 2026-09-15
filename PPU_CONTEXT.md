# PPU 项目上下文与执行边界

## 目的

在 PPU 服务器上继续开发 OSS Ego 数据 pipeline。目标是：输入一个 OSS prefix，在线读取元数据并流式分析视频，输出可复核的动作链、任务难度和多样性报告。

## 当前仓库

- GitHub: `https://github.com/Gilgamesh911/oss-ego-pipeline`
- 默认分支：`main`
- 当前 MVP：`ego_pipeline_mvp.py`
- 示例 manifest：`mvp_manifest_potentia.json`、`mvp_manifest_rec.json`
- 现有报告：`outputs/mvp_potentia_report.json`、`outputs/mvp_rec_report.json`
- 结构报告：`outputs/ego数据整体结构与字段对照.md`

## 重要纠偏

1. `recording` 是难度判定主单位。长 recording 不自动切分，因为连续上下文、恢复过程和多线程切换可能正是高难度证据。
2. `episode` 只有在原始 manifest 明确提供边界时才作为独立任务单元。
3. 长 recording 内的疑似多次执行可以输出候选子段，但不能自动拆成多个难度样本。
4. 同一次执行的多机位只计一次任务时长，并额外记录 camera_count；同一 task 的独立多次录制分别计数。
5. N 的重复动作规则：同阶段连续重复合并；跨阶段或目标变化的重复分别计步。
6. 高难事件证据窗口不设固定时间上限，但必须有同一目标/状态的可解释因果关联；不能因为处于同一 recording 就自动串成一个事件。
7. 多线程首轮采用宽松候选口径：两个对象或两个动作同时存在时产生 candidate；只有出现目标切换、恢复或协调证据，才进入 confirmed H 和正式高难配比。
8. 交互保守判定：必须看到外部反馈，并且人做出响应或调整；普通接触不算高难交互。
9. 无法确认的片段保留为 `unknown`，记录时间段和原因；不计入 N/H，但计入待判定时长。
10. 不要把目录名、模型置信度或文件名直接当作事实字段。必须区分正文确认、目录/文件确认、名称提示和待确认。

## 已确定的 MVP 约束

- 输入：单个 OSS recording/prefix；后续再扩展到供应商目录和全桶。
- 视频访问：OSS 短时签名 URL，服务端流式解码；原视频不落盘。
- 云端 VLM：优先阿里云百炼/通义视觉。
- 云端数据：发送原始分辨率抽帧；不发送原视频、IMU 原始文件或 OSS 凭证。
- 抽帧：默认 2 秒一帧；检测到动作/对象变化后加密到 0.5 秒。
- 单个 recording 最多发送 300 帧；超过部分先用本地检测/变化检测筛选。
- 语义输出：细粒度动作段 + 整段一句话摘要。
- 动作词表：观察、接近、抓取、拿起、放下、移动、放置、打开、关闭、倒入、擦拭、折叠、装配、拆卸、按压、旋拧、交互、等待、恢复、未知；VLM 可生成开放词表扩展，但不直接改变主统计。
- 对象词表：容器、餐具、食材、衣物、工具、家具、电子设备、人体/手、清洁用品、其他；同时保留 VLM 原始对象名。
- 场景词表：居家、厨房、卧室、客厅、办公、酒店、商业、工业、户外、其他；同时保留原始场景名。
- 复核：整体置信度 `<0.75` 进入复核；疑似高难事件无论置信度都进入复核。用户选择了“只复核低置信度”，但高难事件是安全例外。

## pipeline 应有的阶段

```text
OSS 清单
→ grouping（recording 主单位）
→ 在线读取 CSV/JSON/YAML/MD/TXT
→ 容器头/索引探测（MCAP/HDF5/ZIP/TAR）
→ 统一时间轴
→ 短时 URL 流式解码
→ 自适应抽帧
→ 本地 YOLO + ByteTrack：对象、轨迹、变化帧
→ 通义视觉：动作段、对象、结果、前置条件、反馈、摘要
→ 规则引擎恢复 action chain
→ T/N/H 与 low/mid/high/pending
→ 覆盖、分布、多样性统计
→ Markdown/HTML 报告 + 复核队列
```

## 模型分工

- YOLO/RT-DETR + ByteTrack：对象检测、跟踪、变化检测、关键帧筛选。
- VLM：长时序语义、动作段、对象关系、结果、反馈、条件和一句话摘要。
- 规则引擎：时间排序、动作合并、N 计数、高难事件门槛、证据完整性。
- LLM 只能结构化候选和解释，不能生成没有时间区间的动作事实。

## 难度规则

- 高：H confirmed ≥ 1。H 类型为交互、条件决策或多线程协调，并有证据时间区间。
- 低：确认 H=0，N=1–3，任务链完整。
- 中：确认 H=0，N≥4，且有明确先后依赖或综合任务链。
- 待判定：时间轴、任务边界、N 或 H 证据缺失/冲突，或只有 candidate 证据。
- T 只用于时长统计和复核，不参与难度加权。
- 目标配比仍是低 30%、中 50%、高 20%，试行范围 25–35%、45–55%、15–25%；待判定建议 ≤5%。必须用人工样本校准后才能作为正式门槛。

## 输出 schema 方向

每个动作段至少包含：

```json
{
  "recording_id": "...",
  "t_start": 0.0,
  "t_end": 3.5,
  "canonical_action": "抓取",
  "raw_action": "pick up the cup",
  "open_vocab": false,
  "object_category": "餐具",
  "raw_object": "cup",
  "precondition": null,
  "result": "杯子离开桌面",
  "feedback": null,
  "confidence": 0.82,
  "evidence": [{"frame_time": 1.5, "source": "video.mp4"}]
}
```

难度结果至少包含：`recording_id`、`T`、`N`、`H`、`H_type`、`level`、`confidence`、`review_status`、`evidence_intervals`、`unknown_duration`。

## 首轮样本

真正验证抽帧和动作链应使用：

`oss://ccm-ego/补天石ego数据10小时/potentia_10hours_part01/d8a0h3v65vas73cq8vhg/`

已知对象：`video.mp4`、`imu.csv`、`aligned.csv`、`frames.csv`、`calibration.json`、`meta.json`。

`oss://ccm-ego/rec_20260907_141029/` 只有 MCAP、IMU、索引、meta 和标定 YAML，适合元数据链路，不适合首轮 YOLO/VLM 语义验证。

## 不要做的事情

- 不要全量下载视频、ZIP、MCAP 或 HDF5。
- 不要把多机位时长相加。
- 不要自动切分长 recording 并把片段当成独立难度样本。
- 不要用时长、运动量、模型置信度替代高难事件语义证据。
- 不要把 unknown 填成低难度或 N/H=0。
- 不要在没有实际字段证据时声称存在 `robot_state`、`action`、SLAM 轨迹等字段。
- 不要将 API Key、OSS 凭证、SSH 私钥提交到 Git。

## PPU 启动指令

```bash
git clone https://github.com/Gilgamesh911/oss-ego-pipeline.git
cd oss-ego-pipeline
python3 ego_pipeline_mvp.py --help
```

先完成 V0：MP4 + JSON/CSV/MD；把 OSS 列举、短时 URL、流式抽帧、YOLO 关键帧筛选和通义视觉适配器分别实现，任何一步失败都要写入报告的 `review_queue`，不要静默跳过。
