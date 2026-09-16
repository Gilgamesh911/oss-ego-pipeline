# OSS Ego Pipeline MVP

用于从 `oss://ccm-ego/` 等对象存储路径建立 Ego 数据索引、流式视频分析和任务多样性报告的 MVP。

## 当前能力

- 接收单个 OSS recording/prefix 的对象清单
- 区分视频、CSV、JSON/YAML/Markdown、二进制和压缩包
- 不保存原始视频；规划使用短时签名 URL 流式解码
- 自适应抽帧：默认 2 秒，检测到变化后加密到 0.5 秒
- 本地轻量 YOLO/跟踪器负责对象检测、轨迹和关键帧筛选（适配器待接入）
- 阿里云百炼/通义视觉 VLM 负责动作段、对象、结果和一句话摘要（适配器待接入）
- 受控动作、对象、场景词表；无法归一时保留开放词表结果
- 长 recording 作为难度判定主单位，不自动切分
- 生成 T/N/H、难度、复核队列和多样性报告骨架

## 运行

```bash
python3 ego_pipeline_mvp.py \
  --prefix oss://ccm-ego/补天石ego数据10小时/potentia_10hours_part01/d8a0h3v65vas73cq8vhg/ \
  --manifest mvp_manifest_potentia.json \
  --out outputs/mvp_potentia_report.json
```

### Qwen3-VL 抽帧语义识别

环境已配置 `transformers`、`qwen-vl-utils[decord]`，可直接对 OSS 挂载路径的视频运行：

```bash
python3 qwen_vl_pipeline.py \
  --model /mnt/workspace/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master \
  --video /mnt/oss/补天石ego数据10小时/potentia_10hours_part01/d8a0h3v65vas73cq8vhg/video.mp4 \
  --out outputs/qwen3_vl_sample.json \
  --interval 2 --max-frames 300 --frames-cache work/potentia_frames
```

首次运行会从 Hugging Face 下载 `Qwen/Qwen3-VL-4B-Instruct` 权重；模型使用 BF16 和自动设备映射，原始视频不写入报告。若 Hugging Face CAS 返回 401，可先设置 `HF_HUB_DISABLE_XET=1`，或将模型预下载到本地目录后通过 `--model /path/to/model` 使用：

本机已将权重下载到 `/mnt/workspace/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master`，也可以直接使用该目录。长 recording 会在预算内覆盖全程，不再只读取开头。

当前 Qwen 脚本以每 2 秒抽帧为目标、每 16 帧一个 VLM 窗口，并在 recording 级汇总窗口结果。超过300帧预算的长视频目前采用全程均匀粗采样，尚未接入自适应变化筛选。时间seek后会解码到目标时间，不直接把前置关键帧当作目标帧。

`--frames-cache` 仅保存选定帧和逐窗推理检查点，不保存原视频；源文件及采样参数匹配时可复用完整抽帧缓存。PyAV使用FFmpeg原生日志回调，避免多线程解码器析构时Python日志回调死锁。每20帧和每个推理窗口均打印进度。

level规则输出只是初判：现有动作合并不能可靠区分同阶段重复和任务依赖，也不能仅凭模型文字确认高难因果证据。历史商超报告的 `confirmed` 不能作为已核验结论，需结合证据帧和完整性检查复核。JSON解析失败会记录在 `review_queue`。

```bash
HF_HUB_DISABLE_XET=1 python3 qwen_vl_pipeline.py --video /mnt/oss/补天石ego数据10小时/potentia_10hours_part01/d8a0h3v65vas73cq8vhg/video.mp4 --out outputs/qwen3_vl_sample.json --max-frames 300
```

manifest 格式：

```json
{"objects":[{"key":"path/to/file.mp4","size":12345}]}
```

## 当前约束

- VLM 云端目的地优先使用阿里云百炼/通义视觉。
- 云端只发送原始分辨率抽帧，不发送原视频、IMU 原始文件或 OSS 凭证。
- 单个 recording 最多发送 300 帧，超过部分由本地检测/变化筛选。
- 低置信度 `<0.75` 进入复核；疑似高难事件也进入复核。
- `unknown` 保留并计入待判定时长，不计入 N/H。
- 多机位同一 recording 只计一次任务时长；独立重复录制分别计数。

## 文档

- `outputs/ego数据整体结构与字段对照.md`：OSS Ego 数据结构和字段证据整理
- 飞书文档中的“第三步：任务分级”：输入、输出、动作链、模型分工和报告方案

## 下一步

1. 接入 OSS 列举和短时签名 URL
2. 接入 FFmpeg/OpenCV 流式抽帧
3. 接入 YOLO + ByteTrack 关键帧筛选
4. 接入阿里云百炼视觉模型适配器
5. 增加动作段 JSON Schema、规则引擎和 Markdown/HTML 报告
6. 在 PPU 上部署依赖并用真实 recording 做 badcase 迭代
