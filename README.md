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
