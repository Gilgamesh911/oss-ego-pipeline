# OSS Ego 数据整体结构与字段对照

> 数据位置：`oss://ccm-ego/`  
> 浏览日期：2026-09-09  
> 方法：在 OSS Browser 中浏览目录和在线文本预览；未下载数据文件。

## 1. 先看结论

**这个桶汇集了多个来源的 Ego 数据，但没有统一的数据格式。** 各来源分别采用视频文件、视频与传感器配套文件、任务/episode 目录、MCAP、HDF5 或压缩包交付。

为了理解这些差异，可以将数据归纳为五层：

| 层级 | 主要内容 | 用途 |
|---|---|---|
| ① 任务与录制 | 来源、场景、任务、episode、recording | 回答“谁采的、在哪采、做什么、是哪一段” |
| ② 观测数据 | Ego/Exo 视频、IMU、手部、触觉等 | 记录人、环境和交互过程 |
| ③ 时间与空间关系 | 时间戳、帧索引、对齐结果、标定 | 将不同传感器的数据对应起来 |
| ④ 标注与派生结果 | 语义标注、手部位姿、SLAM/VIO 等 | 描述动作、空间运动和任务含义 |
| ⑤ 交付与质量 | 元数据、清单、README、质检和审核 | 解释数据并管理交付、质量和使用方式 |

**以上是用于归纳的概念结构，不代表每家供应商都提供全部五层。** 当前证据足以比较文件组织方式，尚不足以确认全部内部字段、单位、坐标系和同步规则。

## 2. 本文的证据怎么读

为避免把文件名当作实际字段，本报告使用以下标记：

| 标记 | 含义 | 能支持的判断 |
|---|---|---|
| **正文确认** | 已实际读取在线文本正文 | 可以引用正文明确说明的内容 |
| **目录/文件确认** | 已看到目录或文件名，未读取其内部结构 | 只能确认该文件或目录存在 |
| **名称提示** | 名称含 Finger、Tactile、SLAM 等词 | 提示预期内容，不能证明实际字段或数据质量 |
| **待确认** | 当前没有充分证据 | 不计入已确认字段 |

> **本轮实质性正文读取范围：补天石 `PACKS.txt`。** 其他内容主要来自目录和文件列表。此前汇总中将 `timestamp`、`frame_index`、`robot_state`、`action` 等列为“字段”，应理解为候选统一概念，不能视为已读取到的原始 CSV 列名或 JSON 键。

## 3. 整体结构示意

以下为统一理解模型，非任何一家供应商的原样目录：

```text
数据集 dataset
└── 来源 source
    └── 场景 / 任务 scene / task
        └── 一次录制 / 片段 recording / episode
            ├── 元信息 metadata
            ├── 视觉观测：Ego 视频、Exo 视频
            ├── 传感器观测：IMU、触觉等
            ├── 时间关系：时间戳、帧索引、对齐结果
            ├── 空间关系：相机 / IMU 标定
            ├── 标注与结果：语义、手部位姿、SLAM/VIO
            └── 管理信息：manifest、review、quality
```

## 4. 字段与数据组件对照

本表的“统一名称”是建议用于比较和归一化的名称。除分包清单内容外，尚未逐项核对 CSV 表头或 JSON 键。

### 4.1 标识、场景和交付清单

| 统一名称 | 含义 | 可见原始名称或载体 | 来源 | 粒度 | 证据 |
|---|---|---|---|---|---|
| `source` | 数据来源 | 桶内各供应商/项目目录 | 各来源 | 来源级 | 目录确认；并非实际文件字段 |
| `task_id` / `task_name` | 任务编号或任务描述 | `task_0001__adjust_stove_burner_heat_knob`、`task1`、中文任务目录 | 擎羽、lingchu、Astribot、Luming、xquare 等 | 任务级 | 目录确认 |
| `episode_id` | 某任务下的一段采集 | `episode_0001__episode_000001` | 擎羽 | episode 级 | 目录确认 |
| `recording_id` | 一次录制的标识 | `d8a0h3v65vas73cq8vhg`、`rec_20260907_141029`、`aoe_…` | 补天石、独立 rec 目录、mayi、xingjiguitu_ego_QC | 录制级 | 目录确认；具体语义待说明文件核对 |
| `scene` | 采集场景 | 布料店、卧室、酒店客房保洁等 | XJGT、时切科技等 | 场景级 | 目录确认 |
| `package_name` | 分包名称 | `potentia_10hours_part01.tar` 等 | 补天石 | 分包级 | **正文确认**：`PACKS.txt` |
| `task_count` | 每个分包的任务数 | `15 tasks`、`16 tasks` | 补天石 | 分包级 | **正文确认**：共 7 包，清单合计 109 个任务条目；未核对去重与实际完整性 |
| `package_size` | 清单记录的分包大小 | 5.05–5.08 GB/包 | 补天石 | 分包级 | **正文确认**：是清单值，不是本轮递归统计 |
| `task_ids` | 分包所包含的任务 ID 列表 | 逗号分隔的任务 ID | 补天石 | 分包级 | **正文确认**：`PACKS.txt` |
| `manifest` | 数据组织或文件清单 | `episode_manifest.json` | 擎羽 | episode 级 | 文件确认；JSON 键待读取 |
| `metadata` | 录制或数据集元信息 | `meta.json`、`meta/` | 补天石、擎羽、JD、独立 rec 目录 | 待正文确认 | 文件/目录确认 |

### 4.2 观测、时间和标定

| 统一名称 | 含义 | 可见原始名称或载体 | 来源 | 粒度 | 证据与限制 |
|---|---|---|---|---|---|
| `video` | 视频观测 | `video.mp4`、`videos/`、多个 MP4 | 补天石、擎羽、JD、XJGT、youtube、shushuogushi 等 | 视频/流级 | 文件确认；编码、分辨率、帧率未读取 |
| `ego_camera` | 第一视角视频 | `ego_cam1.MP4` | XJGT | 相机流级 | 文件确认 |
| `exo_camera` | 外部视角视频 | `exo_cam1.mp4`–`exo_cam4.mp4` | XJGT；qingtong 名称提示第三视角 | 相机流级 | XJGT 抽样确认；同步关系未核验 |
| `imu` | 惯性传感器数据 | `imu.csv`、`imu.bin` | 补天石、独立 rec 目录 | 样本级待确认 | 文件确认；加速度/角速度具体列、单位、频率未读取 |
| `imu_index` | 与 IMU 二进制配套的索引 | `imu.idx` | 独立 rec 目录 | 索引级待确认 | 文件确认；索引编码未知 |
| `timestamps` | 时间信息 | `timestamps/` | 擎羽 | 待确认 | 目录确认；时间基准、单位、起点未知 |
| `frames` | 帧相关信息 | `frames.csv` | 补天石 | 帧级待确认 | 文件确认；不能据此断言列名为 `frame_index` |
| `alignment` | 对齐相关信息 | `aligned.csv` | 补天石 | 待确认 | 文件名提示；未验证对齐方式与误差 |
| `duration` | 片段时长信息 | `durations.json` | Deepreach | clip 级待确认 | 文件确认；JSON 键和值单位未读取 |
| `calibration` | 标定信息 | `calibration.json`、`calibration/` | 补天石、擎羽 | 设备/批次级待确认 | 文件/目录确认；内外参字段未读取 |
| `camera_imu_calibration` | 相机与 IMU 标定关系 | `stereo_calib-camchain-imucam.yaml`、`stereo_calib-imu.yaml` | 独立 rec 目录 | 设备级待确认 | 文件名提示；坐标系、矩阵方向和单位未知 |

### 4.3 标注、派生结果和质量

| 统一名称 | 含义 | 可见原始名称或载体 | 来源 | 证据与限制 |
|---|---|---|---|---|
| `hand_pose` | 手部位姿相关数据 | `hand_pose/`、`双目ego+手部位姿/` | 擎羽、Lingsheng | 目录确认；关节数、坐标系、表示方式未知 |
| `finger_data` | 手指相关数据 | `Ego+Finger`、`Ego_and_Finger` | GenRobot | 名称提示；不能确定是关键点、关节角或其他表示 |
| `tactile` | 触觉相关数据 | 带触觉样例、`Tactile` | 星忆智能、GenRobot | 名称提示；传感器数量、值域、频率未知 |
| `semantic_annotation` | 任务或动作语义标注 | `semantic/`、任务语义标注、`Annotation` | 擎羽、时切科技、GenRobot | 目录/名称确认；标注字段与粒度待读取 |
| `slam_vio` | 空间定位或运动估计相关结果 | `dr-egoview-teaser-slam`、`GenRobot-Ego-vio-DataSample-V2` | Deepreach、GenRobot | 名称提示；未确认可导出的轨迹、地图或位姿字段 |
| `review` | 审核相关信息 | `review/` | 擎羽 | 目录确认；审核结论字段未知 |
| `quality` | 质量归档或检查信息 | `quality_archive/`、`qc_canary/` | 补天石、擎羽、JD | 目录确认；不能仅据名称认定数据已通过验收 |
| `robot_state` | 机器人状态，例如关节/末端状态 | 本轮未取得原始字段证据 | **待确认** | 不能从 LeRobot 目录名称直接推定存在 |
| `action` | 控制动作或动作标签 | 本轮未取得原始字段证据 | **待确认** | 需区分机器人控制量与人类动作语义标签 |

## 5. 供应商与来源对照

下表只描述本次实际浏览范围，不代表某供应商全部产品或全部数据。

| 来源 | 已见组织方式 | 可见组件 | 当前判断 / 未确认部分 |
|---|---|---|---|
| 补天石 | 分包 → 记录 ID → 文件 | MP4、3 类 CSV、标定 JSON、meta JSON、PACKS 清单 | 多文件采集包；本轮读到分包清单正文，CSV/JSON 内部字段未读取 |
| 擎羽 | 样例 → task → episode | hand_pose、semantic、timestamps、videos、manifest、calibration、meta、review | 任务层级明确；虽目录名含 LeRobot，具体标准版本和数据 schema 未核验 |
| 时切科技 | Data Pack Sample → 场景/标注类别 | 脱敏、任务语义标注、带手腕相机等目录 | 按场景和标注能力组织；内部字段待读取 |
| 数据堂 | 5 个 ZIP | 最大文件 15.09GB | 压缩包内部未展开，无法确认字段 |
| 星忆智能 | 普通样例与带触觉样例 | 带触觉批次内有 5 个 ZIP | 触觉仅名称提示，内部字段未展开 |
| Astribot | sample_data → 10 类办公/家务任务 | 扫台面、换纸巾、摆餐具、叠衣服等；URDF ZIP | 任务目录和机器人模型文件已见；未读取状态/动作字段 |
| GenRobot | sample_data → 多类样例 | Ego、Finger、Tactile、VIO、Annotation 等目录；MCAP | 模态命名丰富；内容、精度和字段待确认 |
| JD | 已展开目录 + 224.55GB ZIP | data、meta、videos、quality_archive、qc_canary | 数据、元信息、视频和质量目录并存；内部 schema 未读取 |
| Deepreach | 三相机交付与 EgoView SLAM 样例 | clips、durations.json、README | 可见切片和时长文件；SLAM 结果字段未确认 |
| lightwheel | EgoDemo / EgoPro / EgoStandard | 三个分组目录 | 不能据产品分组名推定机器人状态或动作字段 |
| Lingsheng | 两个分组目录 | 双目ego+手部位姿、kaiwang100 | 双目和手部为名称提示，实际字段待读取 |
| Luming | sample_data → 4 个 task 目录 | 日期/编号命名任务 | 按任务批次组织，内部格式待确认 |
| mayi | open_data → aoe_日期_时间_p编号 | 大量记录目录 | 页面已拉取 999 项且仍有下一页，不能当作总量 |
| qingtong | sample_data → 工业场景样例 | 名称含手部、Ego、第三视角 | 多模态能力为名称提示，内部字段待读取 |
| lingchu | sample_data → task1–task6 | README、图片、任务目录 | 任务型样例，README 正文尚未读取 |
| xquare | xquare_samples → 6 类家务任务 | 厨具清洗、准备、收纳、叠放、晾晒等 | 场景任务组织已确认，内部字段待读取 |
| XJGT | 6 个场景目录 | 布料店样例有 1 路 Ego + 4 路 Exo MP4 | 文件数量已确认；不等于已验证多机位同步 |
| 独立 rec 目录 | 一次录制 → 6 个文件 | MCAP、IMU bin/idx、meta JSON、标定 YAML | 原始录制包形态明确；MCAP 通道和标定参数未读取 |
| xingjiguitu_ego_QC | 多个 rec 目录 | 页面显示 43 项；抽样一项为空 | 仅目录状态，不应推断整体质检结论 |
| sample-to-upload | 3 个任务样例目录 | README、make_annotation.py | 存在说明和标注脚本；脚本逻辑未读取 |
| Scene_example | 按任务命名的文件 | HDF5、`._` 辅助文件 | HDF5 内部 dataset/group 未展开 |
| shushuogushi / youtube | 批次或直接视频列表 | MP4 | 当前只确认视频文件层，不排除其他位置存在标注 |
| data_pipeline | 日期、实验与处理目录 | lerobot_data、models、opensource_data、post_process 等 | 混有处理过程数据，不能全部按供应商原始数据计数 |

> 更正此前汇总：Ego、Finger、Tactile、VIO 等多类型样例分组属于 **GenRobot**；**Astribot** 本轮看到的是 10 类办公/家务任务和 URDF ZIP。

## 6. 典型实际目录

### 补天石：视频、传感器与元信息配套

```text
补天石ego数据10小时/
├── PACKS.txt                  # 本轮已读正文
├── quality_archive/
└── potentia_10hours_part01/
    └── d8a0h3v65vas73cq8vhg/
        ├── video.mp4
        ├── imu.csv
        ├── aligned.csv
        ├── frames.csv
        ├── calibration.json
        └── meta.json
```

### 擎羽：任务与 episode 层级

```text
擎羽/headset_1080p_lerobot_sample8_v3/
├── calibration/
├── meta/
├── quality_archive/
├── README.md
└── tasks/
    └── task_0001__adjust_stove_burner_heat_knob/
        └── episode_0001__episode_000001/
            ├── hand_pose/
            ├── review/
            ├── semantic/
            ├── timestamps/
            ├── videos/
            └── episode_manifest.json
```

### 独立录制：MCAP、IMU 和标定

```text
rec_20260907_141029/
├── rec.mcap
├── imu.bin
├── imu.idx
├── meta.json
├── stereo_calib-camchain-imucam.yaml
└── stereo_calib-imu.yaml
```

## 7. 关键证据路径与未完成核验

以下路径均相对 `oss://ccm-ego/`。

| 证据路径 | 本轮状态 | 进一步可确认的信息 |
|---|---|---|
| `补天石ego数据10小时/PACKS.txt` | **正文已读取** | 已确认包名、任务数、清单大小、任务 ID 列表 |
| `补天石ego数据10小时/potentia_10hours_part01/d8a0h3v65vas73cq8vhg/{imu.csv,frames.csv,aligned.csv}` | 仅文件列表 | CSV 原始列名、数值单位、帧/时间关联方式 |
| 同目录 `{meta.json,calibration.json}` | 仅文件列表 | JSON 键、设备信息、标定参数及坐标系 |
| `擎羽/headset_1080p_lerobot_sample8_v3/README.md` | 仅文件列表 | 数据格式、各子目录含义、单位与使用方法 |
| `擎羽/headset_1080p_lerobot_sample8_v3/tasks/task_0001__adjust_stove_burner_heat_knob/episode_0001__episode_000001/episode_manifest.json` | 仅文件列表 | episode 的文件关联、模态清单、元数据键 |
| `Deepreach/dr-egoview-teaser-slam/{README.md,durations.json}` | 仅文件列表 | clip 时长结构、SLAM 数据交付说明 |
| `rec_20260907_141029/{meta.json,stereo_calib-camchain-imucam.yaml,stereo_calib-imu.yaml}` | 仅文件列表 | 录制元信息、标定键、坐标变换和单位 |
| `lingchu/sample_data/README.md` | 仅文件列表 | task1–task6 的定义与内部文件格式 |
| `sample-to-upload/{README.md,make_annotation.py}` | 仅文件列表 | 标注 schema 和标注生成方式 |

**当前成果是一份有证据分级的整体结构索引，还不是已完成的原始字段字典。** 要形成可直接用于数据接入的字段规范，仍需在线逐个读取上述 CSV 表头、JSON/YAML 键和 README 正文，并补充单位、坐标系、时间基准、缺失值规则及跨文件关联键。
