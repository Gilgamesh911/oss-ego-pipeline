# demo_sample_list

固定随机种子：20260917。10个样本来自已盘点目录的来源分层随机抽样，不是全桶等概率抽样；单文件必须0 < 时长 ≤ 600秒。过滤派生轨迹、review视频和补充机位；同一目录不重复取样。已存档候选范围和全部时长探测记录。

本轮冻结 pipeline 提交与SHA，统一2秒/帧、300帧预算、16帧/窗、最长600秒。所有首轮及复跑均重新读取视频、重新推理；仅将本次抽帧存为复核证据，不读取历史推理结果。标注只用于事后比较，不送入提示词。

难度列是**原始规则初判**，不是人工真值。工程成功率、词表合规率、时间引用合规率、重复运行一致率与视觉语义支持率分别统计，不混作准确率。没有人工真值的样本不计算正式level准确率。

| ID | 来源 | 时长/秒 | 状态 | 原始level / N / H | 最重要结论 | 文件 |
|---|---|---:|---|---|---|---|
| sample_01 | scene_example_测试样例 | 42.77 | complete | 中 / 11 / 0 | 正确识别餐桌擦拭和餐具移位；后窗把擦桌附近动作称为擦碗；N=11仍可能高估重复清洁阶段。 | [报告](outputs/demo_20260917/reports/sample_01.md) · [JSON](outputs/demo_20260917/reports/sample_01.json) |
| sample_02 | scene_example_测试样例 | 15.23 | failed | None / None / None | 窗口识别可乐瓶清洁基本正确；objects输出字典导致聚合TypeError；末段Release缺乏支持。 | [报告](outputs/demo_20260917/reports/sample_02.md) · [JSON](outputs/demo_20260917/reports/sample_02.json) |
| sample_03 | shushuogushi | 361.37 | complete | 中 / 55 / 0 | 识别头盔相机演示与骑行；出现词表外Speak/Gesture及未抽到的证据时间；N=55混入大量讲解手势。 | [报告](outputs/demo_20260917/reports/sample_03.md) · [JSON](outputs/demo_20260917/reports/sample_03.json) |
| sample_04 | 擎羽 | 4.00 | complete | 低 / 1 / 0 | 仅2帧，输出泛化的手部Move，未识别标签中的倒袋任务；加密复核仍见操作在画面边缘，不能据自动标签断言模型漏检。 | [报告](outputs/demo_20260917/reports/sample_04.md) · [JSON](outputs/demo_20260917/reports/sample_04.json) |
| sample_05 | 补天石ego数据10小时 | 428.21 | complete | 中 / 63 / 0 | 清洁水槽主题正确；90段动作合并后N=63，证据时间44/90合规；末段把拿起展开毛巾误写成擦水槽。 | [报告](outputs/demo_20260917/reports/sample_05.md) · [JSON](outputs/demo_20260917/reports/sample_05.json) |
| sample_06 | 补天石ego数据10小时 | 124.53 | complete | 中 / 23 / 0 | 识别整理→包裹→装箱→封箱；第三窗却描述为拆开包装，和整体包裹过程冲突；物品被称毛绒玩具尚不能确认。 | [报告](outputs/demo_20260917/reports/sample_06.md) · [JSON](outputs/demo_20260917/reports/sample_06.json) |
| sample_07 | astribot | 54.67 | complete | 中 / 13 / 0 | 机器人逐件装箱识别正确；彩色物体被称橡皮擦且动作边界偏早；H=0，未把普通双臂场景直接判高。 | [报告](outputs/demo_20260917/reports/sample_07.md) · [JSON](outputs/demo_20260917/reports/sample_07.json) |
| sample_08 | youtube | 145.85 | complete | 中 / 18 / 0 | 抽到新闻视频，仍输出中难/N=18；缺少非操作视频过滤；采访被写成等待，证据引用也有虚构时间。 | [报告](outputs/demo_20260917/reports/sample_08.md) · [JSON](outputs/demo_20260917/reports/sample_08.json) |
| sample_09 | astribot | 22.03 | complete | 中 / 4 / 0 | 插充电器主题正确；把8–10秒两臂交接误写成已插入，把后续靠近/插入及收回过程写成等待。 | [报告](outputs/demo_20260917/reports/sample_09.md) · [JSON](outputs/demo_20260917/reports/sample_09.json) |
| sample_10 | Deepreach | 82.42 | complete | 中 / 26 / 0 | 识别彩色塑料部件装袋，和源任务包装目标一致；未细分扇叶/手柄，部分Place时点证据不足；N=26。 | [报告](outputs/demo_20260917/reports/sample_10.md) · [JSON](outputs/demo_20260917/reports/sample_10.json) |

## 输入路径

- **sample_01**：`oss://ccm-ego/scene_example_测试样例/双目餐饮1.03h/cam0.mp4`
- **sample_02**：`oss://ccm-ego/scene_example_测试样例/双目商超2   2.48h/cam0.mp4`
- **sample_03**：`oss://ccm-ego/shushuogushi/cc_20260824_02_simple/nD9JbhaH3eM.mp4`
- **sample_04**：`oss://ccm-ego/擎羽/Hugging_face/category__small_object_storage_sorting/task_002__empty_all_sachets_from_box_onto_table/episode_0001_episode_000145_u12_aligned_attempt_clips_20260706_episode_000065/videos/left_cam_left.mp4`
- **sample_05**：`oss://ccm-ego/补天石ego数据10小时/potentia_10hours_part01/d8jv4tf65vas73e3id80/video.mp4`
- **sample_06**：`oss://ccm-ego/补天石ego数据10小时/potentia_10hours_part02/d8b4jo765vas73fkfocg/video.mp4`
- **sample_07**：`oss://ccm-ego/astribot/sample_data/家具-打包装箱/videos/chunk-000/observation.images.head_camera/episode_000003.mp4`
- **sample_08**：`oss://ccm-ego/youtube/-fI-ogq9OC8.mp4`
- **sample_09**：`oss://ccm-ego/astribot/sample_data/家具-插充电器/videos/chunk-000/observation.images.head_camera/episode_000001.mp4`
- **sample_10**：`oss://ccm-ego/Deepreach/dr-egoview-teaser-slam/clips/aec5beea-3f7e-4a2c-b623-2e2e0eb2b730_grp002_p01of01/source.mp4`

## 复跑样本

[sample_01](outputs/demo_20260917/reports/sample_01_repeat.md), [sample_05](outputs/demo_20260917/reports/sample_05_repeat.md), [sample_07](outputs/demo_20260917/reports/sample_07_repeat.md)；每个再运行一次。重复一致只说明可复现，不说明语义正确。

## 问题解决方案与实施难度

难度是研发工作量和联调风险的估计，不是模型识别难度：低=半天至1天，中=1–3天，高=3天以上或需要重新标注/评测。执行顺序按 P0 → P1 → P2；P0 解决结果不可用，P1 提升语义与 level，P2 建立正式评测和性能基线。

| 优先级 | 具体解决方案 | 难度 | 预期收益 | 验收标准 |
|---|---|---|---|---|
| **P0-1** | 模型输出后用 JSON Schema/Pydantic 校验并类型归一化：objects 只接受字符串；非法窗口标 partial，保留 raw_output；聚合前禁止对未知类型直接 set 去重。 | 中 | 消除字段格式导致的整条任务崩溃 | sample_02 不再因 unhashable dict 崩溃；失败窗口仍有可追踪报告 |
| **P0-2** | 模型只返回输入 frame_id；程序根据 frame_id 映射时间，拒绝不存在的 frame_id。动作起止区间和证据帧分开保存。 | 中 | 保证证据可回放，修复虚构时间 | 时间/帧引用合规率达到 100% 或明确标 invalid |
| **P0-3** | canonical_action 在程序端做枚举校验；未知动作转 others，保留 raw_action，不把 Speak/Gesture 静默写入正式词表。 | 低 | 保证下游标注接口稳定 | 词表外动作全部被拦截并有计数 |
| **P0-4** | 增加视频适用性分类：ego 操作、非操作内容、混合剪辑、视角不合格；非操作内容只输出摘要，level 标为不适用。 | 中 | 避免新闻等数据污染任务难度统计 | sample_08 不再输出任务 level/N |
| **P1-1** | 保留 2 秒基础抽帧；在手物接触、抓取、放置、插入、切换等变化区间自适应加密到 0.5–1 秒，并强制纳入首尾帧。 | 中 | 减少短动作和尾部状态遗漏 | 4 秒样本不再只有两帧；关键动作有前后状态帧 |
| **P1-2** | 将 16 帧窗口改为带重叠窗口（建议 8 帧重叠），并传递上一窗口的对象状态、任务阶段和未完成动作。 | 高 | 保持跨窗口动作链连续 | 跨窗离开/返回、交接/插入等阶段可串成一条链 |
| **P1-3** | 统一使用 frame_id 表示证据；用时间戳只作估计区间，并允许动作边界落在抽帧点之间。 | 中 | 减少时间边界假精确 | 报告同时给出可回放证据和估计区间 |
| **P1-4** | 建立对象状态机：未持有→抓取→持有→交接→接近目标→插入/放置→释放；只在状态变化满足时生成对应动作。 | 高 | 修复交接/插入/等待混淆 | sample_09 的交接不再判 Insert，实际插入有状态证据 |
| **P1-5** | 分离 atomic_action_count 与 task_step_count；先识别阶段和重复动作，再以对象、目标和状态变化合并任务步骤。 | 中 | 避免讲解手势和重复动作抬高 N | sample_03 的 N 不再等于大量手势数量 |
| **P1-6** | 为对象建立稳定 object_id，跨窗口合并同一对象；类别不确定时输出 unknown_object，不猜具体物品。 | 高 | 降低水槽/容器、玩具/包装物等对象混淆 | 对象类别可追溯，未知类别不产生高置信度猜测 |
| **P1-7** | 增加 occluded、out_of_view、insufficient_evidence 状态；遮挡或出画时禁止强行生成动作。 | 中 | 把不可见区间与模型错误分开 | 证据不足动作进入 unknown 队列，不计为正确动作 |
| **P1-8** | 将 model_candidate、rule_passed、human_confirmed 三层状态分开；H=0 只表示未确认，不表示没有高难事件。 | 中 | 避免 level 规则把候选当真值 | 所有 level 结果带确认状态和证据链 |
| **P1-9** | 在 recording 级建立阶段图，跟踪目标切换、条件决策、返回恢复和双任务协调；跨窗口汇总后再计算 H/level。 | 高 | 覆盖超市样本的擦拭→收银→返回工作链 | 关键阶段顺序、切换和恢复都有证据区间 |
| **P1-10** | 由两名标注者独立标注任务阶段、高难事件、level 和证据区间，分歧仲裁后形成 gold set。 | 高 | 首次获得可计算的 level 真值 | 输出 level confusion matrix、precision、recall |
| **P2-1** | 基于 gold set 计算任务命中率、动作/对象准确率、遗漏率、时间 IoU、高难 precision/recall 和 level 混淆矩阵。 | 高 | 把主观抽查升级为正式准确率 | 评估报告包含置信区间和按数据源分层结果 |
| **P2-2** | 固定本批 10 条样本和失败样本作为回归集；每次修改自动比较 schema、证据、动作和 level 指标。 | 中 | 防止修复一处又破坏另一处 | CI/批处理报告显示新旧版本差异 |
| **P2-3** | 改为常驻推理进程，一次加载模型，连续处理多个视频；窗口推理可批量化。 | 中 | 减少短视频约 8 秒的重复启动开销 | 单分钟耗时和启动耗时分别可测，并降低短片比例 |
| **P2-4** | 日志拆分 OSS 读取、抽帧、模型加载、单窗口推理、聚合和报告生成耗时，并记录显存/失败原因。 | 低 | 可以定位性能瓶颈和偶发失败 | 每条报告有阶段耗时，能计算秒/视频分钟 |

[评估总报告](outputs/demo_20260917/evaluation_summary.md) · [机器可读清单](demo_sample_list.json) · [汇总指标](outputs/demo_20260917/summary.json) · [问题与修正优先级](outputs/demo_20260917/findings.md)
