# 本次 OSS 批量测试说明

入口：[样本清单](../../demo_sample_list.md)、[结果汇总](evaluation_summary.md)、[机器可读指标](summary.json)。每条样本有 JSON 原始输出、Markdown 报告、工程检查结果、抽帧总览和动作复核证据。

## 抽样与运行

- 固定随机种子 `20260917`。先对有限来源目录做有时间预算的盘点，再按来源分层随机取样，共 10 条。不按识别结果挑选，不替换失败样本。
- `inventory.json` 保存候选路径及盘点是否完整；`selection_probes.json` 保存抽中候选的时长探测和排除原因。不是全 OSS 桶等概率抽样。
- 用视频元数据确认每条 `0 < duration <= 600` 秒。排除可识别的派生轨迹、review 视频和补充机位；目录去重不是完整的 recording ID 去重。
- 冻结 `demo_sample_list.json` 内的 pipeline commit、文件 SHA256、模型和参数。2 秒一帧、最多 300 帧、16 帧一窗，`do_sample=False`。整段覆盖依赖实际时间戳，不把帧数当语义覆盖率。
- 全部首轮与 3 条预先选定的复跑均使用各自全新的帧目录，重新读 OSS 和执行模型。每次独立载入模型，不读取历史推理缓存。
- 源视频保留在 OSS；本机仅保存抽帧、报告、日志。`work/demo_20260917` 是本轮证据帧和窗口检查点。
- pipeline 本体保持不变，便于把失败计入原版本基线；辅助脚本负责编排、核查和报告。

## 文件与脚本

| 内容 | 位置 |
|---|---|
| 固定输入列表、状态及重要结论 | `../../demo_sample_list.json` / `.md` |
| 每条原始输出和报告 | `reports/sample_XX.json` / `.md` |
| 未聚合的窗口输出（包含失败样本） | `reports/sample_XX.windows.raw.json` |
| 字段、词表及时间引用检查 | `reports/sample_XX.window_checks.json` |
| 复跑原始结果与报告 | `reports/sample_XX_repeat.json` / `.md` |
| 总览及动作邻帧证据 | `previews/` |
| 独立视觉复核记录 | `visual_reviews.json` |
| 事后读取的源任务文字 | `reference_metadata.json` |
| Python/库版本、模型配置指纹 | `environment.json` |
| 原始运行日志 | `logs/`（本地保留，Git 默认忽略日志） |
| 时长探测 / 抽样 | `../../scripts/demo_probe.py` / `select_demo_samples.py` |
| 循环执行 / 导出复核图 / 汇总 | `../../scripts/run_demo_batch.py` / `demo_audit_panels.py` / `summarize_demo_batch.py` |

这是一次固定批次：不要重新执行选样脚本覆盖已完成的清单。批处理脚本重启会跳过已完成/失败记录；若未完成记录已有缓存则拒绝覆盖，避免把恢复执行误称为全新测试。

报告再生成（不执行模型）：

```bash
cd /root/oss-ego-pipeline
python3 scripts/demo_audit_panels.py
python3 scripts/summarize_demo_batch.py
```

## 指标解释

1. **完整管线成功率**：进程正常退出并得到完整候选语义结果的样本占比。窗口能解析但聚合崩溃仍算失败；部分窗口失败另行显示。
2. **JSON/词表/字段合规率**：首轮所有可读取窗口的工程指标，包含聚合失败样本的窗口。字段存在不代表类型、含义正确。
3. **时间与帧引用合规率**：动作区间位于视频和所属窗口内，且证据时间能对应实际抽到的帧。允许 0.05 秒浮点误差；不是逐帧边界准确率。
4. **复跑一致率**：3 条样本独立重跑，比较时间戳、语义 JSON、level、N、H 和动作集合。原始时间开销不参与语义比较。一致不代表正确。
5. **任务文字一致性**：对存在源任务文字的样本，比较输出主题是否一致。源标签可能自动生成或与视角不匹配，不当作正式真值；标签没有输入模型。
6. **抽查视觉支持率**：助手查看总览，并系统选取每条第一个、中间、最后一个生成动作，查看该动作及相邻帧。失败样本从窗口检查点抽查。四类判断为 supported、partial、unsupported、insufficient；严格支持率分子只含 supported，分母含全部四类。不能外推全视频所有动作或全桶准确率。
7. **level 准确率**：本批无独立专家真值，保留为空。模型 `confidence`、规则 `confirmed`、`H=0` 均不自动构成真值。

短视频可能因帧数少或遮挡而不可判；需要区分模型错误、抽样证据不足和源标注不匹配。`sample_04_dense_review.jpg` 为事后 0.5 秒间隔补充查看，未重新送入 pipeline，不替换首轮结果。
