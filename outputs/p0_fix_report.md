# P0 修复报告

## 范围

本次修复覆盖四项 P0：输出 schema 防崩溃、证据帧强绑定、原子词表强制校验、非操作视频过滤。修复前基线为 commit `162cb43`；本报告用最新代码重新运行原失败的商超样本和原先误判 level 的新闻样本。

## 修复内容

1. **Schema 防崩溃**：新增 `normalize_semantic()`。`objects`、动作字段、unknown 和高难候选在进入 recording 聚合前统一校验；嵌套字典可提取安全名称，无法转换的字段进入 `validation_issues`，窗口标记为 partial，聚合阶段只处理字符串。
2. **证据帧绑定**：提示词要求模型返回 `evidence_frame_ids`。程序只接受当前输入帧 ID，并根据真实帧表生成 `evidence_times`；旧格式时间只有在 50ms 内能匹配实际输入帧时才接受，否则记录 `no_matching_input_frame` 或 `unknown_frame_id`。
3. **词表校验**：不在 `ATOMIC_ACTIONS` 中的 canonical action 转为 `others`，同时保留 `raw_action` 和违规记录。
4. **适用性过滤**：新增 `task_applicability`，规范为 `ego_task`、`non_task`、`mixed`、`unknown`。`non_task` 样本输出 `level=不适用`，不再计算 N/H。

## 修复前后复现

| 用例 | 修复前 | 修复后 | 结论 |
|---|---|---|---|
| 商超 sample_02 | `TypeError: unhashable type: dict`，整条管线失败 | 进程正常退出，8 帧、4 个动作，objects 全部为字符串 | P0-1 已阻断原崩溃 |
| 新闻 sample_08 | `level=中`、N=18，词表外 Speak 进入结果 | `task_applicability=non_task`、`level=不适用`；对象字典和非法证据进入 partial 队列；历史词表外动作会转为 others | P0-3/P0-4 生效 |
| frame evidence | 模型可引用未抽到的 65/67 秒 | 只接受真实 frame_id；非法引用被拒绝并记录 | P0-2 fail-closed 生效 |

## 实测结果

- 快速回归：`scripts/test_p0_regressions.py` **PASS**。覆盖历史 sample_02 字典对象、词表外动作、未知 frame ID、无法映射的旧时间格式和 non_task level。
- sample_02 最新实跑：`candidate_semantics`，无聚合异常；4 个动作，所有 objects 为字符串，窗口无 schema 问题。
- sample_08 最新实跑：进程正常退出但结果状态为 `partial_semantic_output`，因为本次模型仍生成了 18 个未知证据 frame ID 和 4 个无法转换的对象字典；这些问题已被阻断并进入 `validation_issues`，不是静默通过。本次运行没有词表外 canonical action；历史 sample_03 的 Speak/Gesture 由同一规范化逻辑转为 `others`。视频被识别为 `non_task`，难度为 `不适用`，N/H 不计算。

这里的 `partial` 是保护性结果：它说明模型输出仍有违规字段，但不会再把违规字段当成可信标注。P0 修复解决了崩溃和静默错误传播，不能替代后续 P1 的对象状态、跨窗口动作链和正式准确率评估。

## 文件

- [商超修复后原始结果](p0_retest_sample02.json)
- [新闻修复后原始结果](p0_retest_sample08.json)
- [机器可读修复报告](p0_fix_report.json)
- [快速回归脚本](../scripts/test_p0_regressions.py)
