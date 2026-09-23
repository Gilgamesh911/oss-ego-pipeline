#!/usr/bin/env python3
"""Run Qwen3-VL semantic analysis on sampled frames from an OSS-mounted video."""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

ATOMIC_ACTIONS = [
    # 基础运动与位移
    "Move", "Reach", "Retract", "Lift", "Lower", "Stop", "Drag",
    # 抓取与释放
    "Grasp", "Pick", "Release", "Drop", "Hold", "Handover",
    # 放置与定位
    "Place", "Position", "Rotate", "Tilt", "Flip", "Stack",
    # 施力与接触交互
    "Push", "Pull", "Press", "Squeeze", "Touch", "Tap", "Strike", "Hammer", "Shake", "Rub", "Crush", "Snap",
    # 组装、连接与分离
    "Insert", "Screw", "Unscrew", "Attach", "Detach", "Switch", "Open", "Close",
    # 柔性物体与液体
    "Pour", "Stir", "Fold", "Unfold", "Braid", "Tie", "Wrap", "Thread", "Peel", "Spread",
    # 工具使用与状态改变
    "Cut", "Wipe", "Clean", "Sweep", "Scrub", "Dispose", "Paint",
    # 认知、规划与管理
    "Identify", "Verify", "Group", "Organize", "Wait",
    "others",
]
ACTION_ZH = {"Move":"移动", "Reach":"伸向/接近", "Retract":"收回/撤回", "Lift":"举起/抬起",
 "Lower":"放下/降低", "Stop":"停止/阻挡", "Drag":"拖拽", "Grasp":"抓取", "Pick":"拾取",
 "Release":"释放", "Drop":"丢下", "Hold":"握住/保持", "Handover":"递送", "Place":"放置",
 "Position":"定位", "Rotate":"旋转", "Tilt":"倾斜", "Flip":"翻转", "Stack":"堆叠", "Push":"推",
 "Pull":"拉", "Press":"按压", "Squeeze":"挤压", "Touch":"触摸", "Tap":"轻击/打字", "Strike":"击打",
 "Hammer":"锤击", "Shake":"摇晃", "Rub":"摩擦", "Crush":"压碎/捣碎", "Snap":"折断/弯折",
 "Insert":"插入", "Screw":"旋紧", "Unscrew":"松开/拧开", "Attach":"连接", "Detach":"分离",
 "Switch":"切换/开关", "Open":"打开", "Close":"关闭", "Pour":"倒/注", "Stir":"搅拌",
 "Fold":"折叠", "Unfold":"展开/铺平", "Braid":"编织", "Tie":"打结/系", "Wrap":"包裹/裹",
 "Thread":"穿线", "Peel":"剥", "Spread":"涂抹", "Cut":"切割/切开", "Wipe":"擦拭",
 "Clean":"清扫", "Sweep":"扫", "Scrub":"擦洗/搓", "Dispose":"处理/丢弃", "Paint":"涂画",
 "Identify":"识别/找", "Verify":"确认", "Group":"归类/分拣", "Organize":"整理", "Wait":"等待"}
IGNORED_FOR_N = {"Wait", "等待", "观察", "未知"}
H_TYPES = {"interaction", "conditional_decision", "multi_thread_coordination",
           "交互", "条件决策", "多线程协调"}

APPLICABILITY_VALUES = {"ego_task", "non_task", "mixed", "unknown"}
APPLICABILITY_ALIASES = {
    "ego": "ego_task", "ego_task": "ego_task", "操作任务": "ego_task", "第一视角操作": "ego_task",
    "non_task": "non_task", "非操作": "non_task", "新闻": "non_task", "讲解": "non_task",
    "mixed": "mixed", "混合": "mixed", "混合视频": "mixed",
    "unknown": "unknown", "未知": "unknown", "不确定": "unknown",
}


class JsonlLogger:
    """Small flush-per-event logger safe for long-running batch jobs."""

    def __init__(self, path, run_id=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One log belongs to one output run; avoid mixing repeated reruns of
        # the same --out path while still flushing every event immediately.
        self.path.write_text("", encoding="utf-8")
        self.run_id = run_id or self.path.stem
        self.started = time.monotonic()

    def emit(self, event, **fields):
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "elapsed_since_log_start_s": round(time.monotonic() - self.started, 4),
        }
        record.update(fields)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _timing_metrics(stage_timings, duration_s):
    """Add wall-clock and video-duration ratios without hiding model time."""
    total = float(stage_timings.get("total_s", 0.0) or 0.0)
    inference = float(stage_timings.get("window_inference_s", 0.0) or 0.0)
    model_load = float(stage_timings.get("model_load_s", 0.0) or 0.0)
    script_runtime = max(0.0, total - inference)
    script_overhead = max(0.0, total - inference - model_load)
    stage_timings["end_to_end_s"] = round(total, 4)
    stage_timings["model_inference_s"] = round(inference, 4)
    stage_timings["script_runtime_excluding_inference_s"] = round(script_runtime, 4)
    stage_timings["script_overhead_excluding_model_s"] = round(script_overhead, 4)
    if duration_s and duration_s > 0:
        stage_timings["video_duration_s"] = round(float(duration_s), 4)
        stage_timings["end_to_end_to_video_ratio"] = round(total / duration_s, 4)
        stage_timings["model_inference_to_video_ratio"] = round(inference / duration_s, 4)
        stage_timings["script_runtime_to_video_ratio"] = round(script_runtime / duration_s, 4)
        stage_timings["script_overhead_to_video_ratio"] = round(script_overhead / duration_s, 4)


def _report_paths(out_path):
    output = Path(out_path)
    return output.with_suffix(".report.md"), output.with_suffix(".report.html")


def _fmt_seconds(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return str(value)


def _report_content(report):
    timings = report.get("stage_timings_s", {}) or {}
    semantic = report.get("semantic", {}) or {}
    difficulty = semantic.get("difficulty", {}) or {}
    duration = report.get("duration_s", timings.get("video_duration_s", difficulty.get("T")))
    actions = semantic.get("action_segments", []) or []
    unknown = semantic.get("unknown", []) or []
    validation = semantic.get("validation_issues", []) or []
    rows = []
    for action in actions:
        rows.append({
            "start": action.get("start_s"), "end": action.get("end_s"),
            "stage": action.get("task_stage", ""),
            "action": action.get("canonical_action", action.get("raw_action", "others")),
            "raw": action.get("raw_action", ""), "object": action.get("object", ""),
            "before": action.get("state_before", ""), "after": action.get("state_after", ""),
            "confidence": action.get("confidence"),
            "evidence": ", ".join(str(x) for x in action.get("evidence_frame_ids", [])),
        })
    return {
        "status": report.get("status", "unknown"), "video": report.get("video", ""),
        "model": report.get("model", ""), "duration": duration,
        "frame_count": report.get("frame_count", 0),
        "timestamps": report.get("timestamps_s", []), "timings": timings,
        "semantic": semantic, "difficulty": difficulty, "actions": rows,
        "unknown": unknown, "validation": validation,
        "review_queue": report.get("review_queue", []) or [],
    }


def write_report_files(report, output_json, report_md=None, report_html=None):
    """Write a readable Markdown report and a self-contained HTML timeline."""
    data = _report_content(report)
    default_md, default_html = _report_paths(output_json)
    md_path, html_path = Path(report_md or default_md), Path(report_html or default_html)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    t = data["timings"]
    d = data["difficulty"]
    s = data["semantic"]
    model_load_display = t.get("model_load_s", t.get("shared_model_load_s"))
    lines = [
        f"# Pipeline report: `{data['video']}`", "",
        f"- Status: `{data['status']}`", f"- Model: `{data['model']}`",
        f"- Video duration: `{_fmt_seconds(data['duration'])}`",
        f"- Sampled frames: `{data['frame_count']}`", "",
        "## Runtime breakdown", "",
        "| Metric | Value | Ratio to video duration |", "|---|---:|---:|",
        f"| End-to-end wall time | {_fmt_seconds(t.get('end_to_end_s'))} | {t.get('end_to_end_to_video_ratio', '—')}x |",
        f"| Script runtime excluding model inference | {_fmt_seconds(t.get('script_runtime_excluding_inference_s'))} | {t.get('script_runtime_to_video_ratio', '—')}x |",
        f"| Script overhead excluding model load and inference | {_fmt_seconds(t.get('script_overhead_excluding_model_s'))} | {t.get('script_overhead_to_video_ratio', '—')}x |",
        f"| Model load (or shared batch load) | {_fmt_seconds(model_load_display)} | — |",
        f"| Model inference | {_fmt_seconds(t.get('model_inference_s', t.get('window_inference_s')))} | {t.get('model_inference_to_video_ratio', '—')}x |",
        "", "## Data and semantics", "",
        f"- Scene: {s.get('scene') or '—'}",
        f"- Task applicability: `{s.get('task_applicability', 'unknown')}`",
        f"- Summary: {s.get('summary') or '—'}",
        f"- Objects: {', '.join(s.get('objects', [])) or '—'}",
        f"- Level: `{d.get('level', '—')}`; reason: {d.get('reason', '—')}",
        f"- T/N/H: `{d.get('T', '—')}` / `{d.get('N', '—')}` / `{d.get('H', '—')}`",
        "", "## Action evidence", "",
        "| Start–end | Stage | Action | Object | State change | Confidence | Evidence frames |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for row in data["actions"]:
        lines.append(f"| {row['start']}–{row['end']}s | {row['stage']} | {row['action']} ({row['raw']}) | {row['object']} | {row['before']} → {row['after']} | {row['confidence']} | {row['evidence']} |")
    if not data["actions"]:
        lines.append("| — | — | No validated action segments | — | — | — | — |")
    lines += ["", "## Review basis", "", f"- Unknown intervals: `{json.dumps(data['unknown'], ensure_ascii=False)}`",
              f"- Validation issues: `{len(data['validation'])}`",
              f"- Review queue: `{json.dumps(data['review_queue'], ensure_ascii=False)}`", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    duration_num = float(data["duration"] or 0.0)
    timeline = []
    for row in data["actions"]:
        start = float(row["start"] or 0.0) if row["start"] is not None else 0.0
        end = float(row["end"] or start) if row["end"] is not None else start
        left = max(0.0, min(100.0, start / duration_num * 100)) if duration_num else 0.0
        width = max(0.8, min(100.0 - left, (end - start) / duration_num * 100)) if duration_num else 1.0
        label = f"{row['action']} · {row['object']} · {start:.2f}–{end:.2f}s"
        timeline.append(f'<div class="action" style="left:{left:.3f}%;width:{width:.3f}%" title="{html_escape(label)}"><span>{html_escape(row["action"])}</span></div>')
    html = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline report</title>
<style>
:root {{ color-scheme: light dark; --bg: Canvas; --fg: CanvasText; --muted: color-mix(in srgb, CanvasText 65%, Canvas); --track: color-mix(in srgb, CanvasText 12%, Canvas); --accent: #3976d4; --accent2: #d97706; }}
body {{ font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg); margin:2rem auto; max-width:1100px; padding:0 1rem; line-height:1.45; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0 1.5rem; }} th,td {{ text-align:left; padding:.45rem .6rem; border-bottom:1px solid color-mix(in srgb, CanvasText 18%, Canvas); vertical-align:top; }}
.muted {{ color:var(--muted); }} .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:.7rem; }}
.metric {{ padding:.8rem; border:1px solid color-mix(in srgb, CanvasText 18%, Canvas); border-radius:.5rem; }} .metric strong {{ display:block; font-size:1.15rem; }}
.track {{ position:relative; height:2.2rem; background:var(--track); border-radius:.3rem; margin:1rem 0 1.5rem; }} .action {{ position:absolute; top:.25rem; height:1.7rem; background:var(--accent); color:white; border-radius:.25rem; overflow:hidden; white-space:nowrap; padding:.25rem .35rem; box-sizing:border-box; font-size:.78rem; }}
code {{ overflow-wrap:anywhere; }}
</style>
<h1>Pipeline report</h1>
<p><strong>Status:</strong> {html_escape(str(data['status']))} · <strong>Video:</strong> <code>{html_escape(str(data['video']))}</code></p>
<div class="metrics"><div class="metric"><span class="muted">Video duration</span><strong>{html_escape(_fmt_seconds(data['duration']))}</strong></div><div class="metric"><span class="muted">End-to-end</span><strong>{html_escape(_fmt_seconds(t.get('end_to_end_s')))}</strong><span class="muted">{t.get('end_to_end_to_video_ratio', '—')}× video</span></div><div class="metric"><span class="muted">Script excluding inference</span><strong>{html_escape(_fmt_seconds(t.get('script_runtime_excluding_inference_s')))}</strong><span class="muted">{t.get('script_runtime_to_video_ratio', '—')}× video</span></div><div class="metric"><span class="muted">Model inference</span><strong>{html_escape(_fmt_seconds(t.get('model_inference_s', t.get('window_inference_s'))))}</strong><span class="muted">{t.get('model_inference_to_video_ratio', '—')}× video</span></div></div>
<h2>Action timeline</h2><p class="muted">Timeline is normalized to the recorded video duration; hover an action for details.</p>
<div class="track" role="img" aria-label="Action timeline">{''.join(timeline) or '<span class="muted">No validated action segments</span>'}</div>
<h2>Data and semantics</h2><table><tr><th>Scene</th><td>{html_escape(str(s.get('scene') or '—'))}</td></tr><tr><th>Task applicability</th><td>{html_escape(str(s.get('task_applicability', 'unknown')))}</td></tr><tr><th>Summary</th><td>{html_escape(str(s.get('summary') or '—'))}</td></tr><tr><th>Objects</th><td>{html_escape(', '.join(s.get('objects', [])) or '—')}</td></tr><tr><th>Level / basis</th><td>{html_escape(str(d.get('level', '—')))} / {html_escape(str(d.get('reason', '—')))}</td></tr><tr><th>T / N / H</th><td>{html_escape(str(d.get('T', '—')))} / {html_escape(str(d.get('N', '—')))} / {html_escape(str(d.get('H', '—')))}</td></tr></table>
<h2>Action evidence</h2><table><tr><th>Interval</th><th>Stage</th><th>Action</th><th>Object</th><th>State</th><th>Evidence</th></tr>{''.join(f'<tr><td>{html_escape(str(r["start"]))}–{html_escape(str(r["end"]))}s</td><td>{html_escape(str(r["stage"]))}</td><td>{html_escape(str(r["action"]))}<br><span class="muted">{html_escape(str(r["raw"]))}</span></td><td>{html_escape(str(r["object"]))}</td><td>{html_escape(str(r["before"]))} → {html_escape(str(r["after"]))}</td><td>{html_escape(str(r["evidence"]))}</td></tr>' for r in data["actions"]) or '<tr><td colspan="6">No validated action segments</td></tr>'}</table>
<h2>Review basis</h2><p>Unknown intervals: <code>{html_escape(json.dumps(data['unknown'], ensure_ascii=False))}</code></p><p>Validation issues: <strong>{len(data['validation'])}</strong></p><p>Review queue: <code>{html_escape(json.dumps(data['review_queue'], ensure_ascii=False))}</code></p>
</html>'''
    html_path.write_text(html, encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path)}


def load_vlm(model_id):
    """Load the VLM once so batch callers can reuse it across recordings."""
    started = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, time.monotonic() - started


def resource_snapshot():
    """Return lightweight resource data without requiring a monitoring daemon."""
    result = {"cuda_available": bool(torch.cuda.is_available())}
    if result["cuda_available"]:
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
        result["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
        free, total = torch.cuda.mem_get_info()
        result["cuda_free_bytes"] = int(free)
        result["cuda_total_bytes"] = int(total)
    return result


def _string_value(value, *, field, issues, allow_empty=False):
    """Return a safe scalar string without letting nested model JSON break aggregation."""
    if isinstance(value, str):
        if value or allow_empty:
            return value
        issues.append({"field": field, "reason": "empty_string"})
        return ""
    if isinstance(value, dict):
        for key in ("name", "label", "object", "text", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                issues.append({"field": field, "reason": "dict_coerced", "key": key})
                return candidate
        issues.append({"field": field, "reason": "dict_rejected"})
        return "unknown"
    if value is None and allow_empty:
        return ""
    issues.append({"field": field, "reason": "expected_string", "type": type(value).__name__})
    return str(value) if value is not None else "unknown"


def normalize_applicability(value, issues=None):
    issues = issues if issues is not None else []
    if isinstance(value, dict):
        value = value.get("category") or value.get("type") or value.get("label")
    if not isinstance(value, str):
        if value is not None:
            issues.append({"field": "task_applicability", "reason": "expected_string"})
        return "unknown"
    normalized = APPLICABILITY_ALIASES.get(value.strip().lower(), APPLICABILITY_ALIASES.get(value.strip(), "unknown"))
    if normalized == "unknown" and value.strip().lower() not in {"unknown", "未知", "不确定"}:
        issues.append({"field": "task_applicability", "reason": "unknown_value", "value": value})
    return normalized


def _nearest_frame_id(value, frame_ids, timestamps, tolerance=0.05):
    if not isinstance(value, (int, float)) or not timestamps:
        return None
    index = min(range(len(timestamps)), key=lambda i: abs(float(timestamps[i]) - float(value)))
    return frame_ids[index] if abs(float(timestamps[index]) - float(value)) <= tolerance else None


def normalize_semantic(raw, frame_ids, timestamps, duration_s):
    """Validate one model window before it can be merged into recording output.

    Evidence is represented by real input frame IDs. Legacy evidence_times are
    accepted only when they exactly map to an input timestamp within 50 ms.
    Invalid fields are retained in ``validation_issues`` and never crash the
    recording-level aggregation.
    """
    issues = []
    if not isinstance(raw, dict):
        issues.append({"field": "semantic", "reason": "expected_object"})
        raw = {"raw_output": str(raw)}
    result = {
        "summary": _string_value(raw.get("summary", ""), field="summary", issues=issues, allow_empty=True),
        "scene": _string_value(raw.get("scene", ""), field="scene", issues=issues, allow_empty=True),
        "objects": [], "action_segments": [], "unknown": [],
        "high_difficulty_candidates": [],
        "task_applicability": normalize_applicability(raw.get("task_applicability"), issues),
    }
    coverage_complete = raw.get("coverage_complete")
    if coverage_complete is not None and not isinstance(coverage_complete, bool):
        issues.append({"field": "coverage_complete", "reason": "expected_boolean"})
        coverage_complete = None
    result["coverage_complete"] = coverage_complete
    if "raw_output" in raw:
        result["raw_output"] = raw["raw_output"]

    objects = raw.get("objects", [])
    if not isinstance(objects, list):
        issues.append({"field": "objects", "reason": "expected_array", "type": type(objects).__name__})
        objects = []
    for i, value in enumerate(objects):
        text = _string_value(value, field=f"objects[{i}]", issues=issues)
        if text and text not in result["objects"]:
            result["objects"].append(text)

    actions = raw.get("action_segments", [])
    if not isinstance(actions, list):
        issues.append({"field": "action_segments", "reason": "expected_array", "type": type(actions).__name__})
        actions = []
    for i, item in enumerate(actions):
        if not isinstance(item, dict):
            issues.append({"field": f"action_segments[{i}]", "reason": "expected_object"})
            continue
        action = dict(item)
        canonical = action.get("canonical_action", action.get("action", "others"))
        if canonical not in ATOMIC_ACTIONS:
            issues.append({"field": f"action_segments[{i}].canonical_action", "reason": "outside_vocabulary", "value": canonical})
            action["canonical_action"] = "others"
        else:
            action["canonical_action"] = canonical
        action["raw_action"] = _string_value(action.get("raw_action", ""), field=f"action_segments[{i}].raw_action", issues=issues, allow_empty=True)
        action["object"] = _string_value(action.get("object", "unknown"), field=f"action_segments[{i}].object", issues=issues)
        action["task_stage"] = _string_value(action.get("task_stage", ""), field=f"action_segments[{i}].task_stage", issues=issues, allow_empty=True)
        action["state_before"] = _string_value(action.get("state_before", ""), field=f"action_segments[{i}].state_before", issues=issues, allow_empty=True)
        action["state_after"] = _string_value(action.get("state_after", ""), field=f"action_segments[{i}].state_after", issues=issues, allow_empty=True)
        # Only accept numeric finite values. Range and ordering are checked in the report validator.
        for key in ("start_s", "end_s", "confidence"):
            if not isinstance(action.get(key), (int, float)) or isinstance(action.get(key), bool):
                issues.append({"field": f"action_segments[{i}].{key}", "reason": "expected_number"})
                action[key] = None
        evidence_ids = action.get("evidence_frame_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []
            if action.get("evidence_times") is not None:
                old_times = action.get("evidence_times") if isinstance(action.get("evidence_times"), list) else []
                for value in old_times:
                    frame_id = _nearest_frame_id(value, frame_ids, timestamps)
                    if frame_id is None:
                        issues.append({"field": f"action_segments[{i}].evidence_times", "reason": "no_matching_input_frame", "value": value})
                    else:
                        evidence_ids.append(frame_id)
        valid_ids = [x for x in evidence_ids if x in frame_ids]
        if len(valid_ids) != len(evidence_ids):
            issues.append({"field": f"action_segments[{i}].evidence_frame_ids", "reason": "unknown_frame_id"})
        action["evidence_frame_ids"] = list(dict.fromkeys(valid_ids))
        action["evidence_times"] = [timestamps[frame_ids.index(x)] for x in action["evidence_frame_ids"]]
        result["action_segments"].append(action)

    for field in ("unknown", "high_difficulty_candidates"):
        values = raw.get(field, [])
        if not isinstance(values, list):
            issues.append({"field": field, "reason": "expected_array"})
            values = []
        result[field] = [x for x in values if isinstance(x, dict)]
        if len(result[field]) != len(values):
            issues.append({"field": field, "reason": "non_object_items_dropped"})
    # High-difficulty candidates use the same grounded evidence contract.
    for i, candidate in enumerate(result["high_difficulty_candidates"]):
        evidence_ids = candidate.get("evidence_frame_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []
            old_times = candidate.get("evidence_times", [])
            if isinstance(old_times, list):
                for value in old_times:
                    frame_id = _nearest_frame_id(value, frame_ids, timestamps)
                    if frame_id is None:
                        issues.append({"field": f"high_difficulty_candidates[{i}].evidence_times",
                                       "reason": "no_matching_input_frame", "value": value})
                    else:
                        evidence_ids.append(frame_id)
        valid_ids = [x for x in evidence_ids if x in frame_ids]
        if len(valid_ids) != len(evidence_ids):
            issues.append({"field": f"high_difficulty_candidates[{i}].evidence_frame_ids",
                           "reason": "unknown_frame_id", "value": evidence_ids})
        candidate["evidence_frame_ids"] = list(dict.fromkeys(valid_ids))
        candidate["evidence_times"] = [timestamps[frame_ids.index(x)] for x in candidate["evidence_frame_ids"]]
        for key in ("start_s", "end_s"):
            if not isinstance(candidate.get(key), (int, float)) or isinstance(candidate.get(key), bool):
                issues.append({"field": f"high_difficulty_candidates[{i}].{key}", "reason": "expected_number"})
                candidate[key] = None
        if candidate.get("start_s") is not None and candidate.get("end_s") is not None:
            if not (0 <= candidate["start_s"] < candidate["end_s"] <= duration_s):
                issues.append({"field": f"high_difficulty_candidates[{i}]", "reason": "invalid_interval"})
    if issues:
        result["validation_issues"] = issues
    return result

def recording_difficulty(semantic: dict, duration_s: float) -> dict:
    """Conservative recording-level N/H/level calculation."""
    applicability = normalize_applicability(semantic.get("task_applicability"))
    if applicability == "non_task":
        return {"T": round(duration_s, 3), "N": None, "H": None, "H_type": [],
                "level": "不适用", "confidence": "规则判定", "review_status": "filtered",
                "evidence_intervals": [], "unknown_duration": 0,
                "merged_action_count": 0, "reason": "非操作任务视频，不参与ego任务难度评估"}
    segments = sorted((s for s in semantic.get("action_segments", [])
                       if isinstance(s, dict) and (s.get("canonical_action", s.get("action")) not in IGNORED_FOR_N)
                       and isinstance(s.get("start_s"), (int, float))
                       and isinstance(s.get("end_s"), (int, float))),
                      key=lambda s: s["start_s"])
    atomic_action_count = len(segments)
    merged = []
    for s in segments:
        key = (s.get("canonical_action", s.get("action")), s.get("object"))
        if merged and merged[-1]["key"] == key and s["start_s"] <= merged[-1]["end_s"] + 4:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], s["end_s"])
            merged[-1]["evidence_times"] = sorted(set(merged[-1]["evidence_times"] + s.get("evidence_times", [])))
        else:
            merged.append({"key": key, "start_s": s["start_s"], "end_s": s["end_s"],
                           "evidence_times": list(s.get("evidence_times", []))})
    atomic_stage_keys = []
    all_segments_have_stage = bool(segments)
    for s in segments:
        stage = s.get("task_stage")
        if isinstance(stage, str) and stage.strip():
            key = stage.strip()
            if not atomic_stage_keys or atomic_stage_keys[-1] != key:
                atomic_stage_keys.append(key)
        else:
            all_segments_have_stage = False
    if atomic_stage_keys and all_segments_have_stage:
        # A stage label is the model's task-level grouping; preserve atomic
        # action count separately rather than silently conflating the two.
        n = len(atomic_stage_keys)
        n_method = "task_stage"
    else:
        n = len(merged) if merged else None
        n_method = "action_object_fallback"
    candidates = semantic.get("high_difficulty_candidates", [])
    rule_passed = []
    human_confirmed = []
    for c in candidates:
        typ = c.get("type") or c.get("H_type")
        evidence = c.get("evidence_frame_ids") or c.get("evidence_times") or c.get("evidence_intervals")
        reason = str(c.get("reason", ""))
        # Ordinary contact, tool use, or two-handed assembly is not enough.
        # Require textual evidence of the corresponding causal signal until a
        # dedicated detector supplies state/feedback annotations.
        causal = ((typ in {"interaction", "交互"} and any(x in reason for x in ("反馈", "响应", "调整", "feedback", "response"))) or
                  (typ in {"conditional_decision", "条件决策"} and any(x in reason for x in ("条件", "判断", "决定", "condition", "decision"))) or
                  (typ in {"multi_thread_coordination", "多线程协调"} and any(x in reason for x in ("切换", "恢复", "协调", "返回", "switch", "recover"))))
        interval_ok = (isinstance(c.get("start_s"), (int, float)) and
                       isinstance(c.get("end_s"), (int, float)) and
                       0 <= c["start_s"] < c["end_s"] <= duration_s)
        if causal and interval_ok and evidence:
            rule_passed.append(c)
            if c.get("human_confirmed") is True:
                human_confirmed.append(c)
    unknown_duration = sum(max(0, float(u.get("end_s", 0)) - float(u.get("start_s", 0)))
                         for u in semantic.get("unknown", []) if isinstance(u, dict))
    complete = semantic.get("coverage_complete") is True and not semantic.get("unknown") and bool(segments)
    if human_confirmed:
        level, review = "高", "confirmed"
    elif rule_passed:
        level, review = "待判定", "candidate_high"
    elif not complete or not n:
        level, review = "待判定", "pending"
    elif n <= 3:
        level, review = "低", "candidate"
    else:
        level, review = "中", "candidate"
    return {"T": round(duration_s, 3), "N": n, "H": len(human_confirmed),
            "H_type": [c.get("type") or c.get("H_type") for c in human_confirmed],
            "atomic_action_count": atomic_action_count,
            "task_stage_count": len(atomic_stage_keys) if atomic_stage_keys else None,
            "n_method": n_method,
            "model_candidate_count": len(candidates),
            "rule_passed_count": len(rule_passed),
            "human_confirmed_count": len(human_confirmed),
            "level": level, "confidence": "人工确认" if review == "confirmed" else "待人工复核",
            "review_status": review,
            "evidence_intervals": [{"start_s": c["start_s"], "end_s": c["end_s"],
                                     "evidence_times": c.get("evidence_times", [])} for c in human_confirmed],
            "unknown_duration": round(unknown_duration, 3),
            "merged_action_count": n,
            "reason": "存在人工确认的高难事件" if human_confirmed else
                      ("存在规则通过但尚未人工确认的高难候选" if rule_passed else
                       ("动作链不完整或证据不足" if review == "pending" else "未确认高难事件"))}

def merge_action_segments(segments, tolerance=0.5):
    """Merge duplicate actions produced by overlapping VLM windows.

    Actions with different task stages are never merged. For unlabelled
    actions, the canonical action/object and overlapping time/evidence are
    required, so adjacent but distinct task steps remain separate.
    """
    merged = []
    for source in sorted((s for s in segments if isinstance(s, dict)), key=lambda s: (s.get("start_s", float("inf")), s.get("end_s", float("inf")))):
        if not isinstance(source.get("start_s"), (int, float)) or not isinstance(source.get("end_s"), (int, float)):
            merged.append(source); continue
        source_stage = source.get("task_stage", "")
        source_key = (source.get("canonical_action", source.get("action")), source.get("object"), source_stage)
        found = None
        for current in reversed(merged):
            current_key = (current.get("canonical_action", current.get("action")), current.get("object"), current.get("task_stage", ""))
            if current_key != source_key:
                continue
            overlap = min(current.get("end_s", -float("inf")), source["end_s"]) - max(current.get("start_s", float("inf")), source["start_s"])
            evidence_overlap = set(current.get("evidence_frame_ids", [])) & set(source.get("evidence_frame_ids", []))
            if overlap >= -tolerance or evidence_overlap:
                found = current; break
        if found is None:
            merged.append(dict(source)); continue
        found["start_s"] = min(found["start_s"], source["start_s"])
        found["end_s"] = max(found["end_s"], source["end_s"])
        for field in ("evidence_frame_ids", "evidence_times"):
            found[field] = list(dict.fromkeys(found.get(field, []) + source.get(field, [])))
        for field in ("state_before", "state_after", "raw_action"):
            if not found.get(field) and source.get(field): found[field] = source[field]
    return merged

def merge_high_difficulty_candidates(candidates, tolerance=0.5):
    """Deduplicate candidate events repeated in overlapping windows."""
    merged = []
    for source in candidates:
        if not isinstance(source, dict):
            continue
        source_key = source.get("type") or source.get("H_type")
        found = None
        for current in reversed(merged):
            if (current.get("type") or current.get("H_type")) != source_key:
                continue
            if not all(isinstance(current.get(k), (int, float)) and isinstance(source.get(k), (int, float)) for k in ("start_s", "end_s")):
                continue
            overlap = min(current["end_s"], source["end_s"]) - max(current["start_s"], source["start_s"])
            evidence_overlap = set(current.get("evidence_frame_ids", [])) & set(source.get("evidence_frame_ids", []))
            if overlap >= -tolerance or evidence_overlap:
                found = current
                break
        if found is None:
            merged.append(dict(source))
            continue
        found["start_s"] = min(found["start_s"], source["start_s"])
        found["end_s"] = max(found["end_s"], source["end_s"])
        for field in ("evidence_frame_ids", "evidence_times"):
            found[field] = list(dict.fromkeys(found.get(field, []) + source.get(field, [])))
    return merged

def sample_video(video: str, interval: float, max_frames: int, frames_cache=None, min_frames: int = 4):
    import av
    import math
    from PIL import Image
    # FFmpeg decoder threads must not call back into Python during codec
    # destruction. PyAV's logging callback can deadlock while holding the GIL.
    av.logging.restore_default_callback()
    if interval <= 0 or max_frames <= 0 or min_frames <= 0:
        raise ValueError('interval, max_frames, and min_frames must be positive')
    min_frames = min(min_frames, max_frames)
    cache = Path(frames_cache) if frames_cache else None
    signature = {'video': str(Path(video).resolve()), 'size': Path(video).stat().st_size,
                 'mtime_ns': Path(video).stat().st_mtime_ns,
                 'interval': interval, 'max_frames': max_frames, 'min_frames': min_frames, 'version': 3}
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        index_path = cache / 'frames.json'
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index.get('signature') == signature:
                frames = []
                for item in index['frames']:
                    with Image.open(cache / item['file']) as image:
                        frames.append(image.convert('RGB'))
                return frames, [item['timestamp'] for item in index['frames']]
    container = av.open(video)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    fps = float(stream.average_rate or 30)
    # Cover the whole recording under the per-recording frame budget. The old
    # implementation stopped after max_frames and therefore only saw the
    # beginning of long recordings.
    duration = float(stream.duration * stream.time_base) if stream.duration else None
    frames, timestamps, records = [], [], []
    def retain(frame, target):
        timestamp = round(float(frame.time if frame.time is not None else target), 3)
        if timestamps and timestamp <= timestamps[-1]:
            return
        image = frame.to_image()
        frames.append(image); timestamps.append(timestamp)
        if cache:
            name = f'{len(frames)-1:04d}.jpg'
            image.save(cache / name, quality=95)
            records.append({'file': name, 'timestamp': timestamp, 'target': target})
        if len(frames) % 20 == 0:
            print(f'Sampled {len(frames)} frames, latest {timestamp:.3f}s', flush=True)
    if duration:
        last_time = max(0, duration - 1 / fps)
        # Short clips need enough temporal coverage to expose start, middle,
        # and tail states; long clips remain capped by the recording budget.
        natural_count = max(1, math.ceil(duration / interval))
        count = min(max_frames, max(min_frames, natural_count))
        if count > natural_count:
            # The minimum-frame rule is active: distribute extra samples over
            # the full clip rather than seeking past its end.
            targets = [i * last_time / max(1, count - 1) for i in range(count)]
        else:
            # Preserve the requested interval while always including the tail.
            targets = [i * interval for i in range(max(0, count - 1))] + [last_time]
        for target in targets:
            container.seek(int(target / float(stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                # Seeking returns a preceding keyframe; decode to the requested time.
                if frame.time is not None and frame.time + 1e-5 < target:
                    continue
                retain(frame, target)
                break
    else:
        every = max(1, round(interval * fps))
        for i, frame in enumerate(container.decode(stream)):
            if i % every == 0:
                retain(frame, i/fps)
            if len(frames) >= max_frames: break
    container.close()
    if cache:
        (cache / 'frames.json').write_text(json.dumps({'signature': signature, 'duration_s': duration,
            'frames': records}, ensure_ascii=False, indent=2))
    return frames, timestamps

def analyze(video: str, model_id: str, interval: float, max_frames: int, window_frames: int = 16,
            frames_cache=None, max_duration: float = 600.0, min_frames: int = 4,
            window_overlap: int = 2, model=None, processor=None, logger=None,
            log_path=None, run_id=None) -> dict:
    started = time.monotonic()
    stage_timings = {}
    if logger is None and log_path:
        logger = JsonlLogger(log_path, run_id=run_id)
    if logger:
        logger.emit("video_start", video=video, model=model_id)
    stage_started = time.monotonic()
    import av
    with av.open(video) as probe:
        duration_s = float(probe.streams.video[0].duration * probe.streams.video[0].time_base)
    stage_timings["probe_s"] = round(time.monotonic() - stage_started, 4)
    if logger:
        logger.emit("probe_complete", video=video, duration_s=duration_s,
                    probe_s=stage_timings["probe_s"])
    if duration_s > max_duration:
        stage_timings["total_s"] = round(time.monotonic() - started, 4)
        _timing_metrics(stage_timings, duration_s)
        result = {"video": video, "model": model_id, "duration_s": duration_s, "frame_count": 0,
                "timestamps_s": [], "status": "skipped_too_long",
                "semantic": {"difficulty": {"level": "待处理", "T": round(duration_s, 3)}},
                "review_queue": [{"reason": "recording_exceeds_max_duration",
                                  "duration_s": duration_s, "max_duration_s": max_duration}],
                "stage_timings_s": stage_timings,
                "resource_usage": resource_snapshot(),
                "elapsed_s": round(time.monotonic()-started, 2)}
        if logger:
            logger.emit("video_skipped", video=video, status=result["status"],
                        stage_timings_s=stage_timings)
        return result
    print('Sampling video...', flush=True)
    stage_started = time.monotonic()
    frames, timestamps = sample_video(video, interval, max_frames, frames_cache, min_frames)
    stage_timings["sampling_s"] = round(time.monotonic() - stage_started, 4)
    if logger:
        logger.emit("sampling_complete", video=video, frame_count=len(frames),
                    first_timestamp_s=timestamps[0] if timestamps else None,
                    last_timestamp_s=timestamps[-1] if timestamps else None,
                    sampling_s=stage_timings["sampling_s"])
    if not frames:
        raise ValueError('No video frames decoded')
    print(f'Sampled {len(frames)} frames, {timestamps[0]}–{timestamps[-1]} seconds', flush=True)
    model_reused = model is not None and processor is not None
    if not model_reused:
        model, processor, model_load_s = load_vlm(model_id)
        stage_timings["model_load_s"] = round(model_load_s, 4)
        if logger:
            logger.emit("model_load_complete", video=video, model=model_id,
                        model_load_s=stage_timings["model_load_s"])
    else:
        stage_timings["model_load_s"] = 0.0
        if logger:
            logger.emit("model_reused", video=video, model=model_id)
    instruction = (
        "请分析这些按时间顺序排列的第一视角视频帧。严格只输出 JSON，字段为 "
        "summary（中文一句话）、scene（场景）、objects（对象数组）、"
        "task_applicability（只能是 ego_task、non_task、mixed、unknown；判断是否存在连续的第一视角操作任务）、"
        "action_segments（最多12个主要阶段，每项含 start_s、end_s、task_stage、canonical_action、raw_action、object、state_before、state_after、confidence、"
        "evidence_frame_ids（只能从输入帧ID中选择的数组）），unknown（无法判断的区间及原因），"
        "high_difficulty_candidates（候选事件数组，每项含 type、start_s、end_s、evidence_frame_ids、reason）。"
        "canonical_action 必须严格使用以下原子动作之一：" + ", ".join(ATOMIC_ACTIONS) + "。"
        "动作不在词表时使用 others，并在 raw_action 保留原始描述；不要创造新 canonical_action。"
        "只描述画面证据，不预设场景或任务。综合全程的前后状态识别动作。"
        "每张输入图前都有唯一Frame id；evidence_frame_ids只能复制这些ID，不能自行创造。"
        "start_s/end_s是估计时间区间，可以落在相邻输入帧之间，但不能超出窗口。task_stage用于把连续原子动作归并为一个任务阶段；state_before/state_after描述可见状态变化。"
        "coverage_complete只能在视频从开头到结尾均有足够证据时为true，否则为false。"
        "同阶段连续重复可以合并，跨阶段或目标变化分别保留。"
        "候选高难事件仅限有外部反馈与响应的交互、有条件证据的决策、"
        "有目标切换/恢复/协调证据的多线程协调；普通接触、双手操作或多个对象不等于确认高难。"
        "没有候选则返回空数组；缺乏证据不能确认H=0。输出紧凑但完整的JSON，不要截断。"
        "无法判断时使用unknown并说明原因，不要猜测。"
    )
    review_queue = []
    window_timings = []
    status = 'candidate_semantics'; window_reports = []
    step = max(1, window_frames - max(0, min(window_overlap, window_frames - 1)))
    for begin in range(0, len(frames), step):
        end = min(begin + window_frames, len(frames))
        content = []
        for offset, (image, timestamp) in enumerate(zip(frames[begin:end], timestamps[begin:end])):
            content.extend([{"type": "text", "text": f"Frame id: {begin + offset}; timestamp: {timestamp} seconds."},
                            {"type": "image", "image": image}])
        content.append({"type": "text", "text": instruction})
        window_started = time.monotonic()
        inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=3072, do_sample=False)
        text = processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            parsed = json.loads(text[text.find("{"):text.rfind("}")+1])
            if not isinstance(parsed, dict): raise ValueError('JSON is not an object')
        except (ValueError, TypeError):
            parsed = {"raw_output": text}; status = 'partial_semantic_output'
            review_queue.append({'reason': 'invalid_or_truncated_window_json', 'window': [begin, end]})
        parsed = normalize_semantic(parsed, list(range(begin, end)), timestamps[begin:end], duration_s)
        window_timings.append({"window_index": len(window_reports),
                               "seconds": round(time.monotonic() - window_started, 4),
                               "frame_start": begin, "frame_end": end})
        if logger:
            logger.emit("window_complete", video=video, window_index=len(window_reports),
                        frame_start=begin, frame_end=end,
                        window_inference_s=window_timings[-1]["seconds"],
                        time_start_s=timestamps[begin], time_end_s=timestamps[end - 1])
        if parsed.get("validation_issues"):
            status = 'partial_semantic_output' if status == 'candidate_semantics' else status
            review_queue.append({'reason': 'schema_or_evidence_validation_issue',
                                 'window': [begin, end],
                                 'issues': parsed['validation_issues']})
        window_reports.append({'window_index': len(window_reports), 'frame_start': begin,
                               'frame_end': end, 'time_start': timestamps[begin],
                               'time_end': timestamps[end-1], 'semantic': parsed})
        print(f'Window {len(window_reports)} complete: {timestamps[begin]:.3f}–{timestamps[end-1]:.3f}s', flush=True)
        if frames_cache:
            (Path(frames_cache) / 'windows.json').write_text(json.dumps(window_reports, ensure_ascii=False, indent=2))
        if end >= len(frames):
            break
    aggregation_started = time.monotonic()
    flattened_actions = merge_action_segments(
        [s for x in window_reports for s in x['semantic'].get('action_segments', [])])
    flattened_candidates = merge_high_difficulty_candidates(
        [h for x in window_reports for h in x['semantic'].get('high_difficulty_candidates', [])])
    parsed = {
        'summary': '；'.join(x['semantic'].get('summary','') for x in window_reports if x['semantic'].get('summary')),
        'scene': next((x['semantic'].get('scene') for x in window_reports if x['semantic'].get('scene')), None),
        'objects': sorted({o for x in window_reports for o in x['semantic'].get('objects', []) if isinstance(o, str)}),
        'action_segments': flattened_actions,
        'unknown': [u for x in window_reports for u in x['semantic'].get('unknown', [])],
        'high_difficulty_candidates': flattened_candidates,
        'task_applicability': 'unknown',
        'validation_issues': [issue for x in window_reports for issue in x['semantic'].get('validation_issues', [])],
        'windows': window_reports,
    }
    coverage_values = [x['semantic'].get('coverage_complete') for x in window_reports]
    parsed['coverage_complete'] = True if coverage_values and all(x is True for x in coverage_values) else False
    applicability = {x['semantic'].get('task_applicability', 'unknown') for x in window_reports}
    applicability.discard('unknown')
    parsed['task_applicability'] = next(iter(applicability)) if len(applicability) == 1 else ('mixed' if applicability else 'unknown')
    if parsed['high_difficulty_candidates']:
        review_queue.append({'reason': 'high_difficulty_candidates_require_evidence_review'})
    # A model's generated wording or confidence is never a confirmation of H.
    parsed['difficulty'] = recording_difficulty(parsed, duration_s)
    review_queue.append({'reason': 'recording_level_merge_and_task_boundary_review_required'})
    stage_timings["window_inference_s"] = round(sum(x["seconds"] for x in window_timings), 4)
    stage_timings["aggregation_s"] = round(time.monotonic() - aggregation_started, 4)
    stage_timings["total_s"] = round(time.monotonic() - started, 4)
    _timing_metrics(stage_timings, duration_s)
    result = {"video": video, "model": model_id, "duration_s": duration_s, "frame_count": len(frames),
            "timestamps_s": timestamps, "semantic": parsed, 'status': status,
            'review_queue': review_queue, 'window_timings': window_timings,
            'stage_timings_s': stage_timings,
            'resource_usage': resource_snapshot(),
            'model_reused': model_reused,
            'elapsed_s': round(time.monotonic()-started, 2)}
    if logger:
        logger.emit("video_complete", video=video, status=status,
                    frame_count=len(frames), stage_timings_s=stage_timings,
                    task_applicability=parsed.get("task_applicability"),
                    level=parsed.get("difficulty", {}).get("level"))
    return result


def analyze_many(videos, model_id, interval=2.0, max_frames=300, window_frames=16,
                 frames_cache_root=None, max_duration=600.0, min_frames=4,
                 window_overlap=2, logger=None, log_path=None, run_id=None):
    """Analyze multiple videos while loading the VLM exactly once."""
    if logger is None and log_path:
        logger = JsonlLogger(log_path, run_id=run_id)
    if logger:
        logger.emit("batch_start", video_count=len(videos), model=model_id)
    batch_started = time.monotonic()
    model, processor, model_load_s = load_vlm(model_id)
    if logger:
        logger.emit("batch_model_load_complete", model=model_id, model_load_s=round(model_load_s, 4))
    results = []
    for video in videos:
        cache = None
        if frames_cache_root:
            cache = str(Path(frames_cache_root) / Path(video).stem)
        video_started = time.monotonic()
        try:
            result = analyze(video, model_id, interval, max_frames, window_frames,
                             cache, max_duration, min_frames, window_overlap,
                             model=model, processor=processor, logger=logger)
        except Exception as exc:  # keep the resident batch alive and explain the failure
            result = {
                "video": video, "model": model_id, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "review_queue": [{"reason": "batch_item_failed", "error_type": type(exc).__name__}],
                "stage_timings_s": {"total_s": round(time.monotonic() - video_started, 4)},
                "resource_usage": resource_snapshot(),
                "model_reused": True,
            }
            result["stage_timings_s"]["video_duration_s"] = None
            if logger:
                logger.emit("video_failed", video=video, error=result["error"],
                            stage_timings_s=result["stage_timings_s"])
        timing = result.setdefault("stage_timings_s", {})
        _timing_metrics(timing, timing.get("video_duration_s"))
        timing["shared_model_load_s"] = round(model_load_s, 4)
        results.append(result)
    if logger:
        logger.emit("batch_complete", video_count=len(videos),
                    batch_wall_time_s=round(time.monotonic() - batch_started, 4))
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--interval", type=float, default=2.0)
    # 300 is the per-recording cloud budget from PPU_CONTEXT. Sampling stays
    # at the requested 2 s whenever the recording fits within that budget.
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--window-frames", type=int, default=16,
                    help='frames per VLM window; windows are merged at recording level')
    ap.add_argument("--window-overlap", type=int, default=2,
                    help='overlap frames between adjacent VLM windows (default: 2)')
    ap.add_argument("--min-frames", type=int, default=4,
                    help='minimum evenly distributed frames for short recordings (default: 4)')
    ap.add_argument('--frames-cache', help='cache selected frames and window outputs; never caches the original video')
    ap.add_argument('--max-duration', type=float, default=600.0,
                    help='skip recordings longer than this many seconds (default: 600)')
    ap.add_argument('--log', help='JSONL event log; defaults to <out>.log.jsonl')
    ap.add_argument('--report-md', help='Markdown report path; defaults to <out>.report.md')
    ap.add_argument('--report-html', help='HTML timeline report path; defaults to <out>.report.html')
    args = ap.parse_args()
    cli_started = time.monotonic()
    log_path = args.log or str(Path(args.out).with_suffix('.log.jsonl'))
    logger = JsonlLogger(log_path, run_id=Path(args.out).stem)
    try:
        report = analyze(args.video, args.model, args.interval, args.max_frames, args.window_frames,
                         args.frames_cache, args.max_duration, args.min_frames, args.window_overlap,
                         logger=logger)
    except Exception as exc:
        report = {
            "video": args.video, "model": args.model, "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "review_queue": [{"reason": "pipeline_failed", "error_type": type(exc).__name__}],
            "stage_timings_s": {"total_s": round(time.monotonic() - cli_started, 4)},
            "resource_usage": resource_snapshot(),
        }
        if logger:
            logger.emit("video_failed", video=args.video, error=report["error"],
                        stage_timings_s=report["stage_timings_s"])
    _timing_metrics(report.setdefault("stage_timings_s", {}),
                    report.get("stage_timings_s", {}).get("video_duration_s"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    report_paths = write_report_files(report, args.out, args.report_md, args.report_html)
    report["report_paths"] = report_paths
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.emit("report_written", video=args.video, report_paths=report_paths)
    print(json.dumps({"status": report['status'], "out": args.out,
                      "frame_count": report.get("frame_count", 0),
                      "log": log_path, **report_paths}, ensure_ascii=False))
    if report.get("status") == "failed":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
