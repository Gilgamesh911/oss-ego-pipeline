"""P2 regression tests; model generation is mocked and never invoked."""
from pathlib import Path
import sys
import json
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import qwen_vl_pipeline as pipeline
from evaluate_gold import evaluate_records
from compare_regression import compare_reports


def main():
    # P2-1: gold-set evaluation must report action IoU metrics, omission,
    # high-difficulty precision/recall, level confusion, and source strata.
    records = [{
        "id": "s1", "source": "demo",
        "gold": {"task_applicability": "ego_task", "level": "低",
                 "actions": [{"start_s": 0, "end_s": 2, "canonical_action": "Reach", "object": "cup"}],
                 "high_difficulty": []},
        "prediction": {"semantic": {"task_applicability": "ego_task",
                 "difficulty": {"level": "低"},
                 "action_segments": [{"start_s": 0.2, "end_s": 1.8, "canonical_action": "Reach", "object": "cup"}],
                 "high_difficulty_candidates": []}},
    }]
    metrics = evaluate_records(records)
    assert metrics["action"]["precision"] == 1.0
    assert metrics["action"]["recall"] == 1.0
    assert metrics["action"]["mean_temporal_iou"] > 0.7
    assert metrics["by_source"]["demo"]["sample_count"] == 1
    assert metrics["level_confusion"]["低"]["低"] == 1

    # P2-2: the fixed regression comparator flags schema/evidence regressions
    # and passes improvements.
    baseline = {"status": "candidate_semantics", "semantic": {
        "validation_issues": [{"reason": "old"}],
        "action_segments": [{"evidence_frame_ids": [0]}],
        "difficulty": {"level": "待判定"}}}
    candidate = {"status": "candidate_semantics", "semantic": {
        "validation_issues": [],
        "action_segments": [{"evidence_frame_ids": [0]}],
        "difficulty": {"level": "待判定"}}}
    assert compare_reports(baseline, candidate)["passed"]
    broken = {"status": "failed", "semantic": {"validation_issues": [{}, {}], "action_segments": [{}]}}
    assert not compare_reports(baseline, broken)["passed"]

    # P2-3: resident batching loads the model once and reuses it for every
    # recording. This mocks inference so the test remains deterministic.
    calls = []
    old_load, old_analyze = pipeline.load_vlm, pipeline.analyze
    try:
        pipeline.load_vlm = lambda model_id, **kwargs: ("MODEL", "PROCESSOR", 1.25)
        def fake_analyze(video, *args, **kwargs):
            calls.append((video, kwargs.get("model"), kwargs.get("processor")))
            return {"video": video, "stage_timings_s": {}, "status": "candidate_semantics"}
        pipeline.analyze = fake_analyze
        results = pipeline.analyze_many(["a.mp4", "b.mp4"], "model-id")
    finally:
        pipeline.load_vlm, pipeline.analyze = old_load, old_analyze
    assert len(results) == 2 and len(calls) == 2
    assert calls[0][1:] == ("MODEL", "PROCESSOR")
    assert all(r["stage_timings_s"]["shared_model_load_s"] == 1.25 for r in results)

    # A single bad video must be retained as an explicit failed item instead
    # of aborting the resident batch and hiding which input failed.
    old_load, old_analyze = pipeline.load_vlm, pipeline.analyze
    try:
        pipeline.load_vlm = lambda model_id, **kwargs: ("MODEL", "PROCESSOR", 1.0)
        def failing_analyze(video, *args, **kwargs):
            if video == "bad.mp4":
                raise ValueError("decode failed")
            return {"video": video, "stage_timings_s": {}, "status": "candidate_semantics"}
        pipeline.analyze = failing_analyze
        failed_results = pipeline.analyze_many(["bad.mp4", "good.mp4"], "model-id")
    finally:
        pipeline.load_vlm, pipeline.analyze = old_load, old_analyze
    assert failed_results[0]["status"] == "failed"
    assert "decode failed" in failed_results[0]["error"]
    assert failed_results[1]["status"] == "candidate_semantics"

    # P2-4: resource snapshots are always serializable and expose CUDA state.
    snapshot = pipeline.resource_snapshot()
    assert isinstance(snapshot["cuda_available"], bool)

    # P2-4: single-video CLI failures are persisted as structured reports,
    # rather than disappearing as an uncaught traceback.
    with tempfile.TemporaryDirectory(prefix="p2_failure_") as temp:
        out = Path(temp) / "failed.json"
        failed_process = subprocess.run([sys.executable, str(ROOT / "qwen_vl_pipeline.py"),
                        "--video", str(Path(temp) / "missing.mp4"),
                        "--model", "unused", "--out", str(out)], check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert failed_process.returncode != 0
        failure = json.loads(out.read_text())
        assert failure["status"] == "failed"
        assert failure["review_queue"][0]["reason"] == "pipeline_failed"
        assert Path(temp, "failed.log.jsonl").exists()
        assert Path(temp, "failed.report.md").exists()
        assert Path(temp, "failed.report.html").exists()
        events = [json.loads(line)["event"] for line in Path(temp, "failed.log.jsonl").read_text().splitlines()]
        assert "video_failed" in events and "report_written" in events

    # P2-5: reports expose the requested script/model split, duration ratios,
    # task semantics, level, and evidence basis in both formats.
    synthetic = {
        "video": "/data/example.mp4", "model": "test-model", "status": "candidate_semantics",
        "frame_count": 4, "stage_timings_s": {
            "video_duration_s": 10.0, "total_s": 5.0, "model_load_s": 1.0,
            "window_inference_s": 3.0, "script_runtime_excluding_inference_s": 2.0,
            "script_overhead_excluding_model_s": 1.0, "end_to_end_to_video_ratio": 0.5,
            "model_inference_to_video_ratio": 0.3, "script_runtime_to_video_ratio": 0.2,
            "script_overhead_to_video_ratio": 0.1,
        },
        "semantic": {"scene": "table", "summary": "move cup", "objects": ["cup"],
                     "task_applicability": "ego_task", "action_segments": [],
                     "unknown": [], "validation_issues": [],
                     "difficulty": {"T": 10, "N": 1, "H": 0, "level": "低", "reason": "未确认高难事件"}},
        "review_queue": [{"reason": "recording_level_merge_and_task_boundary_review_required"}],
    }
    with tempfile.TemporaryDirectory(prefix="p2_report_") as temp:
        paths = pipeline.write_report_files(synthetic, Path(temp) / "result.json")
        md = Path(paths["markdown"]).read_text()
        html = Path(paths["html"]).read_text()
        assert "Script runtime excluding model inference" in md
        assert "0.2x" in md
        assert "table" in html and "Action timeline" in html

    # P2-6: frame-grounded evidence and partial coverage no longer turn a
    # valid action chain into an unconditional pending level.
    normalized = pipeline.normalize_semantic({
        "window_coverage": "complete", "task_applicability": "ego_task",
        "action_segments": [{"start_frame_id": "F000001", "end_frame_id": "F000002",
                              "canonical_action": "Move", "raw_action": "move",
                              "object": "box", "task_stage": "move",
                              "evidence_frame_ids": ["F000001", "F000002"]}],
    }, ["F000001", "F000002"], [0.0, 2.0], 2.0)
    assert normalized["action_segments"][0]["start_s"] == 0.0
    assert normalized["action_segments"][0]["end_s"] == 2.0
    partial_semantic = {
        "task_applicability": "ego_task", "coverage": {"status": "partial", "score": 0.7},
        "coverage_complete": None,
        "action_segments": [{"canonical_action": "Move", "object": "box", "start_s": 0,
                              "end_s": 2, "task_stage": "move", "evidence_times": [0]}],
        "unknown": [], "high_difficulty_candidates": [],
    }
    partial_diff = pipeline.recording_difficulty(partial_semantic, 2)
    assert partial_diff["level"] == "低"
    assert partial_diff["review_status"] == "candidate_partial"
    print("P2 regression checks passed")


if __name__ == "__main__":
    main()
