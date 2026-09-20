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
        pipeline.load_vlm = lambda model_id: ("MODEL", "PROCESSOR", 1.25)
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
        pipeline.load_vlm = lambda model_id: ("MODEL", "PROCESSOR", 1.0)
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
    print("P2 regression checks passed")


if __name__ == "__main__":
    main()
