"""Fast regression checks for the P0 output boundary.

These tests use the real malformed sample-02 window shape captured during the
previous batch, plus deterministic synthetic cases for frame grounding and
applicability filtering. They do not load the VLM.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_vl_pipeline import recording_difficulty, normalize_semantic


def main():
    raw_windows = json.loads((ROOT / "outputs/demo_20260917/reports/sample_02.windows.raw.json").read_text())
    malformed = raw_windows[0]["semantic"]
    normalized = normalize_semantic(malformed, list(range(8)), [0, 2, 4, 6, 8, 10, 12, 14], 15.23)
    # The historical objects dictionaries must be safely normalized, not crash
    # set aggregation. Unknown action vocabulary must be downgraded to others.
    assert all(isinstance(x, str) for x in normalized["objects"])
    safe_objects = sorted({x for x in normalized["objects"] if isinstance(x, str)})
    assert safe_objects

    grounded = normalize_semantic({
        "objects": ["瓶"],
        "action_segments": [{
            "start_s": 0, "end_s": 4, "canonical_action": "Speak",
            "raw_action": "说话", "object": "人", "confidence": .9,
            "evidence_frame_ids": [0, 2, 99],
        }],
    }, [0, 1, 2], [0.0, 2.0, 4.0], 4.0)
    action = grounded["action_segments"][0]
    assert action["canonical_action"] == "others"
    assert action["evidence_frame_ids"] == [0, 2]
    assert any(x["reason"] == "unknown_frame_id" for x in grounded["validation_issues"])

    legacy_time = normalize_semantic({
        "action_segments": [{"start_s": 0, "end_s": 4, "canonical_action": "Move",
                              "raw_action": "移动", "object": "手", "confidence": .8,
                              "evidence_times": [1.0, 3.0]}]},
        [0, 1, 2], [0.0, 2.0, 4.0], 4.0)
    assert legacy_time["action_segments"][0]["evidence_frame_ids"] == []
    assert any(x["reason"] == "no_matching_input_frame" for x in legacy_time["validation_issues"])

    assert recording_difficulty({"task_applicability": "non_task", "action_segments": []}, 10)["level"] == "不适用"
    print("P0 regression checks passed")


if __name__ == "__main__":
    main()
