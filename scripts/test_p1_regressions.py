"""Deterministic P1 regressions; no VLM weights or model generation are used."""
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_vl_pipeline import (merge_action_segments, merge_high_difficulty_candidates,
                              recording_difficulty, sample_video)


def segment(action, obj, start, end, stage, evidence):
    return {
        "canonical_action": action,
        "raw_action": action,
        "object": obj,
        "start_s": start,
        "end_s": end,
        "task_stage": stage,
        "state_before": "before",
        "state_after": "after",
        "evidence_frame_ids": evidence,
        "evidence_times": [float(x) for x in evidence],
    }


def main():
    # P1-1: atomic actions in one task phase must not inflate task-level N.
    difficulty = recording_difficulty({
        "task_applicability": "ego_task",
        "coverage_complete": True,
        "action_segments": [
            segment("Reach", "cup", 0, 1, "pick_up", [0]),
            segment("Grasp", "cup", 1, 2, "pick_up", [1]),
            segment("Lift", "cup", 2, 3, "pick_up", [2]),
            segment("Place", "table", 4, 5, "place", [4]),
        ],
        "unknown": [],
    }, 6)
    assert difficulty["atomic_action_count"] == 4
    assert difficulty["N"] == 2
    assert difficulty["n_method"] == "task_stage"

    # P1-2: high-difficulty model candidates and rule-passed candidates must
    # not be silently reported as human-confirmed H.
    candidate = {
        "type": "interaction", "start_s": 1, "end_s": 2,
        "evidence_frame_ids": [1], "evidence_times": [1.0],
        "reason": "外部反馈后调整动作",
    }
    candidate_result = recording_difficulty({
        "task_applicability": "ego_task", "coverage_complete": True,
        "action_segments": [segment("Move", "box", 0, 3, "move", [0])],
        "high_difficulty_candidates": [candidate], "unknown": [],
    }, 3)
    assert candidate_result["model_candidate_count"] == 1
    assert candidate_result["rule_passed_count"] == 1
    assert candidate_result["H"] == 0
    assert candidate_result["review_status"] == "candidate_high"
    confirmed = dict(candidate, human_confirmed=True)
    confirmed_result = recording_difficulty({
        "task_applicability": "ego_task", "coverage_complete": True,
        "action_segments": [segment("Move", "box", 0, 3, "move", [0])],
        "high_difficulty_candidates": [confirmed], "unknown": [],
    }, 3)
    assert confirmed_result["H"] == 1
    assert confirmed_result["review_status"] == "confirmed"

    # P1-3: empty unknown[] is not proof that the full recording was covered.
    incomplete = recording_difficulty({
        "task_applicability": "ego_task", "action_segments": [segment("Move", "box", 0, 1, "move", [0])],
        "unknown": [],
    }, 10)
    assert incomplete["review_status"] == "pending"

    # P1-4: overlapping windows may duplicate the same action, but different
    # task stages must remain separate.
    merged = merge_action_segments([
        segment("Insert", "plug", 1, 3, "insert", [1, 2]),
        segment("Insert", "plug", 2, 4, "insert", [2, 3]),
        segment("Insert", "plug", 3.5, 5, "remove", [4]),
    ])
    assert len(merged) == 2
    assert merged[0]["start_s"] == 1 and merged[0]["end_s"] == 4
    candidates = merge_high_difficulty_candidates([
        {"type": "interaction", "start_s": 1, "end_s": 2, "evidence_frame_ids": [1]},
        {"type": "interaction", "start_s": 1.5, "end_s": 2.5, "evidence_frame_ids": [1, 2]},
    ])
    assert len(candidates) == 1 and candidates[0]["end_s"] == 2.5

    # P1-5: short clips receive a minimum distributed frame budget and retain
    # a tail frame, instead of only returning two endpoint-ish samples.
    short_video = "/mnt/ego/擎羽/Hugging_face/category__small_object_storage_sorting/task_002__empty_all_sachets_from_box_onto_table/episode_0001_episode_000145_u12_aligned_attempt_clips_20260706_episode_000065/videos/left_cam_left.mp4"
    with tempfile.TemporaryDirectory(prefix="p1_frames_") as cache:
        frames, timestamps = sample_video(short_video, interval=2, max_frames=16, frames_cache=cache, min_frames=4)
        assert len(frames) >= 4
        assert timestamps[-1] > timestamps[0]
    print("P1 regression checks passed")


if __name__ == "__main__":
    main()
