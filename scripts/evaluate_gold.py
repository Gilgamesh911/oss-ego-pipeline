#!/usr/bin/env python3
"""Evaluate predictions against an independently authored gold set.

The evaluator is deliberately model-agnostic. Gold JSON format:
{
  "samples": [{"id": "sample_01", "source": "scene",
    "gold": {"task_applicability": "ego_task", "level": "低",
      "actions": [{"start_s": 0, "end_s": 2,
                    "canonical_action": "Reach", "object": "cup"}],
      "high_difficulty": [{"start_s": 1, "end_s": 2, "type": "interaction"}]}}]
}
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def interval_iou(a, b):
    start = max(float(a.get("start_s", 0)), float(b.get("start_s", 0)))
    end = min(float(a.get("end_s", 0)), float(b.get("end_s", 0)))
    inter = max(0.0, end - start)
    union = max(float(a.get("end_s", 0)), float(b.get("end_s", 0))) - min(float(a.get("start_s", 0)), float(b.get("start_s", 0)))
    return inter / union if union > 0 else 0.0


def action_label(action):
    return (action.get("canonical_action", action.get("action")), action.get("object", "unknown"))


def match_segments(predicted, gold, iou_threshold=0.3):
    pairs = []
    for pi, p in enumerate(predicted or []):
        for gi, g in enumerate(gold or []):
            if action_label(p) != action_label(g):
                continue
            iou = interval_iou(p, g)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    matched_p, matched_g, matches = set(), set(), []
    for iou, pi, gi in sorted(pairs, reverse=True):
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi); matched_g.add(gi); matches.append((pi, gi, iou))
    return matches


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _candidate_matches(predicted, gold, iou_threshold=0.3):
    pairs = []
    for pi, p in enumerate(predicted or []):
        for gi, g in enumerate(gold or []):
            if (p.get("type") or p.get("H_type")) != (g.get("type") or g.get("H_type")):
                continue
            iou = interval_iou(p, g)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in sorted(pairs, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); matches.append((pi, gi, iou))
    return matches


def bootstrap_ci(values, seed=0, rounds=1000):
    values = [float(x) for x in values]
    if not values:
        return {"low": None, "high": None}
    if len(values) == 1:
        return {"low": values[0], "high": values[0]}
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(rounds)]
    means.sort()
    return {"low": round(means[int(0.025 * (len(means) - 1))], 4),
            "high": round(means[int(0.975 * (len(means) - 1))], 4)}


def evaluate_records(records, iou_threshold=0.3):
    totals = Counter()
    temporal_ious = []
    applicability = []
    level_confusion = Counter()
    high_totals = Counter()
    per_sample = []
    by_source = defaultdict(list)
    for record in records:
        gold = record.get("gold", {})
        pred = record.get("prediction", {})
        psemantic = pred.get("semantic", pred)
        matches = match_segments(psemantic.get("action_segments", []), gold.get("actions", []), iou_threshold)
        pcount = len(psemantic.get("action_segments", [])); gcount = len(gold.get("actions", []))
        totals.update({"action_tp": len(matches), "action_pred": pcount, "action_gold": gcount})
        temporal_ious.extend(iou for _, _, iou in matches)
        papp = psemantic.get("task_applicability", "unknown")
        gapp = gold.get("task_applicability", "unknown")
        applicability.append(int(papp == gapp))
        plevel = psemantic.get("difficulty", {}).get("level", psemantic.get("level"))
        glevel = gold.get("level")
        if glevel is not None:
            level_confusion[(glevel, plevel)] += 1
        pcandidates = psemantic.get("high_difficulty_candidates", [])
        gcandidates = gold.get("high_difficulty", [])
        hmatches = _candidate_matches(pcandidates, gcandidates, iou_threshold)
        high_totals.update({"tp": len(hmatches), "pred": len(pcandidates), "gold": len(gcandidates)})
        sample = {"id": record.get("id"), "source": record.get("source", "unknown"),
                  "action_tp": len(matches), "action_pred": pcount, "action_gold": gcount,
                  "temporal_iou": sum(temporal_ious[-len(matches):]) / len(matches) if matches else None,
                  "applicability_correct": bool(papp == gapp)}
        per_sample.append(sample); by_source[sample["source"]].append(sample)
    precision = totals["action_tp"] / totals["action_pred"] if totals["action_pred"] else 0.0
    recall = totals["action_tp"] / totals["action_gold"] if totals["action_gold"] else 0.0
    high_precision = high_totals["tp"] / high_totals["pred"] if high_totals["pred"] else 0.0
    high_recall = high_totals["tp"] / high_totals["gold"] if high_totals["gold"] else 0.0
    action_recall_samples = [s["action_tp"] / s["action_gold"] if s["action_gold"] else 1.0 for s in per_sample]
    confusion = defaultdict(Counter)
    for (gold_level, pred_level), count in level_confusion.items():
        confusion[gold_level][str(pred_level)] += count
    high_recall_samples = []
    for record in records:
        gold = record.get("gold", {})
        pred = record.get("prediction", {})
        semantic = pred.get("semantic", pred)
        gold_high = gold.get("high_difficulty", [])
        pred_high = semantic.get("high_difficulty_candidates", [])
        matches = _candidate_matches(pred_high, gold_high, iou_threshold)
        high_recall_samples.append(len(matches) / len(gold_high) if gold_high else 1.0)
    result = {
        "sample_count": len(records),
        "action": {"precision": round(precision, 4), "recall": round(recall, 4),
                    "f1": round(_f1(precision, recall), 4),
                    "omission_rate": round(1 - recall, 4),
                    "mean_temporal_iou": round(sum(temporal_ious) / len(temporal_ious), 4) if temporal_ious else None,
                    "recall_bootstrap_95ci": bootstrap_ci(action_recall_samples)},
        "task_applicability_accuracy": round(sum(applicability) / len(applicability), 4) if applicability else None,
        "task_applicability_bootstrap_95ci": bootstrap_ci(applicability),
        "high_difficulty": {"precision": round(high_precision, 4), "recall": round(high_recall, 4),
                            "f1": round(_f1(high_precision, high_recall), 4),
                            "recall_bootstrap_95ci": bootstrap_ci(high_recall_samples)},
        "level_confusion": {gold_level: dict(confusion[gold_level]) for gold_level in sorted(confusion)},
        "per_sample": per_sample,
        "by_source": {},
    }
    for source, samples in sorted(by_source.items()):
        result["by_source"][source] = {
            "sample_count": len(samples),
            "action_recall": round(sum((s["action_tp"] / s["action_gold"] if s["action_gold"] else 1.0) for s in samples) / len(samples), 4),
            "applicability_accuracy": round(sum(s["applicability_correct"] for s in samples) / len(samples), 4),
        }
    return result


def load_records(gold_path, predictions_path):
    gold = json.loads(Path(gold_path).read_text())
    gold_samples = gold.get("samples", []) if isinstance(gold, dict) else gold
    predictions = json.loads(Path(predictions_path).read_text()) if Path(predictions_path).is_file() else {}
    if isinstance(predictions, list):
        predictions = {str(x.get("id")): x.get("prediction", x) for x in predictions}
    records = []
    for sample in gold_samples:
        sid = str(sample.get("id"))
        prediction = predictions.get(sid, {})
        records.append({"id": sid, "source": sample.get("source", "unknown"),
                        "gold": sample.get("gold", sample), "prediction": prediction})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    args = ap.parse_args()
    result = evaluate_records(load_records(args.gold, args.predictions), args.iou_threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"samples": result["sample_count"], "action": result["action"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
