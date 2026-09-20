#!/usr/bin/env python3
"""Compare a fixed regression set without rerunning model inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def summarize_report(report):
    semantic = report.get("semantic", report)
    issues = semantic.get("validation_issues", [])
    action_segments = semantic.get("action_segments", [])
    difficulty = semantic.get("difficulty", {})
    invalid_evidence = sum(1 for a in action_segments
                           if not isinstance(a, dict) or not a.get("evidence_frame_ids"))
    return {
        "status": report.get("status", "unknown"),
        "schema_issue_count": len(issues),
        "invalid_evidence_count": invalid_evidence,
        "action_count": len(action_segments),
        "coverage_complete": bool(semantic.get("coverage_complete")),
        "level": difficulty.get("level", semantic.get("level")),
        "n": difficulty.get("N", semantic.get("N")),
        "h": difficulty.get("H", semantic.get("H")),
    }


def compare_reports(baseline, candidate):
    before, after = summarize_report(baseline), summarize_report(candidate)
    regressions = []
    improvements = []
    for key in ("schema_issue_count", "invalid_evidence_count"):
        if after[key] > before[key]: regressions.append({"metric": key, "before": before[key], "after": after[key]})
        elif after[key] < before[key]: improvements.append({"metric": key, "before": before[key], "after": after[key]})
    if before["status"] in {"candidate_semantics", "complete"} and after["status"] == "failed":
        regressions.append({"metric": "status", "before": before["status"], "after": after["status"]})
    return {"baseline": before, "candidate": after, "regressions": regressions, "improvements": improvements,
            "passed": not regressions}


def _read_report(root, sample_id):
    for name in (f"{sample_id}.json", f"{sample_id}.report.json"):
        path = Path(root) / name
        if path.exists(): return json.loads(path.read_text())
    raise FileNotFoundError(f"No report for {sample_id} under {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="directory with <sample_id>.json reports")
    ap.add_argument("--candidate", required=True, help="directory with <sample_id>.json reports")
    ap.add_argument("--sample", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for sid in args.sample:
        rows.append({"id": sid, **compare_reports(_read_report(args.baseline, sid), _read_report(args.candidate, sid))})
    result = {"sample_count": len(rows), "passed": all(x["passed"] for x in rows), "samples": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"sample_count": result["sample_count"], "passed": result["passed"]}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
