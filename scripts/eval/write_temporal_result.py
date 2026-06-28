#!/usr/bin/env python3
import json
import math
import sys
import argparse
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    dedup = {}
    for i, row in enumerate(rows):
        idx = row.get("index")
        qid = str(row.get("qid", "")).strip()
        key = f"idx:{idx}" if isinstance(idx, int) else (f"qid:{qid}" if qid else f"row:{i}")
        dedup[key] = row
    return list(dedup.values())


def as_rank(value):
    return int(value) if isinstance(value, int) and value > 0 else None


def rank_metrics(values):
    ranks = [r for r in (as_rank(v) for v in values) if r is not None]
    n = len(ranks)
    if n == 0:
        return {"N": 0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "R@100": 0.0, "MRR": 0.0}
    return {
        "N": n,
        "R@1": sum(r <= 1 for r in ranks) / n,
        "R@5": sum(r <= 5 for r in ranks) / n,
        "R@10": sum(r <= 10 for r in ranks) / n,
        "R@100": sum(r <= 100 for r in ranks) / n,
        "MRR": sum(1.0 / r for r in ranks) / n,
        "mean_rank": sum(ranks) / n,
    }


def as_float(value):
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def temporal_metrics(values):
    vals = [as_float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"N": 0, "mIoU@R1": 0.0, "IoU@0.3@R1": 0.0, "IoU@0.5@R1": 0.0, "IoU@0.7@R1": 0.0}
    return {
        "N": n,
        "mIoU@R1": sum(vals) / n,
        "IoU@0.3@R1": sum(v >= 0.3 for v in vals) / n,
        "IoU@0.5@R1": sum(v >= 0.5 for v in vals) / n,
        "IoU@0.7@R1": sum(v >= 0.7 for v in vals) / n,
    }


def pick_turn(row, turn_idx):
    turns = row.get("turns")
    if not isinstance(turns, list) or len(turns) < turn_idx:
        return {}
    turn = turns[turn_idx - 1]
    return turn if isinstance(turn, dict) else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_json")
    parser.add_argument("--source-label", default=None)
    args = parser.parse_args()

    src = Path(args.input_jsonl)
    dst = Path(args.output_json)
    rows = load_rows(src)
    max_turn = 1
    for row in rows:
        if isinstance(row.get("max_turn"), int):
            max_turn = max(max_turn, int(row["max_turn"]))
        turns = row.get("turns")
        if isinstance(turns, list):
            max_turn = max(max_turn, len(turns))

    result = {
        "source_jsonl": args.source_label or str(src),
        "num_examples": len(rows),
        "final_retrieval": rank_metrics([row.get("final_rank", row.get("rank")) for row in rows]),
        "original_retrieval": rank_metrics([row.get("orig_rank") for row in rows]),
        "final_temporal": temporal_metrics([
            row.get("final_iou_r1", row.get("final_iou_raw", row.get("final_iou", row.get("iou_r1", row.get("iou")))))
            for row in rows
        ]),
        "turns": {},
    }
    for turn_idx in range(1, max_turn + 1):
        turns = [pick_turn(row, turn_idx) for row in rows]
        result["turns"][str(turn_idx)] = {
            "retrieval": rank_metrics([turn.get("policy_rank", turn.get("rank")) for turn in turns]),
            "temporal": temporal_metrics([turn.get("iou_r1", turn.get("iou_raw", 0.0)) for turn in turns]),
        }

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[write_temporal_result] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
