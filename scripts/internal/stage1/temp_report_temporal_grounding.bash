#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash temp_report_temporal_grounding.bash /path/to/external_verified_test_temporal_grounding_*.jsonl
  JSONL=/path/to/file.jsonl bash temp_report_temporal_grounding.bash
  bash temp_report_temporal_grounding.bash /path/to/file.jsonl --watch 15
  bash temp_report_temporal_grounding.bash /path/to/file.jsonl --merge_jsonl /path/to/old.jsonl

Options:
  --watch SEC          Reprint report every SEC seconds.
  --merge_jsonl PATH   Also print merged metrics with another jsonl (dedup by qid/index, latest kept).
EOF
}

jsonl="${JSONL:-}"
watch_sec=""
merge_jsonl=""
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --watch)
      watch_sec="${2:-}"
      shift 2
      ;;
    --merge_jsonl)
      merge_jsonl="${2:-}"
      shift 2
      ;;
    *)
      if [[ -z "${jsonl}" ]]; then
        jsonl="$1"
      else
        echo "Unexpected arg: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${jsonl}" ]]; then
  echo "jsonl path is required (arg or JSONL env)." >&2
  usage
  exit 1
fi
if [[ ! -f "${jsonl}" ]]; then
  echo "jsonl not found: ${jsonl}" >&2
  exit 1
fi
if [[ -n "${merge_jsonl}" && ! -f "${merge_jsonl}" ]]; then
  echo "merge_jsonl not found: ${merge_jsonl}" >&2
  exit 1
fi
if [[ -n "${watch_sec}" ]]; then
  if ! [[ "${watch_sec}" =~ ^[0-9]+$ ]] || [[ "${watch_sec}" -le 0 ]]; then
    echo "--watch expects positive integer seconds." >&2
    exit 1
  fi
fi

run_once() {
  JSONL_PATH="$jsonl" MERGE_JSONL_PATH="$merge_jsonl" python3 - <<'PY'
import json
import os
import sys
from collections import Counter
from datetime import datetime
from statistics import median

jsonl_path = os.environ["JSONL_PATH"]
merge_jsonl = os.environ.get("MERGE_JSONL_PATH", "").strip()


def parse_rows(path):
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad


def dedup_rows(rows):
    by_key = {}
    for i, r in enumerate(rows):
        idx = r.get("index")
        qid = str(r.get("qid", "")).strip()
        if isinstance(idx, int):
            key = f"idx:{idx}"
        elif qid:
            key = f"qid:{qid}"
        else:
            key = f"row:{i}"
        by_key[key] = r
    return list(by_key.values())


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def safe_float(v, default=0.0):
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not (x == x):
        return float(default)
    return float(x)


def metrics_from_rank_values(vals):
    vals = [int(v) for v in vals if isinstance(v, int) and int(v) > 0]
    if not vals:
        return {}
    out = {
        "N": len(vals),
        "R@1": sum(1 for v in vals if v <= 1) / len(vals),
        "R@5": sum(1 for v in vals if v <= 5) / len(vals),
        "R@10": sum(1 for v in vals if v <= 10) / len(vals),
        "R@100": sum(1 for v in vals if v <= 100) / len(vals),
        "MRR": sum(1.0 / v for v in vals) / len(vals),
        "mean_rank": sum(vals) / len(vals),
        "median_rank": median(vals),
    }
    return out


def metrics_from_rank_key(rows, key):
    vals = [r.get(key) for r in rows if isinstance(r.get(key), int) and int(r.get(key)) > 0]
    return metrics_from_rank_values(vals)


def temporal_metrics(rows, key, thresholds=(0.3, 0.5, 0.7)):
    vals = []
    for r in rows:
        v = r.get(key, 0.0)
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        if not (fv >= 0.0):
            fv = 0.0
        vals.append(fv)
    if not vals:
        out = {"N": 0, "mIoU@R1": 0.0}
        for th in thresholds:
            out[f"IoU@{th:.1f}@R1"] = 0.0
        return out

    out = {
        "N": len(vals),
        "mIoU@R1": sum(vals) / len(vals),
    }
    for th in thresholds:
        out[f"IoU@{th:.1f}@R1"] = sum(1 for v in vals if v >= th) / len(vals)
    return out


def temporal_metrics_from_values(vals, matched_only_vals, thresholds=(0.3, 0.5, 0.7)):
    vals = [safe_float(v, 0.0) for v in vals]
    if not vals:
        out = {"N": 0, "mIoU@R1": 0.0, "mIoU_matched_only": 0.0}
        for th in thresholds:
            out[f"IoU@{th:.1f}@R1"] = 0.0
        return out
    out = {
        "N": len(vals),
        "mIoU@R1": sum(vals) / len(vals),
        "mIoU_matched_only": (sum(matched_only_vals) / len(matched_only_vals)) if matched_only_vals else 0.0,
    }
    for th in thresholds:
        out[f"IoU@{th:.1f}@R1"] = sum(1 for v in vals if v >= th) / len(vals)
    return out


def extract_policy_turns(row):
    turns = row.get("turns", [])
    if not isinstance(turns, list):
        return []
    parsed = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        policy_rank = t.get("policy_rank", None)
        rank_before = t.get("rank_before_refine", t.get("rank", None))
        if not isinstance(policy_rank, int) or policy_rank <= 0:
            if bool(t.get("refine_applied", False)):
                if i + 1 < len(turns) and isinstance(turns[i + 1], dict):
                    nxt_rank = turns[i + 1].get("rank", None)
                    if isinstance(nxt_rank, int) and nxt_rank > 0:
                        policy_rank = int(nxt_rank)
                if (not isinstance(policy_rank, int) or policy_rank <= 0) and isinstance(row.get("final_rank"), int):
                    policy_rank = int(row.get("final_rank"))
            if (not isinstance(policy_rank, int) or policy_rank <= 0) and isinstance(rank_before, int):
                policy_rank = int(rank_before)
        if not isinstance(policy_rank, int) or policy_rank <= 0:
            continue
        parsed.append(
            {
                "turn": safe_int(t.get("turn", i + 1), i + 1),
                "rank_before_refine": safe_int(rank_before, policy_rank),
                "policy_rank": int(policy_rank),
                "answer": str(t.get("answer", "unknown")).strip().lower() or "unknown",
                "time_parse_ok": bool(t.get("time_parse_ok", False)),
                "iou_raw": safe_float(t.get("iou_raw", 0.0), 0.0),
                "iou_r1": safe_float(t.get("iou_r1", 0.0), 0.0),
            }
        )
    return parsed


def collect_turn_series(rows):
    max_turn = 1
    for r in rows:
        mt = r.get("max_turn")
        if isinstance(mt, int) and mt > max_turn:
            max_turn = int(mt)
        turns = extract_policy_turns(r)
        if len(turns) > max_turn:
            max_turn = len(turns)

    policy_ranks = {t: [] for t in range(1, max_turn + 1)}
    strict_ranks = {t: [] for t in range(1, max_turn + 1)}
    policy_iou = {t: [] for t in range(1, max_turn + 1)}
    strict_iou = {t: [] for t in range(1, max_turn + 1)}
    policy_iou_matched = {t: [] for t in range(1, max_turn + 1)}
    strict_iou_matched = {t: [] for t in range(1, max_turn + 1)}

    for r in rows:
        turns = extract_policy_turns(r)
        if not turns:
            fallback_rank = r.get("final_rank")
            if not isinstance(fallback_rank, int) or fallback_rank <= 0:
                fallback_rank = r.get("orig_rank")
            if isinstance(fallback_rank, int) and fallback_rank > 0:
                turns = [
                    {
                        "turn": 1,
                        "rank_before_refine": int(fallback_rank),
                        "policy_rank": int(fallback_rank),
                        "answer": str(r.get("final_answer", "unknown")).strip().lower() or "unknown",
                        "time_parse_ok": bool(r.get("final_time_parse_ok", False)),
                        "iou_raw": safe_float(r.get("final_iou_raw", 0.0), 0.0),
                        "iou_r1": safe_float(r.get("final_iou_r1", 0.0), 0.0),
                    }
                ]
        if not turns:
            continue

        n_exec = len(turns)
        last = turns[-1]
        for t in range(1, max_turn + 1):
            td = turns[t - 1] if t <= n_exec else last
            policy_ranks[t].append(int(td["policy_rank"]))
            policy_iou[t].append(safe_float(td.get("iou_r1", 0.0), 0.0))
            if str(td.get("answer", "unknown")) == "matched" and bool(td.get("time_parse_ok", False)):
                policy_iou_matched[t].append(safe_float(td.get("iou_raw", 0.0), 0.0))

        for t in range(1, n_exec + 1):
            td = turns[t - 1]
            strict_ranks[t].append(int(td["policy_rank"]))
            strict_iou[t].append(safe_float(td.get("iou_r1", 0.0), 0.0))
            if str(td.get("answer", "unknown")) == "matched" and bool(td.get("time_parse_ok", False)):
                strict_iou_matched[t].append(safe_float(td.get("iou_raw", 0.0), 0.0))

    return {
        "max_turn": int(max_turn),
        "policy_ranks": policy_ranks,
        "strict_ranks": strict_ranks,
        "policy_iou": policy_iou,
        "strict_iou": strict_iou,
        "policy_iou_matched": policy_iou_matched,
        "strict_iou_matched": strict_iou_matched,
    }


def step_bucket(delta):
    if delta == 1:
        return "1"
    if delta == 2:
        return "2"
    if 3 <= delta <= 5:
        return "3-5"
    if 6 <= delta <= 10:
        return "6-10"
    if 11 <= delta <= 20:
        return "11-20"
    if 21 <= delta <= 50:
        return "21-50"
    if 51 <= delta <= 100:
        return "51-100"
    return "101+"


def collect_refine_effect_by_turn(rows, max_turn):
    out = {}
    for t in range(1, max_turn + 1):
        out[t] = {
            "applied": 0,
            "improved": 0,
            "same": 0,
            "worse": 0,
            "top1_after": 0,
            "improve_steps": [],
            "worse_steps": [],
            "improve_buckets": Counter(),
            "worse_buckets": Counter(),
        }

    for r in rows:
        raw_turns = r.get("turns", [])
        turns = extract_policy_turns(r)
        if not isinstance(raw_turns, list):
            raw_turns = []
        raw_by_turn = {}
        for i_raw, raw_t in enumerate(raw_turns):
            if not isinstance(raw_t, dict):
                continue
            raw_turn = safe_int(raw_t.get("turn", i_raw + 1), i_raw + 1)
            raw_by_turn[raw_turn] = raw_t
        for i, td in enumerate(turns):
            turn = safe_int(td.get("turn", i + 1), i + 1)
            if turn < 1 or turn > max_turn:
                continue
            raw_t = raw_by_turn.get(turn, {})
            refine_applied = bool(raw_t.get("refine_applied", False))
            if not refine_applied:
                continue

            rank_before = safe_int(td.get("rank_before_refine", 0), 0)
            rank_after = safe_int(td.get("policy_rank", 0), 0)
            if rank_before <= 0 or rank_after <= 0:
                continue

            st = out[turn]
            st["applied"] += 1
            if rank_after == 1:
                st["top1_after"] += 1
            delta = rank_before - rank_after
            if delta > 0:
                st["improved"] += 1
                st["improve_steps"].append(int(delta))
                st["improve_buckets"][step_bucket(int(delta))] += 1
            elif delta == 0:
                st["same"] += 1
            else:
                d = int(-delta)
                st["worse"] += 1
                st["worse_steps"].append(d)
                st["worse_buckets"][step_bucket(d)] += 1
    return out


def answer_accuracy(rows, answer_key, gt_key):
    valid = 0
    correct = 0
    pred_match = 0
    pred_not = 0
    correct_match = 0
    correct_not = 0

    for r in rows:
        ans = str(r.get(answer_key, "unknown")).strip().lower()
        gt = r.get(gt_key, None)
        if ans not in {"matched", "not_matched"} or not isinstance(gt, bool):
            continue
        valid += 1
        if ans == "matched":
            pred_match += 1
            if gt:
                correct += 1
                correct_match += 1
        else:
            pred_not += 1
            if not gt:
                correct += 1
                correct_not += 1

    return {
        "N": int(valid),
        "overall_acc": (correct / valid) if valid else 0.0,
        "pred_matched": int(pred_match),
        "pred_not_matched": int(pred_not),
        "match_acc_pred": (correct_match / pred_match) if pred_match else 0.0,
        "not_match_acc_pred": (correct_not / pred_not) if pred_not else 0.0,
    }


def fmt_retrieval(title, m):
    if not m:
        print(f"{title}: N/A")
        return
    print(
        f"{title}: N={m['N']} | "
        f"R@1={m['R@1']:.4f} R@5={m['R@5']:.4f} R@10={m['R@10']:.4f} R@100={m['R@100']:.4f} | "
        f"MRR={m['MRR']:.4f} mean_rank={m['mean_rank']:.2f} median_rank={m['median_rank']}"
    )


def fmt_temporal(title, m, thresholds=(0.3, 0.5, 0.7)):
    if not m:
        print(f"{title}: N/A")
        return
    th_str = " ".join([f"IoU@{th:.1f}@R1={m[f'IoU@{th:.1f}@R1']:.4f}" for th in thresholds])
    print(f"{title}: N={m['N']} | {th_str} | mIoU@R1={m['mIoU@R1']:.4f}")


def report(rows, tag):
    total = len(rows)
    idxs = [r.get("index") for r in rows if isinstance(r.get("index"), int)]
    idx_min = min(idxs) if idxs else None
    idx_max = max(idxs) if idxs else None

    answer_counts = Counter(str(r.get("final_answer", "unknown")).strip().lower() or "unknown" for r in rows)
    stop_reason_counts = Counter(str(r.get("stop_reason", "unknown")).strip() or "unknown" for r in rows)
    matched_turn_counts = Counter(int(r.get("matched_turn", 0) or 0) for r in rows)

    orig_retr = metrics_from_rank_key(rows, "orig_rank")
    turn_series = collect_turn_series(rows)
    max_turn = int(turn_series["max_turn"])
    refine_effect = collect_refine_effect_by_turn(rows, max_turn)
    policy_retr = {
        t: metrics_from_rank_values(turn_series["policy_ranks"].get(t, []))
        for t in range(1, max_turn + 1)
    }
    strict_retr = {
        t: metrics_from_rank_values(turn_series["strict_ranks"].get(t, []))
        for t in range(1, max_turn + 1)
    }
    policy_temp = {
        t: temporal_metrics_from_values(
            turn_series["policy_iou"].get(t, []),
            turn_series["policy_iou_matched"].get(t, []),
        )
        for t in range(1, max_turn + 1)
    }
    strict_temp = {
        t: temporal_metrics_from_values(
            turn_series["strict_iou"].get(t, []),
            turn_series["strict_iou_matched"].get(t, []),
        )
        for t in range(1, max_turn + 1)
    }

    turn1_retr = policy_retr.get(1, {})
    final_retr = policy_retr.get(max_turn, {})
    turn1_temp = policy_temp.get(1, {})
    final_temp = policy_temp.get(max_turn, {})
    turn1_ans_acc = answer_accuracy(rows, "turn1_answer", "turn1_top1_is_gt")
    final_ans_acc = answer_accuracy(rows, "final_answer", "final_top1_is_gt")

    used_refine_any = sum(1 for r in rows if bool(r.get("used_refine_any", False)))
    refine_generated_turns = sum(int(r.get("refine_token_generated_turns", 0) or 0) for r in rows)
    refine_applied_turns = sum(int(r.get("refine_applied_turns", 0) or 0) for r in rows)

    print("=" * 100)
    print(f"[{tag}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"rows={total} index_range={idx_min}..{idx_max}")

    print("\n[Answer/Turn]")
    print(
        f"- final_answer: matched={answer_counts.get('matched', 0)} "
        f"not_matched={answer_counts.get('not_matched', 0)} unknown={answer_counts.get('unknown', 0)}"
    )
    print(
        f"- turn1_answer_acc: overall={turn1_ans_acc['overall_acc']:.4f} "
        f"match_acc(pred)={turn1_ans_acc['match_acc_pred']:.4f} "
        f"not_match_acc(pred)={turn1_ans_acc['not_match_acc_pred']:.4f} "
        f"(N={turn1_ans_acc['N']}, pred_match={turn1_ans_acc['pred_matched']}, pred_not={turn1_ans_acc['pred_not_matched']})"
    )
    print(
        f"- final_answer_acc: overall={final_ans_acc['overall_acc']:.4f} "
        f"match_acc(pred)={final_ans_acc['match_acc_pred']:.4f} "
        f"not_match_acc(pred)={final_ans_acc['not_match_acc_pred']:.4f} "
        f"(N={final_ans_acc['N']}, pred_match={final_ans_acc['pred_matched']}, pred_not={final_ans_acc['pred_not_matched']})"
    )
    mt0 = matched_turn_counts.get(0, 0)
    mt1 = matched_turn_counts.get(1, 0)
    mt2 = matched_turn_counts.get(2, 0)
    mt3plus = sum(v for k, v in matched_turn_counts.items() if isinstance(k, int) and k >= 3)
    print(
        f"- matched_turn: turn1={mt1} turn2={mt2} turn3+={mt3plus} never_matched={mt0}"
    )

    print("\n[Stop Reason]")
    for k, v in stop_reason_counts.most_common():
        print(f"- {k}: {v} ({pct(v, total):.2f}%)")

    print("\n[Refine]")
    print(f"- used_refine_any: {used_refine_any} / {total} ({pct(used_refine_any, total):.2f}%)")
    print(f"- refine_token_generated_turns: {refine_generated_turns}")
    print(f"- refine_applied_turns: {refine_applied_turns}")

    print("\n[Refine Effect by Turn]")
    bucket_order = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101+"]
    for t in range(1, max_turn + 1):
        st = refine_effect.get(t, {})
        applied = int(st.get("applied", 0))
        improved = int(st.get("improved", 0))
        same = int(st.get("same", 0))
        worse = int(st.get("worse", 0))
        top1_after = int(st.get("top1_after", 0))
        if applied <= 0:
            print(f"- turn{t}: applied=0")
            continue
        improve_steps = st.get("improve_steps", [])
        worse_steps = st.get("worse_steps", [])
        improve_mean = (sum(improve_steps) / len(improve_steps)) if improve_steps else 0.0
        worse_mean = (sum(worse_steps) / len(worse_steps)) if worse_steps else 0.0
        print(
            f"- turn{t}: applied={applied} | "
            f"improved={improved} ({pct(improved, applied):.2f}%) "
            f"same={same} ({pct(same, applied):.2f}%) "
            f"worse={worse} ({pct(worse, applied):.2f}%) | "
            f"top1_after={top1_after} ({pct(top1_after, applied):.2f}%)"
        )
        imp_b = st.get("improve_buckets", Counter())
        wor_b = st.get("worse_buckets", Counter())
        imp_line = " ".join([f"+{b}:{int(imp_b.get(b, 0))}" for b in bucket_order])
        wor_line = " ".join([f"-{b}:{int(wor_b.get(b, 0))}" for b in bucket_order])
        print(f"  improve_steps_mean={improve_mean:.2f} | {imp_line}")
        print(f"  worsen_steps_mean={worse_mean:.2f} | {wor_line}")

    print("\n[Retrieval]")
    fmt_retrieval("- orig_query_only", orig_retr)
    for t in range(1, max_turn + 1):
        fmt_retrieval(f"- policy@{t}", policy_retr.get(t, {}))
    print("- strict_by_turn")
    for t in range(1, max_turn + 1):
        fmt_retrieval(f"  strict_turn{t}", strict_retr.get(t, {}))
    if turn1_retr and final_retr:
        print(
            f"- delta(policy@{max_turn}-policy@1): "
            f"R@1={final_retr['R@1']-turn1_retr['R@1']:+.4f} "
            f"R@5={final_retr['R@5']-turn1_retr['R@5']:+.4f} "
            f"R@10={final_retr['R@10']-turn1_retr['R@10']:+.4f} "
            f"R@100={final_retr['R@100']-turn1_retr['R@100']:+.4f} "
            f"MRR={final_retr['MRR']-turn1_retr['MRR']:+.4f}"
        )

    print("\n[Temporal IoU@R1]")
    for t in range(1, max_turn + 1):
        fmt_temporal(f"- policy@{t}", policy_temp.get(t, {}))
    print("- strict_by_turn")
    for t in range(1, max_turn + 1):
        fmt_temporal(f"  strict_turn{t}", strict_temp.get(t, {}))
    if turn1_temp and final_temp:
        print(
            f"- delta(policy@{max_turn}-policy@1): "
            f"IoU@0.3@R1={final_temp['IoU@0.3@R1']-turn1_temp['IoU@0.3@R1']:+.4f} "
            f"IoU@0.5@R1={final_temp['IoU@0.5@R1']-turn1_temp['IoU@0.5@R1']:+.4f} "
            f"IoU@0.7@R1={final_temp['IoU@0.7@R1']-turn1_temp['IoU@0.7@R1']:+.4f} "
            f"mIoU@R1={final_temp['mIoU@R1']-turn1_temp['mIoU@R1']:+.4f}"
        )


rows, bad = parse_rows(jsonl_path)
rows = dedup_rows(rows)
if not rows:
    print("No valid rows yet.")
    if bad:
        print(f"bad_lines={bad}")
    sys.exit(0)

report(rows, tag="CURRENT")
if bad:
    print(f"[WARN] bad_json_lines={bad}")

if merge_jsonl:
    rows2, bad2 = parse_rows(merge_jsonl)
    merged = {}
    for r in rows:
        idx = r.get("index")
        qid = str(r.get("qid", "")).strip()
        key = f"idx:{idx}" if isinstance(idx, int) else f"qid:{qid}"
        merged[key] = r
    for r in dedup_rows(rows2):
        idx = r.get("index")
        qid = str(r.get("qid", "")).strip()
        key = f"idx:{idx}" if isinstance(idx, int) else f"qid:{qid}"
        merged[key] = r

    print()
    report(list(merged.values()), tag="MERGED(current + merge_jsonl)")
    if bad2:
        print(f"[WARN] merge_bad_json_lines={bad2}")
PY
}

if [[ -n "${watch_sec}" ]]; then
  while true; do
    clear || true
    run_once
    sleep "${watch_sec}"
  done
else
  run_once
  result_json="${RESULT_JSON:-${jsonl%.jsonl}.result.json}"
  python3 "${repo_root}/scripts/eval/write_temporal_result.py" "${jsonl}" "${result_json}"
fi
