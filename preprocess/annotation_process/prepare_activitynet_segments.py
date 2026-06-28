import argparse
import json
import os
import random
from typing import Optional


def resolve_video_path(video_base: str, video_id: str) -> Optional[str]:
    if video_id is None:
        return None
    if os.path.splitext(video_id)[1]:
        p = os.path.join(video_base, video_id)
        if os.path.exists(p):
            return p
    for ext in [".mp4", ".mkv", ".webm", ".avi"]:
        p = os.path.join(video_base, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default=None)
    ap.add_argument("--train_jsonl", default=None)
    ap.add_argument("--test_jsonl", default=None)
    ap.add_argument("--video_base", required=True)
    ap.add_argument(
        "--out_dir",
        default=os.path.join(os.environ.get("DATA_ROOT", "data"), "structured"),
    )
    ap.add_argument("--test_size", type=int, default=10)
    ap.add_argument("--test_limit", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig_score_th", type=float, default=0.0)
    ap.add_argument("--use_full_video", action="store_true")
    ap.add_argument("--train_limit", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    corpus = []
    seen_doc = set()
    seen_qid = set()

    def add_from_jsonl(path, query_out):
        total = 0
        valid = 0
        missing_video = 0
        bad_time = 0
        missing_desc = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue

                video_id = j.get("video")
                desc_id = str(j.get("desc_id", total))
                duration = float(j.get("duration", 0.0))
                fig_desc = (j.get("fig_desc") or "").strip()
                fig_score = float(j.get("fig_desc_score", 0.0))

                if args.use_full_video:
                    start = 0.0
                    end = duration if duration > 0 else 0.0
                else:
                    time = j.get("time")
                    if not time or len(time) < 2:
                        bad_time += 1
                        continue
                    start = float(time[0])
                    end = float(time[1])
                    if start < 0 or end <= start:
                        bad_time += 1
                        continue
                    if duration > 0 and end > duration:
                        end = min(end, duration)
                    if end <= start:
                        bad_time += 1
                        continue

                video_path = resolve_video_path(args.video_base, video_id)
                if not video_path:
                    missing_video += 1
                    continue

                doc_id = video_id if args.use_full_video else f"{video_id}|{start:.2f}|{end:.2f}|{desc_id}"
                if doc_id not in seen_doc:
                    corpus.append(
                        {
                            "doc_id": doc_id,
                            "video_id": video_id,
                            "video_path": video_path,
                            "start": start,
                            "end": end,
                            "duration": duration,
                        }
                    )
                    seen_doc.add(doc_id)
                    valid += 1

                if not fig_desc:
                    missing_desc += 1
                if fig_desc:
                    qid = f"{desc_id}_fig"
                    if qid in seen_qid:
                        qid = f"{qid}_{total}"
                    seen_qid.add(qid)
                    query_out.append(
                        {
                            "qid": qid,
                            "query": fig_desc,
                            "pos_doc_id": doc_id,
                            "weight": 1.0,
                            "fig_score": fig_score,
                        }
                    )
        return total, valid, missing_video, bad_time, missing_desc

    train_queries = []
    test_queries = []

    if args.train_jsonl or args.test_jsonl:
        train_stats = None
        test_stats = None
        if args.train_jsonl:
            train_stats = add_from_jsonl(args.train_jsonl, train_queries)
        if args.test_jsonl:
            test_stats = add_from_jsonl(args.test_jsonl, test_queries)
        random.seed(args.seed)
        random.shuffle(train_queries)
        random.shuffle(test_queries)
        if args.train_limit > 0:
            train_queries = train_queries[: args.train_limit]
        if args.test_limit > 0:
            test_queries = test_queries[: args.test_limit]
    else:
        if not args.input_jsonl:
            raise ValueError("Provide --input_jsonl or --train_jsonl/--test_jsonl.")
        all_queries = []
        all_stats = add_from_jsonl(args.input_jsonl, all_queries)
        random.seed(args.seed)
        random.shuffle(all_queries)
        test_size = min(args.test_size, len(all_queries))
        if args.test_limit > 0:
            test_size = min(args.test_limit, len(all_queries))
        test_queries = all_queries[:test_size]
        train_queries = all_queries[test_size:]
        if args.train_limit > 0:
            train_queries = train_queries[: args.train_limit]
        train_stats = all_stats
        test_stats = None

    corpus_path = os.path.join(args.out_dir, "corpus_segments.jsonl")
    train_path = os.path.join(args.out_dir, "train_queries.jsonl")
    test_path = os.path.join(args.out_dir, "test_queries.jsonl")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for ex in corpus:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_queries:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test_queries:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if args.train_jsonl or args.test_jsonl:
        if args.train_jsonl and train_stats:
            total, valid, missing_video, bad_time, missing_desc = train_stats
            print("[Train] Total lines:", total)
            print("[Train] Valid segments:", valid)
            print("[Train] Missing video:", missing_video)
            print("[Train] Bad time:", bad_time)
            print("[Train] Missing desc:", missing_desc)
        if args.test_jsonl and test_stats:
            total, valid, missing_video, bad_time, missing_desc = test_stats
            print("[Test] Total lines:", total)
            print("[Test] Valid segments:", valid)
            print("[Test] Missing video:", missing_video)
            print("[Test] Bad time:", bad_time)
            print("[Test] Missing desc:", missing_desc)
    else:
        total, valid, missing_video, bad_time, missing_desc = train_stats
        print("Total lines:", total)
        print("Valid segments:", valid)
        print("Missing video:", missing_video)
        print("Bad time:", bad_time)
        print("Missing desc:", missing_desc)
    print("Wrote:", corpus_path)
    print("Wrote:", train_path)
    print("Wrote:", test_path)

    for i, ex in enumerate(corpus[:5]):
        print(f"[Sample corpus {i}]", ex)
    for i, ex in enumerate(train_queries[:5]):
        print(f"[Sample train {i}]", ex)


if __name__ == "__main__":
    main()
