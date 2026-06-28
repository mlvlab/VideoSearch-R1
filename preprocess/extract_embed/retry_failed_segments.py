import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


def _iter_jsonl(path: str, skip_bad: bool = False) -> Iterable[Dict[str, Any]]:
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                if not skip_bad:
                    raise
                bad += 1
                if bad <= 5:
                    print(f"[retry][warn] skip malformed jsonl line path={path} line={ln}")
    if skip_bad and bad > 0:
        print(f"[retry][warn] skipped malformed jsonl lines path={path} count={bad}")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return list(_iter_jsonl(path, skip_bad=True))


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_cfg(base_out_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(base_out_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _failed_doc_ids(failed_jsonl: str) -> Tuple[Set[str], List[Dict[str, Any]]]:
    rows = _read_jsonl(failed_jsonl)
    ids: Set[str] = set()
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id or doc_id == "__batch__":
            continue
        ids.add(doc_id)
    return ids, rows


def _build_failed_corpus(corpus_path: str, failed_ids: Set[str], out_path: str) -> Tuple[int, int, Set[str]]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    found: Set[str] = set()
    total = 0
    written = 0
    with open(corpus_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id in failed_ids:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                found.add(doc_id)
    return total, written, found


def _pick(cfg: Dict[str, Any], key: str, arg_val: Any, fallback: Any) -> Any:
    if arg_val is not None and arg_val != "":
        return arg_val
    if key in cfg and cfg[key] not in (None, ""):
        return cfg[key]
    return fallback


def _run_retry_embed_module(
    args: argparse.Namespace, cfg: Dict[str, Any], failed_corpus: str, retry_out_dir: str
) -> str:
    embed_model = str(_pick(cfg, "embed_model", args.embed_model, "Qwen/Qwen3-VL-Embedding-2B"))
    embed_device = _pick(cfg, "embed_device", args.embed_device, "")
    instruction = str(_pick(cfg, "instruction", args.instruction, "Represent the user's input"))

    sample_fps = float(_pick(cfg, "sample_fps", args.sample_fps, 1.0))
    sample_max_frames = int(_pick(cfg, "sample_max_frames", args.sample_max_frames, 64))
    num_frames = int(_pick(cfg, "num_frames", args.num_frames, 64))
    frame_size = int(_pick(cfg, "frame_size", args.frame_size, 0))
    max_input_tokens = int(_pick(cfg, "max_input_tokens", args.max_input_tokens, 16384))
    input_token_reserve = int(_pick(cfg, "input_token_reserve", args.input_token_reserve, 512))
    video_backend = str(_pick(cfg, "video_backend", args.video_backend, "opencv"))
    segment_batch_size = int(_pick(cfg, "segment_batch_size", args.segment_batch_size, 1))

    auto_sample_cfg = cfg.get("auto_sample_by_token_budget", True)
    if args.auto_sample_by_token_budget is None:
        auto_sample = bool(auto_sample_cfg)
    else:
        auto_sample = bool(args.auto_sample_by_token_budget)

    cmd = [
        args.python_bin,
        "-m",
        "extract_embed.embed_segments",
        "--corpus",
        failed_corpus,
        "--out_dir",
        retry_out_dir,
        "--embed_model",
        embed_model,
        "--instruction",
        instruction,
        "--sample_fps",
        str(sample_fps),
        "--sample_max_frames",
        str(sample_max_frames),
        "--num_frames",
        str(num_frames),
        "--frame_size",
        str(frame_size),
        "--max_input_tokens",
        str(max_input_tokens),
        "--input_token_reserve",
        str(input_token_reserve),
        "--video_backend",
        video_backend,
        "--segment_batch_size",
        str(max(1, segment_batch_size)),
        "--save_every",
        "1",
        "--limit",
        "0",
        "--shard_id",
        "0",
        "--num_shards",
        "1",
    ]

    if embed_device is not None and str(embed_device).strip():
        cmd.extend(["--embed_device", str(embed_device)])

    if auto_sample:
        cmd.append("--auto_sample_by_token_budget")
    else:
        cmd.append("--no_auto_sample_by_token_budget")

    if args.use_tqdm:
        cmd.append("--tqdm")
    else:
        cmd.append("--no_tqdm")

    env = os.environ.copy()
    cur_pp = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = "." if not cur_pp else f".:{cur_pp}"

    print("[retry] running(module):", " ".join(cmd))
    subprocess.check_call(cmd, cwd=args.repo_root, env=env)
    return retry_out_dir


def _infer_out_suffix(base_out_dir: str) -> str:
    name = os.path.basename(os.path.normpath(base_out_dir))
    prefix = "video_embedding_"
    if name.startswith(prefix) and len(name) > len(prefix):
        return name[len(prefix) :]
    return "1fps"


def _copy_query_files_for_split(src_split_dir: str, dst_split_dir: str) -> None:
    os.makedirs(dst_split_dir, exist_ok=True)
    copied = 0
    for qn in ("train_queries.jsonl", "val_queries.jsonl", "test_queries.jsonl"):
        src = os.path.join(src_split_dir, qn)
        dst = os.path.join(dst_split_dir, qn)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
    if copied == 0:
        # 02_embed_segments.bash only checks existence for default mode.
        open(os.path.join(dst_split_dir, "train_queries.jsonl"), "a", encoding="utf-8").close()


def _run_retry_embed_bash(args: argparse.Namespace, cfg: Dict[str, Any], failed_corpus: str, tmp_root: str) -> str:
    split = str(args.split).strip()
    if not split:
        split = os.path.basename(os.path.dirname(os.path.normpath(args.base_out_dir)))
        if split not in {"train", "val", "test"}:
            split = "train"

    out_suffix = str(args.out_suffix).strip() or _infer_out_suffix(args.base_out_dir)
    embed_model = str(_pick(cfg, "embed_model", args.embed_model, "Qwen/Qwen3-VL-Embedding-2B"))
    instruction = str(_pick(cfg, "instruction", args.instruction, "Represent the user's input"))
    sample_fps = float(_pick(cfg, "sample_fps", args.sample_fps, 1.0))
    sample_max_frames = int(_pick(cfg, "sample_max_frames", args.sample_max_frames, 64))
    num_frames = int(_pick(cfg, "num_frames", args.num_frames, 64))
    frame_size = int(_pick(cfg, "frame_size", args.frame_size, 0))
    max_input_tokens = int(_pick(cfg, "max_input_tokens", args.max_input_tokens, 16384))
    input_token_reserve = int(_pick(cfg, "input_token_reserve", args.input_token_reserve, 512))
    video_backend = str(_pick(cfg, "video_backend", args.video_backend, "opencv"))
    segment_batch_size = int(_pick(cfg, "segment_batch_size", args.segment_batch_size, 1))
    auto_sample_cfg = cfg.get("auto_sample_by_token_budget", True)
    if args.auto_sample_by_token_budget is None:
        auto_sample = bool(auto_sample_cfg)
    else:
        auto_sample = bool(args.auto_sample_by_token_budget)

    gpu_ids = str(args.gpu_ids).strip()
    if not gpu_ids:
        cfg_dev = str(cfg.get("embed_device", "")).strip()
        gpu_ids = cfg_dev if cfg_dev else "0"

    num_shards = args.num_shards
    if num_shards is None:
        cfg_shards = cfg.get("num_shards", 0)
        try:
            cfg_shards = int(cfg_shards)
        except Exception:
            cfg_shards = 0
        if cfg_shards > 0:
            num_shards = cfg_shards
        else:
            num_shards = max(1, len([x for x in gpu_ids.split(",") if x.strip()]))

    in_root = os.path.join(tmp_root, "structured_in")
    in_split_dir = os.path.join(in_root, split)
    out_root = os.path.join(tmp_root, "structured_out")
    out_dir = os.path.join(out_root, split, f"video_embedding_{out_suffix}")
    os.makedirs(in_split_dir, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    shutil.copy2(failed_corpus, os.path.join(in_split_dir, "corpus_segments.jsonl"))
    src_split_dir = os.path.dirname(os.path.normpath(args.corpus))
    _copy_query_files_for_split(src_split_dir, in_split_dir)

    env = os.environ.copy()
    env["INPUT_STRUCTURED_ROOT"] = in_root
    env["OUTPUT_STRUCTURED_ROOT"] = out_root
    env["SPLITS"] = split
    env["GPU_IDS"] = gpu_ids
    env["NUM_SHARDS"] = str(max(1, int(num_shards)))
    env["QWEN_VL_EMBED_ID"] = embed_model
    env["INSTRUCTION"] = instruction
    env["SAMPLE_FPS"] = str(sample_fps)
    env["SAMPLE_MAX_FRAMES"] = str(sample_max_frames)
    env["NUM_FRAMES"] = str(num_frames)
    env["FRAME_SIZE"] = str(frame_size)
    env["MAX_INPUT_TOKENS"] = str(max_input_tokens)
    env["INPUT_TOKEN_RESERVE"] = str(input_token_reserve)
    env["VIDEO_BACKEND"] = video_backend
    env["SEGMENT_BATCH_SIZE"] = str(max(1, segment_batch_size))
    env["AUTO_SAMPLE_BY_TOKEN_BUDGET"] = "1" if auto_sample else "0"
    env["USE_TQDM"] = "1" if args.use_tqdm else "0"
    env["LIMIT"] = "0"
    env["RESUME"] = "0"
    env["ONLY_TRAIN"] = "0"
    env["ONLY_QUERIES"] = "0"
    env["REUSE_FOLDER"] = ""
    env["LIVE_META"] = "1"
    env["KEEP_SHARD_DIRS"] = "0"
    env["PRESERVE_LIVE_LOGS_ON_RESUME"] = "1"
    env["OUT_SUFFIX"] = out_suffix
    env["PYTHON_BIN"] = args.python_bin

    cmd = ["bash", args.embed_bash]
    print("[retry] running(bash):", " ".join(cmd))
    print(
        f"[retry] bash env split={split} gpus={gpu_ids} shards={env['NUM_SHARDS']} "
        f"fps={sample_fps} max_frames={sample_max_frames} batch={segment_batch_size}"
    )
    subprocess.check_call(cmd, cwd=args.repo_root, env=env)
    return out_dir


def _run_retry_embed(args: argparse.Namespace, cfg: Dict[str, Any], failed_corpus: str, tmp_root: str) -> str:
    runner = str(args.runner).strip().lower()
    if runner == "bash":
        return _run_retry_embed_bash(args, cfg, failed_corpus, tmp_root)
    retry_out_dir = os.path.join(tmp_root, "retry_embed")
    if os.path.exists(retry_out_dir):
        shutil.rmtree(retry_out_dir, ignore_errors=True)
    os.makedirs(retry_out_dir, exist_ok=True)
    return _run_retry_embed_module(args, cfg, failed_corpus, retry_out_dir)


def _patch_embeddings(
    base_embed_path: str,
    base_map_path: str,
    retry_embed_path: str,
    retry_map_path: str,
) -> Tuple[Set[str], int, int]:
    if not os.path.exists(retry_embed_path) or not os.path.exists(retry_map_path):
        return set(), 0, 0

    with open(base_map_path, "r", encoding="utf-8") as f:
        base_map = json.load(f)
    with open(retry_map_path, "r", encoding="utf-8") as f:
        retry_map = json.load(f)

    if not isinstance(base_map, dict) or not isinstance(retry_map, dict):
        raise RuntimeError("invalid docid2row json format")

    base = np.load(base_embed_path, mmap_mode="r")
    retry = np.load(retry_embed_path, mmap_mode="r")

    if base.ndim != 2 or retry.ndim != 2:
        raise RuntimeError("embeddings must be 2D arrays")
    if int(base.shape[1]) != int(retry.shape[1]):
        raise RuntimeError(f"embedding dim mismatch: base={base.shape} retry={retry.shape}")

    replaced = 0
    append_docs: List[str] = []
    recovered: Set[str] = set()
    replace_pairs: List[Tuple[int, np.ndarray]] = []

    for doc_id, ridx in retry_map.items():
        r = int(ridx)
        if r < 0 or r >= int(retry.shape[0]):
            continue
        vec = np.asarray(retry[r], dtype=np.float32)
        recovered.add(str(doc_id))
        if doc_id in base_map:
            replace_pairs.append((int(base_map[doc_id]), vec))
        else:
            append_docs.append(str(doc_id))

    new_rows = int(base.shape[0]) + len(append_docs)
    dim = int(base.shape[1])
    tmp_embed = base_embed_path + ".retry_patch.tmp.npy"
    patched = np.lib.format.open_memmap(tmp_embed, mode="w+", dtype=np.float32, shape=(new_rows, dim))
    patched[: int(base.shape[0])] = np.asarray(base, dtype=np.float32)

    for row_idx, vec in replace_pairs:
        patched[row_idx] = vec
        replaced += 1

    cursor = int(base.shape[0])
    appended = 0
    for doc_id in append_docs:
        ridx = int(retry_map[doc_id])
        vec = np.asarray(retry[ridx], dtype=np.float32)
        patched[cursor] = vec
        base_map[doc_id] = int(cursor)
        cursor += 1
        appended += 1

    del patched
    os.replace(tmp_embed, base_embed_path)

    tmp_map = base_map_path + ".retry_patch.tmp.json"
    with open(tmp_map, "w", encoding="utf-8") as f:
        json.dump(base_map, f, ensure_ascii=False, indent=2)
    os.replace(tmp_map, base_map_path)

    return recovered, replaced, appended


def _append_recovered_meta(base_meta: str, retry_meta: str, recovered: Set[str]) -> int:
    if not os.path.exists(retry_meta) or not recovered:
        return 0
    existing: Set[str] = set()
    if os.path.exists(base_meta):
        for row in _iter_jsonl(base_meta, skip_bad=True):
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id:
                existing.add(doc_id)

    appended = 0
    with open(base_meta, "a", encoding="utf-8") as fout:
        for row in _iter_jsonl(retry_meta, skip_bad=True):
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id or doc_id not in recovered:
                continue
            if doc_id in existing:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(doc_id)
            appended += 1
    return appended


def _patch_failed_jsonl(
    old_failed_rows: List[Dict[str, Any]],
    retry_fail_path: str,
    failed_jsonl_path: str,
    recovered: Set[str],
) -> int:
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    def _push(row: Dict[str, Any]) -> None:
        doc_id = str(row.get("doc_id", "")).strip()
        err = str(row.get("error", "")).strip()
        key = (doc_id, err)
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    for row in old_failed_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id in recovered:
            continue
        _push(row)

    for row in _read_jsonl(retry_fail_path):
        _push(row)

    _write_jsonl(failed_jsonl_path, rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default="preprocess")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base_out_dir", required=True)
    ap.add_argument("--failed_jsonl", default="")
    ap.add_argument("--tmp_root", default="")
    ap.add_argument("--runner", choices=["bash", "module"], default="bash")
    ap.add_argument("--embed_bash", default="preprocess/scripts/02_embed_segments.bash")
    ap.add_argument("--split", default="")
    ap.add_argument("--out_suffix", default="")
    ap.add_argument("--gpu_ids", default="")
    ap.add_argument("--num_shards", type=int, default=None)
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--embed_model", default="")
    ap.add_argument("--embed_device", default="")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--sample_fps", type=float, default=None)
    ap.add_argument("--sample_max_frames", type=int, default=None)
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--frame_size", type=int, default=None)
    ap.add_argument("--max_input_tokens", type=int, default=None)
    ap.add_argument("--input_token_reserve", type=int, default=None)
    ap.add_argument("--video_backend", default="")
    ap.add_argument("--segment_batch_size", type=int, default=None)
    ap.add_argument("--auto_sample_by_token_budget", dest="auto_sample_by_token_budget", action="store_true")
    ap.add_argument("--no_auto_sample_by_token_budget", dest="auto_sample_by_token_budget", action="store_false")
    ap.set_defaults(auto_sample_by_token_budget=None)
    ap.add_argument("--tqdm", dest="use_tqdm", action="store_true")
    ap.add_argument("--no_tqdm", dest="use_tqdm", action="store_false")
    ap.set_defaults(use_tqdm=False)
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    base_out_dir = args.base_out_dir
    failed_jsonl = args.failed_jsonl or os.path.join(base_out_dir, "failed_docs.jsonl")
    base_embed = os.path.join(base_out_dir, "segment_embeds.npy")
    base_map = os.path.join(base_out_dir, "docid2row.json")
    base_meta = os.path.join(base_out_dir, "meta.jsonl")

    for path in [args.corpus, failed_jsonl, base_embed, base_map]:
        if not os.path.exists(path):
            raise SystemExit(f"Missing required path: {path}")

    failed_ids, old_failed_rows = _failed_doc_ids(failed_jsonl)
    if not failed_ids:
        print("[retry] no failed doc_id entries found")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_root = args.tmp_root or os.path.join(base_out_dir, f"retry_failed_tmp_{ts}")
    failed_corpus = os.path.join(tmp_root, "failed_corpus.jsonl")

    os.makedirs(tmp_root, exist_ok=True)

    total, written, found_ids = _build_failed_corpus(args.corpus, failed_ids, failed_corpus)
    missing_ids = failed_ids - found_ids
    print(f"[retry] corpus_rows={total} failed_ids={len(failed_ids)} subset_rows={written}")
    if missing_ids:
        print(f"[retry] warning: missing in corpus={len(missing_ids)}")

    if written == 0:
        print("[retry] no failed docs matched corpus; nothing to run")
        return

    cfg = _load_cfg(base_out_dir)
    retry_out_dir = _run_retry_embed(args, cfg, failed_corpus, tmp_root)

    retry_embed = os.path.join(retry_out_dir, "segment_embeds.npy")
    retry_map = os.path.join(retry_out_dir, "docid2row.json")
    retry_meta = os.path.join(retry_out_dir, "meta.jsonl")
    retry_fail = os.path.join(retry_out_dir, "failed_docs.jsonl")

    recovered, replaced, appended = _patch_embeddings(base_embed, base_map, retry_embed, retry_map)
    appended_meta = _append_recovered_meta(base_meta, retry_meta, recovered)
    remaining_failed_count = _patch_failed_jsonl(old_failed_rows, retry_fail, failed_jsonl, recovered)

    print(
        f"[retry] recovered={len(recovered)} replaced={replaced} appended={appended} "
        f"meta_appended={appended_meta} remaining_failed={remaining_failed_count}"
    )
    print(f"[retry] patched base_out_dir={base_out_dir}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"[retry] removed tmp_root={tmp_root}")
    else:
        print(f"[retry] kept tmp_root={tmp_root}")


if __name__ == "__main__":
    main()
