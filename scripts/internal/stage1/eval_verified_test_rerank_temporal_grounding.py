#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except Exception:
    LLM = None
    SamplingParams = None
    _VLLM_AVAILABLE = False

try:
    from tqdm.auto import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except Exception:
    _tqdm = None
    _TQDM_AVAILABLE = False


def _setup_import_paths() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3]
    videosearch_root = repo_root / "videosearch_r1"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(videosearch_root))


_setup_import_paths()


def _default_activitynet_path(*parts: str) -> str:
    return os.path.join(os.environ.get("VIDEOSEARCH_DATA_ROOT", "./data"), "activitynet", *parts)

from model.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration  # noqa: E402
from model.qwen_vl_utils.vision_process import process_vision_info  # noqa: E402
from utils.video_metadata import (  # noqa: E402
    load_video_meta_index,
    resolve_video_meta_for_video_path,
)


logger = logging.getLogger("eval_verified_test_rerank_temporal_grounding")

SYSTEM_PROMPT_TEMPLATE = (
    "You are a video retrieval assistant. Your task is to analyze a retrieved video against the user query. "
    "Inside <think>...</think>, perform a step by step comparison between the query requirements and the visible "
    "evidence in the video. Identify whether a scene corresponding to the query appears in the video and determine "
    "the exact time span where it occurs. If a scene corresponding to the query appears in the video, output "
    "strictly in the following format: <answer>matched</answer> <start>START_TIME_IN_SECONDS</start> "
    "<end>END_TIME_IN_SECONDS</end> {refine_suffix}. Even if matched, you must still append the special token(s) "
    "{refine_suffix} at the very end to allow further latent refinement. If no scene corresponding to the query "
    "appears in the video, output strictly: <answer>not_matched</answer> {refine_suffix}. In this case, the "
    "special token(s) are required to initiate a latent query update. You must always append the special token(s) "
    "{refine_suffix} at the very end of the output. Do not invent details beyond what is visible. Be concise inside "
    "<think>. Do not output anything outside the specified tags."
)
# SYSTEM_PROMPT_TEMPLATE = (
#     "You are a video retrieval assistant. Your task is to analyze a retrieved video against the user query. "
#     "Inside <think>...</think>, perform a step by step comparison between the query requirements and the visible "
#     "evidence in the video. Identify whether a scene corresponding to the query appears in the video and determine "
#     "the exact time span where it occurs. "
#     "If a scene corresponding to the query appears in the video, output strictly in the following format: "
#     "<think>...</think> <answer>matched</answer> <start>START_TIME_IN_SECONDS</start> "
#     "<end>END_TIME_IN_SECONDS</end>. "
#     "If no scene corresponding to the query appears in the video, output strictly: "
#     "<think>...</think> <answer>not_matched</answer> <REFINE>. "
#     "The <REFINE> token must appear only when the answer is not_matched, and must not appear when the answer is matched. "
#     "Do not invent details beyond what is visible. Be concise inside <think>. "
#     "Do not output anything outside the specified tags."
# )
ANSWER_RE = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
START_RE = re.compile(r"<start>\s*([^<]+?)\s*</start>", flags=re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"<end>\s*([^<]+?)\s*</end>", flags=re.IGNORECASE | re.DOTALL)
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _patch_vllm_qwen3_loader_for_soft_refine(
    ignore_prefixes: Tuple[str, ...] = (
        "refine_projector",
        "refine_gate",
        "refine_latent_input_projector",
        "refine_append_input_projector",
        "query_embedder_head",
        "query_embedder_model",
    ),
) -> None:
    """
    vLLM's Qwen3-VL loader can fail on extra soft-refine tensors in checkpoint.
    Patch once to ignore known unexpected prefixes, mirroring GRPO trainer behavior.
    """
    if not _VLLM_AVAILABLE:
        return
    try:
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLForConditionalGeneration as _VLLMQwen3VLForConditionalGeneration,
        )
        from vllm.model_executor.models.utils import AutoWeightsLoader
    except Exception as exc:
        logger.warning("vLLM qwen3 loader patch unavailable: %s", exc)
        return

    model_cls = _VLLMQwen3VLForConditionalGeneration
    patch_flag = "_soft_refine_ignore_patch_done"
    if getattr(model_cls, patch_flag, False):
        return

    def patched_load_weights(self, weights):
        skip_prefixes = []
        if getattr(self, "visual", None) is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
            ignore_unexpected_prefixes=list(ignore_prefixes),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    model_cls.load_weights = patched_load_weights
    setattr(model_cls, patch_flag, True)
    logger.info("Patched vLLM Qwen3-VL loader to ignore soft-refine tensors.")


def _key_matches_prefix(name: str, prefixes: Tuple[str, ...]) -> bool:
    key = str(name or "")
    for p in prefixes:
        pref = str(p or "").strip()
        if not pref:
            continue
        if key == pref or key.startswith(pref + "."):
            return True
    return False


def _safe_symlink_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _prepare_vllm_compatible_model_dir(
    model_path: str,
    ignore_prefixes: Tuple[str, ...] = (
        "refine_projector",
        "refine_gate",
        "refine_latent_input_projector",
        "refine_append_input_projector",
        "query_embedder_head",
        "query_embedder_model",
    ),
) -> str:
    """
    Build a vLLM-compatible model directory by removing soft-refine-only tensors.
    This avoids vLLM worker-process loader failures under spawn mode.
    """
    src_dir = os.path.abspath(str(model_path))
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"model_path not found: {src_dir}")

    dst_dir = os.path.join(src_dir, "_vllm_compat")
    marker_path = os.path.join(dst_dir, "_compat_meta.json")

    src_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    src_files_set = set(src_files)
    required_passthrough_files = [
        fname
        for fname in (
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        if fname in src_files_set
    ]
    src_stat_signature: Dict[str, float] = {}
    shard_to_keys_all: Dict[str, List[str]] = {}
    index_obj: Dict[str, object] = {}
    index_name = ""

    def _sig_add(fname: str):
        p = os.path.join(src_dir, fname)
        if os.path.exists(p):
            src_stat_signature[fname] = float(os.path.getmtime(p))

    # Detect weight format (single-file > sharded).
    fmt = None
    if "model.safetensors" in src_files_set:
        fmt = "single_safetensors"
        _sig_add("model.safetensors")
    elif "pytorch_model.bin" in src_files_set:
        fmt = "single_bin"
        _sig_add("pytorch_model.bin")
    elif "model.safetensors.index.json" in src_files_set:
        fmt = "sharded_safetensors"
        _sig_add("model.safetensors.index.json")
    elif "pytorch_model.bin.index.json" in src_files_set:
        fmt = "sharded_bin"
        _sig_add("pytorch_model.bin.index.json")
    else:
        raise RuntimeError(f"No supported weight file found under {src_dir}")

    if fmt in {"sharded_safetensors", "sharded_bin"}:
        index_name = "model.safetensors.index.json" if fmt == "sharded_safetensors" else "pytorch_model.bin.index.json"
        src_index_path = os.path.join(src_dir, index_name)
        with open(src_index_path, "r", encoding="utf-8") as f:
            index_obj = json.load(f)
        weight_map = dict(index_obj.get("weight_map", {}))
        for k, shard in weight_map.items():
            shard_to_keys_all.setdefault(str(shard), []).append(str(k))
        for shard in shard_to_keys_all.keys():
            _sig_add(shard)

    expected_meta = {
        "source_model_path": src_dir,
        "format": fmt,
        "ignore_prefixes": list(ignore_prefixes),
        "source_stat_signature": src_stat_signature,
    }
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
            if old_meta == expected_meta:
                missing_required = [
                    fname
                    for fname in required_passthrough_files
                    if not os.path.exists(os.path.join(dst_dir, fname))
                ]
                if missing_required:
                    logger.warning(
                        "Cached vLLM-compatible dir is missing required files %s; rebuilding: %s",
                        missing_required,
                        dst_dir,
                    )
                else:
                    logger.info("Reusing cached vLLM-compatible model dir: %s", dst_dir)
                    return dst_dir
        except Exception:
            pass

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    weight_files_to_handle = set()

    def _link_non_weight_files():
        for fname in src_files:
            if fname in weight_files_to_handle:
                continue
            src_p = os.path.join(src_dir, fname)
            dst_p = os.path.join(dst_dir, fname)
            _safe_symlink_or_copy(src_p, dst_p)

    dropped_total = 0

    if fmt == "single_safetensors":
        weight_files_to_handle.add("model.safetensors")
        _link_non_weight_files()
        from safetensors.torch import load_file, save_file

        src_weight = os.path.join(src_dir, "model.safetensors")
        state = load_file(src_weight, device="cpu")
        kept = {}
        for k, v in state.items():
            if _key_matches_prefix(k, ignore_prefixes):
                dropped_total += 1
                continue
            kept[k] = v
        dst_weight = os.path.join(dst_dir, "model.safetensors")
        save_file(kept, dst_weight)
    elif fmt == "single_bin":
        weight_files_to_handle.add("pytorch_model.bin")
        _link_non_weight_files()
        src_weight = os.path.join(src_dir, "pytorch_model.bin")
        state = torch.load(src_weight, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError("Unsupported pytorch_model.bin format: expected state_dict dict")
        kept = {}
        for k, v in state.items():
            if _key_matches_prefix(k, ignore_prefixes):
                dropped_total += 1
                continue
            kept[k] = v
        dst_weight = os.path.join(dst_dir, "pytorch_model.bin")
        torch.save(kept, dst_weight)
    elif fmt in {"sharded_safetensors", "sharded_bin"}:
        weight_files_to_handle.add(index_name)
        weight_files_to_handle.update(shard_to_keys_all.keys())
        _link_non_weight_files()

        new_weight_map = {}
        shards_rewritten = 0
        for shard, keys_all in shard_to_keys_all.items():
            keep_keys = [k for k in keys_all if not _key_matches_prefix(k, ignore_prefixes)]
            drop_keys = [k for k in keys_all if _key_matches_prefix(k, ignore_prefixes)]
            dropped_total += len(drop_keys)
            src_shard = os.path.join(src_dir, shard)
            dst_shard = os.path.join(dst_dir, shard)

            if not keep_keys:
                continue
            if not drop_keys:
                _safe_symlink_or_copy(src_shard, dst_shard)
            else:
                shards_rewritten += 1
                if fmt == "sharded_safetensors":
                    from safetensors.torch import load_file, save_file

                    shard_state = load_file(src_shard, device="cpu")
                    shard_kept = {k: shard_state[k] for k in keep_keys if k in shard_state}
                    save_file(shard_kept, dst_shard)
                else:
                    shard_state = torch.load(src_shard, map_location="cpu")
                    if not isinstance(shard_state, dict):
                        raise RuntimeError(f"Unsupported shard format: {src_shard}")
                    shard_kept = {k: shard_state[k] for k in keep_keys if k in shard_state}
                    torch.save(shard_kept, dst_shard)
            for k in keep_keys:
                new_weight_map[k] = shard

        new_index_obj = dict(index_obj)
        new_index_obj["weight_map"] = new_weight_map
        dst_index_path = os.path.join(dst_dir, index_name)
        with open(dst_index_path, "w", encoding="utf-8") as f:
            json.dump(new_index_obj, f, ensure_ascii=False, indent=2)
        logger.info(
            "Prepared sharded vLLM-compatible dir: rewritten_shards=%d, dropped_keys=%d",
            int(shards_rewritten),
            int(dropped_total),
        )

    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(expected_meta, f, ensure_ascii=False, indent=2)
    missing_required = [
        fname for fname in required_passthrough_files if not os.path.exists(os.path.join(dst_dir, fname))
    ]
    if missing_required:
        raise RuntimeError(
            f"Prepared vLLM-compatible dir is missing required files {missing_required}: {dst_dir}"
        )
    logger.info(
        "Prepared vLLM-compatible model dir: %s (dropped_keys=%d, format=%s)",
        dst_dir,
        int(dropped_total),
        fmt,
    )
    return dst_dir


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def _resolve_refine_token(refine_token: str) -> str:
    token = str(refine_token or "<REFINE>").strip()
    if not token:
        token = "<REFINE>"
    return token


def _resolve_rollout_depth(refine_token_count: int) -> int:
    return int(max(1, int(refine_token_count)))


def _make_system_prompt(refine_token: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(refine_suffix=str(refine_token))


def _resolve_text_hidden_size_from_config(config_obj, default: int = 2048) -> Tuple[int, str]:
    text_cfg = getattr(config_obj, "text_config", None)
    candidates = [
        ("config.text_config.hidden_size", getattr(text_cfg, "hidden_size", None)),
        ("config.hidden_size", getattr(config_obj, "hidden_size", None)),
    ]
    for source, value in candidates:
        if value is None:
            continue
        try:
            hidden = int(value)
        except (TypeError, ValueError):
            continue
        if hidden > 0:
            return hidden, source
    return int(default), f"default({int(default)})"


def _resolve_text_hidden_size(model: torch.nn.Module, default: int = 2048) -> Tuple[int, str]:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        hidden, source = _resolve_text_hidden_size_from_config(cfg, default=default)
        if not source.startswith("default("):
            return hidden, source
    lm_head = getattr(model, "lm_head", None)
    in_features = getattr(lm_head, "in_features", None)
    if in_features is not None:
        try:
            hidden = int(in_features)
            if hidden > 0:
                return hidden, "model.lm_head.in_features"
        except (TypeError, ValueError):
            pass
    return int(default), f"default({int(default)})"


def _make_latent_input_projector(in_dim: int, out_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.LayerNorm(int(in_dim)),
        torch.nn.Linear(int(in_dim), int(out_dim)),
    )


def _init_partial_identity_linear(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        torch.nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            torch.nn.init.zeros_(linear.bias)
        diag_dim = min(int(linear.in_features), int(linear.out_features))
        if diag_dim > 0:
            idx = torch.arange(diag_dim, device=linear.weight.device)
            linear.weight[idx, idx] = 1.0


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _split_csv_paths(value: str) -> List[str]:
    items: List[str] = []
    for raw in str(value or "").split(","):
        p = str(raw).strip()
        if p:
            items.append(p)
    return items


def _load_done_indices_from_jsonl_paths(paths: List[str]) -> Tuple[set[int], int, int, int]:
    done_indices: set[int] = set()
    files_found = 0
    rows_loaded = 0
    bad_lines = 0
    for path in paths:
        p = str(path or "").strip()
        if not p or not os.path.exists(p):
            continue
        files_found += 1
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row_obj = json.loads(line)
                except Exception:
                    bad_lines += 1
                    continue
                idx_val = row_obj.get("index")
                if isinstance(idx_val, int):
                    done_indices.add(int(idx_val))
                    rows_loaded += 1
                    continue
                try:
                    idx_int = int(idx_val)
                except Exception:
                    continue
                done_indices.add(int(idx_int))
                rows_loaded += 1
    return done_indices, files_found, rows_loaded, bad_lines


def _l2_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _normalize_embedder_instruction(instruction: str) -> str:
    instr = str(instruction or "").strip()
    if instr and not unicodedata.category(instr[-1]).startswith("P"):
        instr = instr + "."
    return instr


def _build_query_embedder_text(tokenizer, query_text: str, instruction: str) -> str:
    q = str(query_text or "").strip()
    instr = _normalize_embedder_instruction(instruction)
    if not instr:
        return q
    if not q:
        return instr
    messages = [
        {"role": "system", "content": instr},
        {"role": "user", "content": q},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str) and rendered.strip():
            return rendered
    except Exception:
        pass
    return f"{instr}\n{q}"


def _encode_query_with_embedder(
    *,
    query_text: str,
    query_embedder_model,
    query_embedder_tokenizer,
    query_embedder_head,
    qfinal_pooling: str,
    query_embedder_max_length: int,
    query_embedder_input_prefix: str,
    retrieval_dim: int,
) -> torch.Tensor:
    if query_embedder_model is None or query_embedder_tokenizer is None:
        raise RuntimeError("query embedder is not initialized.")
    if not str(query_text or "").strip():
        raise ValueError("query_text is empty for updated-query encoding.")

    embedder_query_text = _build_query_embedder_text(
        query_embedder_tokenizer,
        query_text,
        query_embedder_input_prefix,
    )
    tokenized = query_embedder_tokenizer(
        [embedder_query_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(query_embedder_max_length),
    )
    device = next(query_embedder_model.parameters()).device
    input_ids = tokenized["input_ids"].to(device=device)
    attention_mask = tokenized["attention_mask"].to(device=device)

    with torch.no_grad():
        outputs = query_embedder_model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state.to(dtype=torch.float32)

    if str(qfinal_pooling).lower() == "mean":
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
    else:
        pooled = hidden[:, -1, :]

    if query_embedder_head is not None:
        try:
            q_head_dtype = next(query_embedder_head.parameters()).dtype
        except StopIteration:
            q_head_dtype = pooled.dtype
        pooled = query_embedder_head(pooled.to(dtype=q_head_dtype))

    q_vec = pooled.squeeze(0).to(dtype=torch.float32)
    if q_vec.size(-1) != int(retrieval_dim):
        raise ValueError(
            "updated query embedding dim mismatch: "
            f"{q_vec.size(-1)} vs expected {int(retrieval_dim)}"
        )
    return F.normalize(q_vec, p=2, dim=-1)


def _resolve_initial_query_vector(
    *,
    use_updated_query: bool,
    query_row: int,
    query_text: str,
    query_embeddings: np.ndarray,
    updated_query_cache: Dict[int, torch.Tensor],
    query_embedder_model,
    query_embedder_tokenizer,
    query_embedder_head,
    qfinal_pooling: str,
    query_embedder_max_length: int,
    query_embedder_input_prefix: str,
    retrieval_dim: int,
    device: torch.device,
) -> torch.Tensor:
    q_row = int(query_row)
    if not bool(use_updated_query):
        return torch.from_numpy(query_embeddings[q_row]).to(device=device, dtype=torch.float32)

    cached = updated_query_cache.get(q_row)
    if cached is not None:
        return cached

    q_vec = _encode_query_with_embedder(
        query_text=query_text,
        query_embedder_model=query_embedder_model,
        query_embedder_tokenizer=query_embedder_tokenizer,
        query_embedder_head=query_embedder_head,
        qfinal_pooling=qfinal_pooling,
        query_embedder_max_length=int(query_embedder_max_length),
        query_embedder_input_prefix=str(query_embedder_input_prefix or ""),
        retrieval_dim=int(retrieval_dim),
    ).to(device=device, dtype=torch.float32)
    updated_query_cache[q_row] = q_vec
    return q_vec


def _compute_rank_from_scores(scores: torch.Tensor, pos_row: int) -> int:
    pos_score = scores[pos_row]
    return int((scores > pos_score).sum().item() + 1)


def _metrics_from_ranks(ranks: List[int], ks: List[int]) -> Dict[str, float]:
    arr = np.asarray(ranks, dtype=np.int64)
    if arr.size == 0:
        out: Dict[str, float] = {
            "count": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
        }
        for k in ks:
            out[f"R@{k}"] = 0.0
        return out

    out = {
        "count": float(len(arr)),
        "mean_rank": float(np.mean(arr)),
        "median_rank": float(np.median(arr)),
        "mrr": float(np.mean(1.0 / arr)),
    }
    for k in ks:
        out[f"R@{k}"] = float(np.mean(arr <= k))
    return out


def _parse_rerank_topk(value: str, max_docs: int) -> Optional[int]:
    s = str(value).strip().lower()
    if s in {"all", "none", "0", "-1"}:
        return None
    try:
        k = int(s)
    except Exception as exc:
        raise ValueError(f"Invalid --rerank_topk={value}") from exc
    if k < 1:
        raise ValueError(f"--rerank_topk must be >=1 or 'all', got: {value}")
    return int(min(k, max_docs))


def _invert_docid2row(docid2row: Dict[str, int]) -> List[str]:
    n = max(docid2row.values()) + 1
    row2doc = [""] * n
    for doc, row in docid2row.items():
        row_i = int(row)
        if 0 <= row_i < n:
            row2doc[row_i] = str(doc)
    return row2doc


def _maybe_load_refine_weights(model: torch.nn.Module, model_path: str) -> bool:
    if not os.path.isdir(model_path):
        return False
    prefix = (
        "refine_projector.",
        "refine_gate.",
        "refine_latent_input_projector.",
        "refine_append_input_projector.",
        "query_embedder_head.",
    )
    refine_state = {}

    def _collect_from_state_dict(sd):
        for k, v in sd.items():
            if k.startswith(prefix):
                refine_state[k] = v

    try:
        safetensor_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensor_path):
            from safetensors.torch import load_file

            _collect_from_state_dict(load_file(safetensor_path, device="cpu"))
    except Exception as exc:
        logger.warning(f"Failed loading model.safetensors for refine weights: {exc}")

    if not refine_state:
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            try:
                _collect_from_state_dict(torch.load(bin_path, map_location="cpu"))
            except Exception as exc:
                logger.warning(f"Failed loading pytorch_model.bin for refine weights: {exc}")

    if not refine_state:
        index_candidates = [
            os.path.join(model_path, "model.safetensors.index.json"),
            os.path.join(model_path, "pytorch_model.bin.index.json"),
        ]
        for index_path in index_candidates:
            if not os.path.exists(index_path):
                continue
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_obj = json.load(f)
                weight_map = index_obj.get("weight_map", {})
                shard_files = {shard for k, shard in weight_map.items() if k.startswith(prefix)}
                for shard in shard_files:
                    shard_path = os.path.join(model_path, shard)
                    if not os.path.exists(shard_path):
                        continue
                    if shard.endswith(".safetensors"):
                        from safetensors.torch import load_file

                        _collect_from_state_dict(load_file(shard_path, device="cpu"))
                    else:
                        _collect_from_state_dict(torch.load(shard_path, map_location="cpu"))
            except Exception as exc:
                logger.warning(f"Failed loading sharded refine weights from {index_path}: {exc}")

    if "query_embedder_head.weight" in refine_state:
        q_w = refine_state["query_embedder_head.weight"]
        q_b = refine_state.get("query_embedder_head.bias", None)
        if isinstance(q_w, torch.Tensor) and q_w.ndim == 2:
            out_dim, in_dim = int(q_w.shape[0]), int(q_w.shape[1])
            head = getattr(model, "query_embedder_head", None)
            needs_replace = True
            if isinstance(head, torch.nn.Linear):
                has_bias = head.bias is not None
                needs_replace = not (
                    int(head.in_features) == in_dim
                    and int(head.out_features) == out_dim
                    and has_bias == (q_b is not None)
                )
            if needs_replace:
                model.query_embedder_head = torch.nn.Linear(in_dim, out_dim, bias=(q_b is not None))
                logger.info(
                    "Initialized query_embedder_head from checkpoint weights "
                    f"({in_dim} -> {out_dim}, bias={q_b is not None})"
                )

    if refine_state:
        model_state = model.state_dict()
        loadable = {}
        dropped = []
        for k, v in refine_state.items():
            target = model_state.get(k)
            if target is None:
                dropped.append(f"{k}:missing_target")
                continue
            if tuple(target.shape) != tuple(v.shape):
                dropped.append(f"{k}:shape_ckpt={tuple(v.shape)}!=model={tuple(target.shape)}")
                continue
            loadable[k] = v
        if dropped:
            preview = ", ".join(dropped[:5])
            logger.warning(
                "Skipped incompatible refine tensors while loading checkpoint "
                f"(dropped={len(dropped)}; examples={preview})"
            )
        if not loadable:
            logger.warning("No compatible refine tensors found in checkpoint.")
            return False
        missing, unexpected = model.load_state_dict(loadable, strict=False)
        logger.info(
            "Loaded refine weights from checkpoint "
            f"({len(loadable)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return True

    logger.warning("No refine weights found in checkpoint; using initialized refine modules.")
    return False


def _resolve_video_path(video_root: str, video_id: str) -> Optional[str]:
    vid = str(video_id or "").strip()
    root = str(video_root or "").strip()
    if not vid:
        return None

    if os.path.isabs(vid) and os.path.exists(vid):
        return vid

    exts = [".npy", ".npz", ".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"]
    cands: List[str] = []
    if root:
        cands.append(os.path.join(root, vid))
        cands.append(os.path.join(root, "test_video_npy", vid))
        for ext in exts:
            cands.append(os.path.join(root, f"{vid}{ext}"))
            cands.append(os.path.join(root, "test_video_npy", f"{vid}{ext}"))

    for p in cands:
        if os.path.exists(p):
            return p
    return None


def _resolve_video_meta_path(video_meta_path: str, video_root: str) -> str:
    hint = str(video_meta_path or "").strip()
    if hint and os.path.exists(hint):
        return hint
    root = str(video_root or "").strip()
    if not root:
        return ""
    cand1 = os.path.join(root, "meta.jsonl")
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(os.path.dirname(root.rstrip("/")), "meta.jsonl")
    if os.path.exists(cand2):
        return cand2
    return ""


def _load_query_meta(query_meta_path: str) -> Dict[str, Tuple[int, str, str]]:
    out: Dict[str, Tuple[int, str, str]] = {}
    for row_idx, row in enumerate(_iter_jsonl(query_meta_path)):
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        qid = str(row.get("qid", "")).strip()
        pos_doc_id = str(row.get("pos_doc_id", "")).strip()
        out[query] = (row_idx, qid, pos_doc_id)
    return out


def _parse_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    if not m:
        return "unknown"
    ans = m.group(1).strip().lower().replace(" ", "_")
    if ans in {"matched", "match"}:
        return "matched"
    if ans in {"not_matched", "notmatch", "not-matched", "mismatch"}:
        return "not_matched"
    return "unknown"


def _parse_float(raw: str) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = FLOAT_RE.search(text.replace(",", " "))
    if not m:
        return None
    try:
        v = float(m.group(0))
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _parse_temporal_span(text: str, duration: Optional[float] = None) -> Optional[Tuple[float, float]]:
    m_start = START_RE.search(text or "")
    m_end = END_RE.search(text or "")
    if not m_start or not m_end:
        return None
    start = _parse_float(m_start.group(1))
    end = _parse_float(m_end.group(1))
    if start is None or end is None:
        return None
    if duration is not None and np.isfinite(duration):
        duration_v = float(max(0.0, duration))
        start = min(max(start, 0.0), duration_v)
        end = min(max(end, 0.0), duration_v)
    if end < start:
        start, end = end, start
    return float(start), float(end)


def _compute_iou(gt_span: Tuple[float, float], pred_span: Tuple[float, float], eps: float = 1e-9) -> float:
    gs, ge = float(gt_span[0]), float(gt_span[1])
    ps, pe = float(pred_span[0]), float(pred_span[1])
    if ge < gs:
        gs, ge = ge, gs
    if pe < ps:
        ps, pe = pe, ps
    if (ge - gs) <= eps or (pe - ps) <= eps:
        return 0.0
    inter = max(0.0, min(ge, pe) - max(gs, ps))
    union = (ge - gs) + (pe - ps) - inter
    if union <= eps:
        return 0.0
    return float(max(0.0, min(1.0, inter / union)))


def _build_generation_inputs(
    processor,
    system_prompt: str,
    query_text: str,
    video_path: str,
    video_meta: Optional[dict],
    model_max_length: int,
    image_patch_size: int,
    video_min_pixels: int,
    video_max_pixels: int,
    video_total_pixels: int,
):
    video_payload = {
        "type": "video",
        "video": video_path,
        "min_pixels": int(video_min_pixels),
        "max_pixels": int(video_max_pixels),
        "total_pixels": int(video_total_pixels),
    }
    if isinstance(video_meta, dict) and video_meta:
        video_payload.update(video_meta)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'Query: "{query_text}"\nRetrieved video: '},
                video_payload,
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, packed_video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        return_video_metadata=True,
        image_patch_size=image_patch_size,
    )
    if packed_video_inputs:
        if isinstance(packed_video_inputs[0], tuple) and len(packed_video_inputs[0]) == 2:
            video_inputs, video_metadatas = zip(*packed_video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_inputs = packed_video_inputs
            video_metadatas = None
    else:
        video_inputs = None
        video_metadatas = None

    return {
        "messages": messages,
        "text": text,
        "image_inputs": image_inputs,
        "packed_video_inputs": packed_video_inputs,
        "video_inputs": video_inputs,
        "video_metadatas": video_metadatas,
        "video_kwargs": video_kwargs,
    }


def _build_generation_batch_from_inputs(
    processor,
    generation_inputs: Dict[str, object],
    model_max_length: int,
):
    processor_kwargs = dict(
        text=[generation_inputs["text"]],
        images=generation_inputs["image_inputs"],
        videos=generation_inputs["video_inputs"],
        return_tensors="pt",
        truncation=True,
        max_length=int(model_max_length),
        padding=True,
        **generation_inputs["video_kwargs"],
    )
    if generation_inputs["video_metadatas"] is not None:
        processor_kwargs["video_metadata"] = generation_inputs["video_metadatas"]
    batch = processor(**processor_kwargs)
    return batch


def _build_vllm_input_from_generation_inputs(generation_inputs: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {"prompt": str(generation_inputs["text"])}
    mm_data: Dict[str, object] = {}

    image_inputs = generation_inputs["image_inputs"]
    if image_inputs:
        mm_data["image"] = image_inputs

    packed_video_inputs = generation_inputs["packed_video_inputs"]
    video_inputs = generation_inputs["video_inputs"]
    if packed_video_inputs:
        mm_data["video"] = packed_video_inputs
    elif video_inputs:
        mm_data["video"] = video_inputs

    if mm_data:
        out["multi_modal_data"] = mm_data
    if packed_video_inputs or video_inputs:
        out["mm_processor_kwargs"] = generation_inputs["video_kwargs"]
    return out


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _decode_completion_from_token_ids(tokenizer, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _extract_update_from_ids(
    model,
    model_inputs: Dict[str, torch.Tensor],
    full_ids: torch.Tensor,
    prompt_len: int,
    completion_ids: torch.Tensor,
    refine_token_ids: List[int],
    use_refine_gate: bool,
) -> Tuple[Optional[torch.Tensor], bool]:
    # [legacy full-completion hidden-state fallback note]
    # 아래 블록을 사용하면 completion 전체에서 <REFINE> 위치를 찾고,
    # full_ids(= prompt + completion 전체)로 hidden을 추출합니다.
    #
    # comp_mask = torch.zeros_like(completion_ids[0], dtype=torch.bool)
    # for tok_id in refine_token_ids:
    #     comp_mask |= completion_ids[0].eq(int(tok_id))
    # comp_ref_pos = torch.nonzero(comp_mask, as_tuple=False).squeeze(1)
    # had_refine = bool(comp_ref_pos.numel() > 0)
    # if not had_refine:
    #     return None, False
    # # 마지막 generated <REFINE>만 사용
    # comp_ref_pos = comp_ref_pos[-1:]

    last_comp_ref_pos = _find_last_refine_pos_in_completion(
        completion_ids=completion_ids,
        refine_token_ids=refine_token_ids,
    )
    had_refine = last_comp_ref_pos is not None
    if not had_refine:
        return None, False

    # Match training behavior by running the hidden-state forward only up to the
    # selected <REFINE> position (prompt + completion prefix).
    run_ids = _truncate_full_ids_to_last_refine(
        full_ids=full_ids,
        prompt_len=prompt_len,
        last_comp_ref_pos=int(last_comp_ref_pos),
    )
    # [legacy full-completion hidden-state fallback]
    # run_ids = full_ids
    fwd_inputs = {}
    for k, v in model_inputs.items():
        if k in {"input_ids", "attention_mask"}:
            continue
        fwd_inputs[k] = v
    fwd_inputs["input_ids"] = run_ids
    fwd_inputs["attention_mask"] = torch.ones_like(run_ids, device=run_ids.device)
    fwd_inputs["output_hidden_states"] = True

    with torch.no_grad():
        outputs = model(**fwd_inputs)
        hidden = outputs.hidden_states[-1]

    pos_in_full = torch.tensor([int(run_ids.size(1)) - 1], device=run_ids.device, dtype=torch.long)
    # [legacy full-completion hidden-state fallback]
    # pos_in_full = comp_ref_pos + int(prompt_len)
    try:
        projector_dtype = next(model.refine_projector.parameters()).dtype
    except StopIteration:
        projector_dtype = hidden.dtype
    h_ref = hidden[0, pos_in_full]
    delta = model.refine_projector(h_ref.to(dtype=projector_dtype)).to(dtype=torch.float32)

    if bool(use_refine_gate):
        try:
            gate_dtype = next(model.refine_gate.parameters()).dtype
        except StopIteration:
            gate_dtype = projector_dtype
        alpha = torch.sigmoid(model.refine_gate(h_ref.to(dtype=gate_dtype))).to(dtype=torch.float32)
        update = delta * alpha
    else:
        update = delta
    return update, had_refine


def _find_last_refine_pos_in_completion(
    completion_ids: torch.Tensor,
    refine_token_ids: List[int],
) -> Optional[int]:
    if completion_ids.numel() == 0:
        return None
    comp_mask = torch.zeros_like(completion_ids[0], dtype=torch.bool)
    for tok_id in refine_token_ids:
        comp_mask |= completion_ids[0].eq(int(tok_id))
    comp_ref_pos = torch.nonzero(comp_mask, as_tuple=False).squeeze(1)
    if comp_ref_pos.numel() == 0:
        return None
    return int(comp_ref_pos[-1].item())


def _truncate_full_ids_to_last_refine(
    full_ids: torch.Tensor,
    prompt_len: int,
    last_comp_ref_pos: int,
) -> torch.Tensor:
    end_exclusive = int(prompt_len) + int(last_comp_ref_pos) + 1
    end_exclusive = int(max(1, min(end_exclusive, int(full_ids.size(1)))))
    return full_ids[:, :end_exclusive]


def _build_rollout_inputs_with_full_ids(
    model_inputs: Dict[str, torch.Tensor],
    full_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    rollout_inputs = dict(model_inputs)
    rollout_inputs["input_ids"] = full_ids
    rollout_inputs["attention_mask"] = torch.ones_like(full_ids, device=full_ids.device)
    return rollout_inputs


def _inject_latent_tokens(
    query_token_embeds: torch.Tensor,
    latent_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    insert_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if latent_tokens.dim() != 3:
        raise ValueError(f"latent_tokens must be 3D [B,L,D], got shape={tuple(latent_tokens.shape)}")
    if query_token_embeds.size(0) != latent_tokens.size(0):
        raise ValueError(
            "batch size mismatch between query_token_embeds and latent_tokens: "
            f"{query_token_embeds.size(0)} vs {latent_tokens.size(0)}"
        )
    seq_len = int(query_token_embeds.size(1))
    if insert_idx is None:
        tail_len = int(min(6, max(seq_len, 0)))
        insert_idx = int(seq_len - tail_len)
    insert_idx = int(max(0, min(seq_len, int(insert_idx))))

    head_embeds = query_token_embeds[:, :insert_idx]
    tail_embeds = query_token_embeds[:, insert_idx:]
    inputs_embeds = torch.cat([head_embeds, latent_tokens, tail_embeds], dim=1)

    head_attn = attention_mask[:, :insert_idx]
    tail_attn = attention_mask[:, insert_idx:]
    latent_attn = torch.ones(
        (attention_mask.size(0), int(latent_tokens.size(1))),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    full_attention = torch.cat([head_attn, latent_attn, tail_attn], dim=1)
    return inputs_embeds, full_attention


def _rollout_refine_latents(
    *,
    model,
    first_update: torch.Tensor,
    rollout_depth: int,
    row_model_inputs: Optional[Dict[str, torch.Tensor]],
    refine_token_ids: List[int],
    use_refine_gate: bool,
    qfinal_pooling: str,
) -> torch.Tensor:
    if first_update.dim() == 1:
        first_update = first_update.unsqueeze(0)
    if first_update.dim() != 2:
        raise ValueError(f"first_update must be 1D/2D tensor, got shape={tuple(first_update.shape)}")
    depth = int(max(1, rollout_depth))
    z_seed = first_update.mean(dim=0, keepdim=True).to(dtype=torch.float32)
    # 커리큘럼: depth=1이면 seed 1개만 사용.
    if depth <= 1:
        return z_seed

    if row_model_inputs is None or "input_ids" not in row_model_inputs:
        return z_seed.repeat(depth, 1)

    base_model = model.module if hasattr(model, "module") else model
    vlm_core = getattr(base_model, "model", None)
    if vlm_core is None or not hasattr(base_model, "get_input_embeddings"):
        return z_seed.repeat(depth, 1)

    refine_projector = getattr(base_model, "refine_projector", None)
    refine_gate = getattr(base_model, "refine_gate", None)
    if refine_projector is None:
        return z_seed.repeat(depth, 1)
    latent_input_projector = getattr(base_model, "refine_latent_input_projector", None)

    input_ids = row_model_inputs["input_ids"]
    attention_mask = row_model_inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    query_token_embeds = base_model.get_input_embeddings()(input_ids)
    refine_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for tok_id in refine_token_ids:
        refine_mask |= input_ids.eq(int(tok_id))


    #Refeine 위치 찾아서
    positions = torch.nonzero(refine_mask[0], as_tuple=False).squeeze(-1)
    if positions.numel() > 0:
        vlm_insert_idx = int(positions[-1].item()) + 1 #Refeine 위치 바로 다음칸에 insert 
    else:
        vlm_insert_idx = int(query_token_embeds.size(1))

    extra_forward_inputs: Dict[str, torch.Tensor] = {}
    # [기존 로직 설명 - 주석 보존]
    # latent를 삽입하면서 시퀀스 길이가 바뀌므로, 위치 관련 입력은 전달하지 않고
    # 모델이 attention_mask 기준으로 position_ids를 다시 계산하게 두었다.
    skip_forward_keys = {
        "input_ids",
        "attention_mask",
        "labels",
        "output_hidden_states",
        "position_ids",
        "cache_position",
        "rope_deltas",
    }
    for key, value in row_model_inputs.items():
        if key in skip_forward_keys:
            continue
        if torch.is_tensor(value):
            extra_forward_inputs[key] = value

    # [변경 로직] input_ids=None + inputs_embeds 경로에서도 영상 mRoPE를 유지하기 위해
    # 원본 input_ids 기준 position_ids를 먼저 구해두고, latent 삽입 길이만큼 재배치해서 전달한다.
    base_position_ids = None
    if hasattr(vlm_core, "get_rope_index"):
        base_position_ids, _ = vlm_core.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=extra_forward_inputs.get("image_grid_thw"),
            video_grid_thw=extra_forward_inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
        )

    z_list = [z_seed.squeeze(0)]
    # depth=2면 [u1, u2], depth=3이면 [u1, u2, u3] 형태로 누적한다.
    for _ in range(1, depth):
        z_context = torch.stack(z_list, dim=0)
        z_context_for_llm = z_context
        if latent_input_projector is not None:
            try:
                latent_proj_dtype = next(latent_input_projector.parameters()).dtype
            except StopIteration:
                latent_proj_dtype = z_context.dtype
            z_context_for_llm = latent_input_projector(
                z_context.to(dtype=latent_proj_dtype)
            ).to(dtype=torch.float32)

        latent_tokens = z_context_for_llm.to(dtype=query_token_embeds.dtype).unsqueeze(0)
        inputs_embeds, full_attention = _inject_latent_tokens(
            query_token_embeds=query_token_embeds,
            latent_tokens=latent_tokens,
            attention_mask=attention_mask,
            insert_idx=vlm_insert_idx,
        )
        full_position_ids = None
        if base_position_ids is not None:
            latent_len = int(latent_tokens.size(1))
            head_pos = base_position_ids[:, :, :vlm_insert_idx]
            tail_pos = base_position_ids[:, :, vlm_insert_idx:]
            if vlm_insert_idx > 0:
                prev_pos = head_pos[:, :, -1:]
            else:
                prev_pos = base_position_ids[:, :, :1] - 1
            latent_offsets = torch.arange(
                1,
                latent_len + 1,
                device=base_position_ids.device,
                dtype=base_position_ids.dtype,
            ).view(1, 1, -1)
            latent_pos = prev_pos + latent_offsets
            full_position_ids = torch.cat(
                [head_pos, latent_pos, tail_pos + latent_len],
                dim=-1,
            )
        #breakpoint()
        with torch.no_grad():
            # [기존 로직 - 주석 보존] position_ids를 전달하지 않고 attention_mask만 전달했다.
            # outputs = vlm_core(
            #     input_ids=None,
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=full_attention,
            #     use_cache=False,
            #     **extra_forward_inputs,
            # )
            # [변경 로직] 재구성한 position_ids를 명시적으로 전달해
            # latent 삽입 후에도 멀티모달 위치 정렬(mRoPE)이 유지되게 한다.
            outputs = vlm_core(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention,
                position_ids=full_position_ids,
                use_cache=False,
                **extra_forward_inputs,
            )
            hidden = outputs.last_hidden_state.to(dtype=torch.float32)

        if str(qfinal_pooling).lower() == "mean":
            mask = full_attention.to(dtype=hidden.dtype).unsqueeze(-1)
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        else:
            extracted_idx = vlm_insert_idx + int(latent_tokens.size(1)) - 1
            pooled = hidden[:, extracted_idx, :]

        try:
            projector_dtype = next(refine_projector.parameters()).dtype
        except StopIteration:
            projector_dtype = pooled.dtype
        delta = refine_projector(pooled.to(dtype=projector_dtype)).to(dtype=torch.float32)

        if bool(use_refine_gate) and refine_gate is not None:
            try:
                gate_dtype = next(refine_gate.parameters()).dtype
            except StopIteration:
                gate_dtype = pooled.dtype
            alpha = torch.sigmoid(refine_gate(pooled.to(dtype=gate_dtype))).to(dtype=torch.float32)
            z_next = delta * alpha
        else:
            z_next = delta

        if z_next.dim() == 1:
            z_next = z_next.unsqueeze(0)
        if z_next.size(0) != 1:
            z_next = z_next.mean(dim=0, keepdim=True)
        z_list.append(z_next.squeeze(0))
    return torch.stack(z_list, dim=0)


def _summarize_rollout_latents(update: Optional[torch.Tensor], head_dim: int = 6) -> Dict[str, object]:
    out: Dict[str, object] = {
        "rollout_depth_actual": 0,
        "rollout_shape": [],
        "rollout_latent_norms": [],
        "rollout_latent_cos_prev": [],
        "rollout_latent_head": [],
    }
    if update is None:
        return out

    with torch.no_grad():
        lat = update.detach().to(dtype=torch.float32, device="cpu")
        if lat.dim() == 1:
            lat = lat.unsqueeze(0)
        if lat.dim() != 2:
            out["rollout_shape"] = [int(x) for x in lat.shape]
            return out

        l_count = int(lat.size(0))
        l_dim = int(lat.size(1))
        out["rollout_depth_actual"] = l_count
        out["rollout_shape"] = [l_count, l_dim]
        out["rollout_latent_norms"] = [float(x) for x in torch.norm(lat, dim=-1).tolist()]
        if l_count > 1:
            out["rollout_latent_cos_prev"] = [
                float(x) for x in F.cosine_similarity(lat[:-1], lat[1:], dim=-1).tolist()
            ]
        head_k = int(max(0, min(int(head_dim), l_dim)))
        if head_k > 0:
            out["rollout_latent_head"] = [
                [float(v) for v in row] for row in lat[:, :head_k].tolist()
            ]
    return out


def _build_q_final(
    *,
    q_orig: torch.Tensor,
    update: torch.Tensor,
    query_text: str,
    use_query_embedder_path: bool,
    query_embedder_model,
    query_embedder_tokenizer,
    query_embedder_head,
    append_input_projector,
    latent_input_projector,
    qfinal_pooling: str,
    qfinal_normalize: bool,
    query_embedder_max_length: int,
    query_embedder_input_prefix: str,
):
    debug_info: Dict[str, object] = {
        "forward_mode": "none",
        "input_ids_is_none": None,
        "inputs_embeds_shape": [],
        "full_attention_shape": [],
    }
    if update.dim() == 1:
        update = update.unsqueeze(0)
    if update.dim() != 2:
        raise ValueError(f"update must be 1D/2D tensor, got shape={tuple(update.shape)}")

    if bool(use_query_embedder_path):
        if query_embedder_model is None or query_embedder_tokenizer is None:
            raise RuntimeError("use_query_embedder_path=True but query embedder is not initialized.")
        q_embedder = query_embedder_model
        q_embedder.eval()
        embedder_query_text = _build_query_embedder_text(
            query_embedder_tokenizer,
            query_text,
            query_embedder_input_prefix,
        )
        tokenized = query_embedder_tokenizer(
            [embedder_query_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(query_embedder_max_length),
        )
        device = update.device
        input_ids = tokenized["input_ids"].to(device=device)
        attention_mask = tokenized["attention_mask"].to(device=device)

        query_token_embeds = q_embedder.get_input_embeddings()(input_ids)
        update_for_llm = update
        # q_final append projector를 우선 사용하고, 구버전 체크포인트는 rollout projector로 폴백.
        append_in_projector = append_input_projector
        if append_in_projector is None:
            append_in_projector = latent_input_projector
        if append_in_projector is not None:
            try:
                latent_proj_dtype = next(append_in_projector.parameters()).dtype
            except StopIteration:
                latent_proj_dtype = update.dtype
            update_for_llm = append_in_projector(
                update.to(dtype=latent_proj_dtype)
            ).to(dtype=torch.float32)
        latent_tokens = update_for_llm.to(dtype=query_token_embeds.dtype).unsqueeze(0)
        inputs_embeds, full_attention = _inject_latent_tokens(
            query_token_embeds=query_token_embeds,
            latent_tokens=latent_tokens,
            attention_mask=attention_mask,
        )

        with torch.no_grad():
            outputs = q_embedder.language_model(
                input_ids=None,
                attention_mask=full_attention,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state.to(dtype=torch.float32)
        debug_info["forward_mode"] = "inputs_embeds"
        debug_info["input_ids_is_none"] = True
        debug_info["inputs_embeds_shape"] = [int(x) for x in inputs_embeds.shape]
        debug_info["full_attention_shape"] = [int(x) for x in full_attention.shape]

        if str(qfinal_pooling).lower() == "mean":
            mask = full_attention.to(dtype=hidden.dtype).unsqueeze(-1)
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        else:
            pooled = hidden[:, -1, :]

        if query_embedder_head is not None:
            try:
                q_head_dtype = next(query_embedder_head.parameters()).dtype
            except StopIteration:
                q_head_dtype = pooled.dtype
            pooled = query_embedder_head(pooled.to(dtype=q_head_dtype))

        q_final = pooled.squeeze(0).to(dtype=torch.float32)
        if q_final.size(-1) != q_orig.size(-1):
            raise ValueError(
                "q_final dim mismatch in query_embedder path: "
                f"{q_final.size(-1)} vs expected {q_orig.size(-1)}"
            )
        mode = "query_embedder"
    else:
        q_final = q_orig + update.mean(dim=0)
        mode = "residual_add"
        debug_info["forward_mode"] = "residual_add"
        debug_info["input_ids_is_none"] = None

    if bool(qfinal_normalize):
        q_final = F.normalize(q_final, p=2, dim=-1)
    return q_final, mode, debug_info


def _default_output_path(model_path: str) -> str:
    ckpt_name = os.path.basename(os.path.normpath(model_path))
    parent = os.path.dirname(os.path.normpath(model_path))
    logs_dir = os.path.join(parent, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"verified_test_temporal_grounding_{ckpt_name}.json")


def _resolve_processor_path(model_path: str, processor_path_arg: str) -> str:
    if processor_path_arg and str(processor_path_arg).strip():
        return str(processor_path_arg).strip()
    cand_local = os.path.join(model_path, "preprocessor_config.json")
    if os.path.exists(cand_local):
        return model_path
    parent = os.path.dirname(os.path.normpath(model_path))
    cand_parent = os.path.join(parent, "preprocessor_config.json")
    if os.path.exists(cand_parent):
        return parent
    return "Qwen/Qwen3-VL-2B-Instruct"


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def _safe_int(v, default: int = 0) -> int:
    try:
        x = int(v)
    except Exception:
        return int(default)
    return int(x)


def _to_numpy_fp16_1d(vec: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if vec is None or not torch.is_tensor(vec):
        return None
    x = vec.detach()
    if x.dim() == 2 and int(x.size(0)) == 1:
        x = x.squeeze(0)
    if x.dim() != 1:
        x = x.reshape(-1)
    return x.to(device="cpu", dtype=torch.float32).numpy().astype(np.float16, copy=False)


def _save_query_vector_artifact(
    *,
    save_dir: str,
    row_index: int,
    qid: str,
    gt_pos_doc_id: str,
    query_text: str,
    q_orig_initial: Optional[torch.Tensor],
    gt_video_vec: Optional[torch.Tensor],
    q_policy_turn_vectors: Optional[Dict[int, torch.Tensor]],
    q_profile_turn_vectors: Optional[Dict[int, Dict[int, torch.Tensor]]],
    turn_details: List[dict],
) -> int:
    q_orig_np = _to_numpy_fp16_1d(q_orig_initial)
    if q_orig_np is None or q_orig_np.size == 0:
        return 0

    os.makedirs(save_dir, exist_ok=True)
    stem = f"idx_{int(row_index):06d}"
    npz_path = os.path.join(save_dir, f"{stem}.npz")
    meta_path = os.path.join(save_dir, f"{stem}.json")

    turn_ids: List[int] = []
    q_policy_rows: List[np.ndarray] = []
    q_policy_turn_vectors = q_policy_turn_vectors or {}
    for t in sorted(q_policy_turn_vectors.keys()):
        arr = _to_numpy_fp16_1d(q_policy_turn_vectors.get(t))
        if arr is None or arr.size != q_orig_np.size:
            continue
        turn_ids.append(int(t))
        q_policy_rows.append(arr)

    if q_policy_rows:
        q_policy_np = np.stack(q_policy_rows, axis=0).astype(np.float16, copy=False)
    else:
        q_policy_np = np.zeros((0, int(q_orig_np.size)), dtype=np.float16)

    q_gt_np = _to_numpy_fp16_1d(gt_video_vec)
    if q_gt_np is None or q_gt_np.size != q_orig_np.size:
        q_gt_np = np.zeros((0,), dtype=np.float16)

    # Optional profile vectors for turn=1: depth-wise q vectors (e.g., d1,d2,d4,d8,d16)
    profile_depth_ids: List[int] = []
    profile_rows: List[np.ndarray] = []
    q_profile_turn_vectors = q_profile_turn_vectors or {}
    t1_profile = q_profile_turn_vectors.get(1, {})
    if isinstance(t1_profile, dict):
        for d in sorted(t1_profile.keys()):
            arr = _to_numpy_fp16_1d(t1_profile.get(d))
            if arr is None or arr.size != q_orig_np.size:
                continue
            profile_depth_ids.append(int(d))
            profile_rows.append(arr)
    if profile_rows:
        q_profile_np = np.stack(profile_rows, axis=0).astype(np.float16, copy=False)
    else:
        q_profile_np = np.zeros((0, int(q_orig_np.size)), dtype=np.float16)

    np.savez_compressed(
        npz_path,
        q_orig=q_orig_np,
        q_gt_video=q_gt_np,
        turn_ids=np.asarray(turn_ids, dtype=np.int16),
        q_policy=q_policy_np,
        profile_depth_ids_turn1=np.asarray(profile_depth_ids, dtype=np.int16),
        q_profile_turn1=q_profile_np,
    )

    meta_obj = {
        "index": int(row_index),
        "qid": str(qid),
        "gt_pos_doc_id": str(gt_pos_doc_id),
        "query": str(query_text),
        "saved_turn_ids": [int(x) for x in turn_ids],
        "saved_profile_depth_ids_turn1": [int(x) for x in profile_depth_ids],
        "turns": [
            {
                "turn": int(_safe_int(t.get("turn"), 0)),
                "rank_before_refine": int(_safe_int(t.get("rank_before_refine"), 0)),
                "policy_rank": int(_safe_int(t.get("policy_rank"), 0)),
                "policy_top1_doc_id": str(t.get("policy_top1_doc_id", "")),
                "policy_top1_is_gt": bool(t.get("policy_top1_is_gt", False)),
                "refine_applied": bool(t.get("refine_applied", False)),
            }
            for t in (turn_details or [])
            if isinstance(t, dict)
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)
    saved_bytes = int(os.path.getsize(npz_path)) + int(os.path.getsize(meta_path))
    return int(saved_bytes)


def _parse_profile_depths(raw: str, max_depth: int) -> List[int]:
    vals: List[int] = []
    for s in str(raw or "").split(","):
        s = s.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception as exc:
            raise ValueError(f"Invalid profile depth: {s}") from exc
        if v < 1:
            raise ValueError(f"profile depth must be >=1, got {v}")
        vals.append(int(v))
    vals = sorted(set(vals))
    if max_depth > 0:
        vals = [int(v) for v in vals if int(v) <= int(max_depth)]
    return vals


def _parse_iou_thresholds(raw: str) -> List[float]:
    vals = []
    for s in str(raw).split(","):
        s = s.strip()
        if not s:
            continue
        try:
            v = float(s)
        except Exception as exc:
            raise ValueError(f"Invalid IoU threshold: {s}") from exc
        if v < 0.0 or v > 1.0:
            raise ValueError(f"IoU threshold must be in [0,1], got {v}")
        vals.append(v)
    if not vals:
        vals = [0.3, 0.5, 0.7]
    vals = sorted(set(float(v) for v in vals))
    return vals


def _temporal_metrics(iou_r1: List[float], iou_matched_only: List[float], thresholds: List[float]) -> Dict[str, float]:
    arr = np.asarray(iou_r1, dtype=np.float32)
    out: Dict[str, float] = {
        "count": float(arr.size),
        "mIoU@R1": float(np.mean(arr)) if arr.size else 0.0,
        "matched_with_time_count": float(len(iou_matched_only)),
        "mIoU_matched_only": float(np.mean(iou_matched_only)) if iou_matched_only else 0.0,
    }
    for th in thresholds:
        key = f"IoU@{th:.1f}@R1"
        out[key] = float(np.mean(arr >= th)) if arr.size else 0.0
    return out


def _extract_policy_turns_from_detail_row(row: dict, max_turn: int) -> List[dict]:
    turns = row.get("turns", [])
    if not isinstance(turns, list) or not turns:
        return []

    parsed: List[dict] = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue

        rank_before = t.get("rank_before_refine", t.get("rank", None))
        policy_rank = t.get("policy_rank", None)

        if not isinstance(policy_rank, int) or int(policy_rank) <= 0:
            if bool(t.get("refine_applied", False)):
                # Backward-compatible reconstruction for old rows that only logged pre-refine rank.
                if i + 1 < len(turns):
                    n = turns[i + 1]
                    if isinstance(n, dict) and isinstance(n.get("rank"), int) and int(n.get("rank")) > 0:
                        policy_rank = int(n["rank"])
                if (not isinstance(policy_rank, int) or int(policy_rank) <= 0) and isinstance(
                    row.get("final_rank"), int
                ):
                    policy_rank = int(row["final_rank"])
            if (not isinstance(policy_rank, int) or int(policy_rank) <= 0) and isinstance(rank_before, int):
                policy_rank = int(rank_before)

        if not isinstance(policy_rank, int) or int(policy_rank) <= 0:
            continue

        parsed.append(
            {
                "turn": int(_safe_int(t.get("turn", i + 1), default=i + 1)),
                "rank_before_refine": int(_safe_int(rank_before, default=policy_rank)),
                "policy_rank": int(policy_rank),
                "answer": str(t.get("answer", "unknown")).strip().lower() or "unknown",
                "time_parse_ok": bool(t.get("time_parse_ok", False)),
                "iou_raw": float(_safe_float(t.get("iou_raw"), 0.0)),
                "iou_r1": float(_safe_float(t.get("iou_r1"), 0.0)),
            }
        )
        if len(parsed) >= int(max_turn):
            break
    return parsed


def _accumulate_turn_metrics_from_policy_turns(
    *,
    policy_turns: List[dict],
    max_turn: int,
    policy_ranks_by_turn: Dict[int, List[int]],
    strict_ranks_by_turn: Dict[int, List[int]],
    policy_iou_r1_by_turn: Dict[int, List[float]],
    strict_iou_r1_by_turn: Dict[int, List[float]],
    policy_iou_matched_by_turn: Dict[int, List[float]],
    strict_iou_matched_by_turn: Dict[int, List[float]],
) -> None:
    if not policy_turns:
        return

    capped = policy_turns[: int(max_turn)]
    n_exec = len(capped)
    if n_exec == 0:
        return

    last = capped[-1]
    for turn in range(1, int(max_turn) + 1):
        td = capped[turn - 1] if turn <= n_exec else last
        rank = int(_safe_int(td.get("policy_rank"), default=0))
        if rank > 0:
            policy_ranks_by_turn[int(turn)].append(rank)
        iou_r1 = float(_safe_float(td.get("iou_r1"), 0.0))
        policy_iou_r1_by_turn[int(turn)].append(iou_r1)
        if str(td.get("answer", "unknown")) == "matched" and bool(td.get("time_parse_ok", False)):
            policy_iou_matched_by_turn[int(turn)].append(float(_safe_float(td.get("iou_raw"), 0.0)))

    for turn in range(1, n_exec + 1):
        td = capped[turn - 1]
        rank = int(_safe_int(td.get("policy_rank"), default=0))
        if rank > 0:
            strict_ranks_by_turn[int(turn)].append(rank)
        strict_iou_r1_by_turn[int(turn)].append(float(_safe_float(td.get("iou_r1"), 0.0)))
        if str(td.get("answer", "unknown")) == "matched" and bool(td.get("time_parse_ok", False)):
            strict_iou_matched_by_turn[int(turn)].append(float(_safe_float(td.get("iou_raw"), 0.0)))


def _accumulate_summary_from_detail(
    *,
    row: dict,
    orig_ranks: List[int],
    final_ranks: List[int],
    turn1_iou_r1: List[float],
    final_iou_r1: List[float],
    turn1_iou_matched_only: List[float],
    final_iou_matched_only: List[float],
    answer_counts: Counter,
    stop_reason_counts: Counter,
    matched_by_turn: Counter,
    refine_counters: Counter,
    max_turn: int,
    policy_ranks_by_turn: Dict[int, List[int]],
    strict_ranks_by_turn: Dict[int, List[int]],
    policy_iou_r1_by_turn: Dict[int, List[float]],
    strict_iou_r1_by_turn: Dict[int, List[float]],
    policy_iou_matched_by_turn: Dict[int, List[float]],
    strict_iou_matched_by_turn: Dict[int, List[float]],
):
    if isinstance(row.get("orig_rank"), int):
        orig_ranks.append(int(row["orig_rank"]))
    if isinstance(row.get("final_rank"), int):
        final_ranks.append(int(row["final_rank"]))

    policy_turns = _extract_policy_turns_from_detail_row(row=row, max_turn=int(max_turn))
    _accumulate_turn_metrics_from_policy_turns(
        policy_turns=policy_turns,
        max_turn=int(max_turn),
        policy_ranks_by_turn=policy_ranks_by_turn,
        strict_ranks_by_turn=strict_ranks_by_turn,
        policy_iou_r1_by_turn=policy_iou_r1_by_turn,
        strict_iou_r1_by_turn=strict_iou_r1_by_turn,
        policy_iou_matched_by_turn=policy_iou_matched_by_turn,
        strict_iou_matched_by_turn=strict_iou_matched_by_turn,
    )

    if policy_turns:
        first = policy_turns[0]
        last = policy_turns[-1]
        turn1_iou_r1.append(float(_safe_float(first.get("iou_r1"), 0.0)))
        final_iou_r1.append(float(_safe_float(last.get("iou_r1"), 0.0)))
        if str(first.get("answer", "unknown")) == "matched" and bool(first.get("time_parse_ok", False)):
            turn1_iou_matched_only.append(float(_safe_float(first.get("iou_raw"), 0.0)))
        if str(last.get("answer", "unknown")) == "matched" and bool(last.get("time_parse_ok", False)):
            final_iou_matched_only.append(float(_safe_float(last.get("iou_raw"), 0.0)))
    else:
        turn1_iou_r1.append(_safe_float(row.get("turn1_iou_r1"), 0.0))
        final_iou_r1.append(_safe_float(row.get("final_iou_r1"), 0.0))
        if str(row.get("turn1_answer", "")).strip().lower() == "matched" and bool(row.get("turn1_time_parse_ok", False)):
            turn1_iou_matched_only.append(_safe_float(row.get("turn1_iou_raw"), 0.0))
        if str(row.get("final_answer", "")).strip().lower() == "matched" and bool(row.get("final_time_parse_ok", False)):
            final_iou_matched_only.append(_safe_float(row.get("final_iou_raw"), 0.0))

    answer = str(row.get("final_answer", "unknown")).strip().lower() or "unknown"
    answer_counts[answer] += 1
    stop_reason = str(row.get("stop_reason", "unknown")).strip() or "unknown"
    stop_reason_counts[stop_reason] += 1

    try:
        matched_turn = int(row.get("matched_turn", 0))
    except Exception:
        matched_turn = 0
    if matched_turn > 0:
        matched_by_turn[str(matched_turn)] += 1

    if bool(row.get("used_refine_any", False)):
        refine_counters["samples_with_refine"] += 1
    refine_counters["refine_token_generated_turns"] += int(max(0, row.get("refine_token_generated_turns", 0) or 0))
    refine_counters["refine_applied_turns"] += int(max(0, row.get("refine_applied_turns", 0) or 0))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Temporal-grounding eval on verified_test. "
            "Iteratively runs VLM+retrieval up to max_turn: stop on <answer>matched</answer>, "
            "otherwise apply refine token latent update and retrieve next top1."
        )
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--processor_path",
        type=str,
        default="",
        help="Optional processor/tokenizer path. If empty, auto-resolve from model_path then parent.",
    )
    p.add_argument(
        "--verified_test_jsonl",
        type=str,
        default="./data/ActivityNet-test/verified_test.jsonl",
    )
    p.add_argument(
        "--video_root",
        type=str,
        default=_default_activitynet_path("test", "video_npy_with_meta"),
        help=(
            "Root for video files. Supports npy/npz and raw video "
            "(mp4/mkv/webm/avi/mov/m4v)."
        ),
    )
    p.add_argument(
        "--video_meta_path",
        type=str,
        default="",
        help="Optional video meta jsonl path. If empty, auto-detect from video_root.",
    )
    p.add_argument(
        "--query_embeddings_path",
        type=str,
        default=_default_activitynet_path("test", "query_embedding", "query_embeddings.test.npy"),
    )
    p.add_argument(
        "--query_meta_path",
        type=str,
        default=_default_activitynet_path("test", "query_embedding", "query_meta.test.jsonl"),
    )
    p.add_argument(
        "--video_embeddings_path",
        type=str,
        default=_default_activitynet_path("test", "video_embedding_1fps", "segment_embeds.npy"),
    )
    p.add_argument(
        "--video_docid2row_path",
        type=str,
        default=_default_activitynet_path("test", "video_embedding_1fps", "docid2row.json"),
    )
    p.add_argument("--temporal", type=_str2bool, default=True)
    p.add_argument("--max_turn", type=int, default=2)
    p.add_argument("--iou_thresholds", type=str, default="0.3,0.5,0.7")
    p.add_argument("--refine_token", type=str, default="<REFINE>")
    p.add_argument(
        "--refine_token_count",
        type=int,
        default=1,
        help=(
            "Refine latent rollout depth. Special-token registration is always single-token "
            "(--refine_token only)."
        ),
    )
    p.add_argument("--use_refine_gate", type=_str2bool, default=True)
    p.add_argument("--use_query_embedder_path", type=_str2bool, default=True)
    p.add_argument("--query_embedder_model_path", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    p.add_argument(
        "--use_updated_query",
        type=_str2bool,
        default=False,
        help=(
            "If True, ignore precomputed initial query vectors and re-encode each test query text with "
            "query_embedder_model_path. Refine-turn logic remains unchanged."
        ),
    )
    p.add_argument("--qfinal_pooling", type=str, default="latent_last", choices=["latent_last", "mean"])
    p.add_argument("--qfinal_normalize", type=_str2bool, default=True)
    p.add_argument("--query_embedder_max_length", type=int, default=128)
    p.add_argument(
        "--query_embedder_input_prefix",
        type=str,
        default=os.environ.get("QUERY_EMBEDDER_INPUT_PREFIX", ""),
        help="Optional prefix prepended before query text for query embedder tokenization.",
    )
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--use_vllm", type=_str2bool, default=False)
    p.add_argument(
        "--vllm_model_path",
        type=str,
        default="",
        help=(
            "Optional model path used only by vLLM generation backend. "
            "If empty, a vLLM-compatible copy is auto-prepared from --model_path."
        ),
    )
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--vllm_max_num_seqs", type=int, default=1)
    p.add_argument(
        "--vllm_eval_batch_size",
        type=int,
        default=0,
        help=(
            "Batch size for vLLM eval requests. "
            "0 means auto (=vllm_max_num_seqs). "
            "Only used when --use_vllm True."
        ),
    )
    p.add_argument("--vllm_disable_custom_all_reduce", type=_str2bool, default=False)
    p.add_argument("--vllm_enforce_eager", type=_str2bool, default=False)
    p.add_argument("--topk", type=str, default="1,5,10,100")
    p.add_argument(
        "--rerank_topk",
        type=str,
        default="all",
        help="Temporal eval currently supports only 'all' for stable multi-turn behavior.",
    )
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--eval_limit_ratio", type=float, default=1.0)
    p.add_argument("--num_shards", type=int, default=1, help="Total shard count for data-parallel eval.")
    p.add_argument("--shard_id", type=int, default=0, help="Shard id in [0, num_shards).")
    p.add_argument(
        "--resume_skip_jsonl",
        type=str,
        default="",
        help=(
            "Comma-separated jsonl paths. Rows whose 'index' appears here are dropped "
            "before shard split (useful for multi-process resume)."
        ),
    )
    p.add_argument(
        "--progress_interval_sec",
        type=float,
        default=2.0,
        help="Progress log interval in seconds (real-time ETA/remaining updates).",
    )
    p.add_argument("--use_tqdm", type=_str2bool, default=True)
    p.add_argument(
        "--tqdm_mininterval",
        type=float,
        default=0.2,
        help="Min refresh interval for tqdm progress bar.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", type=_str2bool, default=True)
    p.add_argument("--image_patch_size", type=int, default=16)
    p.add_argument("--video_min_pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--video_max_pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--video_total_pixels", type=int, default=115200 * 28 * 28)
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--output_jsonl", type=str, default="")
    p.add_argument(
        "--save_query_vectors",
        type=_str2bool,
        default=False,
        help=(
            "If True, dump per-sample query vectors (q_orig + policy query per turn) "
            "and GT video vector for later visualization."
        ),
    )
    p.add_argument(
        "--save_query_vectors_dir",
        type=str,
        default="",
        help=(
            "Directory for query vector artifacts. "
            "If empty and --save_query_vectors=True, defaults to <output_dir>/query_vectors."
        ),
    )
    p.add_argument(
        "--save_query_vectors_max_gb",
        type=float,
        default=0.0,
        help="If >0, stop dumping query vectors after reaching this approximate directory budget (GB).",
    )
    p.add_argument(
        "--save_query_vectors_only_refined",
        type=_str2bool,
        default=False,
        help="If True, save query vectors only for samples where refine was actually applied.",
    )
    p.add_argument(
        "--profile_rollout_depths",
        type=str,
        default="",
        help=(
            "Optional CSV depths (e.g., 1,2,4,8,16). "
            "When refine is applied, records per-depth policy rank/top1 from one pass."
        ),
    )
    p.add_argument(
        "--resume_from_jsonl",
        type=_str2bool,
        default=False,
        help="If True and output_jsonl exists, skip already written indices and append new rows.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not bool(args.temporal):
        raise ValueError("This script is temporal-only. Set --temporal True.")
    if int(args.max_turn) < 1:
        raise ValueError("--max_turn must be >= 1")
    if int(args.vllm_tensor_parallel_size) < 1:
        raise ValueError("--vllm_tensor_parallel_size must be >= 1")
    if int(args.vllm_max_num_seqs) < 1:
        raise ValueError("--vllm_max_num_seqs must be >= 1")
    if int(args.vllm_eval_batch_size) < 0:
        raise ValueError("--vllm_eval_batch_size must be >= 0")
    if float(args.vllm_gpu_memory_utilization) <= 0.0 or float(args.vllm_gpu_memory_utilization) > 1.0:
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1].")
    if int(args.num_shards) < 1:
        raise ValueError("--num_shards must be >= 1")
    if int(args.shard_id) < 0 or int(args.shard_id) >= int(args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    if float(args.progress_interval_sec) <= 0.0:
        raise ValueError("--progress_interval_sec must be > 0.")
    if float(args.tqdm_mininterval) < 0.0:
        raise ValueError("--tqdm_mininterval must be >= 0.")

    iou_thresholds = _parse_iou_thresholds(args.iou_thresholds)
    ks = sorted({max(1, int(x.strip())) for x in str(args.topk).split(",") if x.strip()})
    output_json = args.output_json.strip() or _default_output_path(args.model_path)
    output_dir_for_default = os.path.dirname(output_json) or os.getcwd()
    save_query_vectors_enabled = bool(args.save_query_vectors)
    save_query_vectors_only_refined = bool(args.save_query_vectors_only_refined)
    save_query_vectors_max_bytes = int(max(0.0, float(args.save_query_vectors_max_gb)) * (1024**3))
    save_query_vectors_dir = str(args.save_query_vectors_dir or "").strip()
    if save_query_vectors_enabled and not save_query_vectors_dir:
        save_query_vectors_dir = os.path.join(output_dir_for_default, "query_vectors")
    if save_query_vectors_enabled:
        os.makedirs(save_query_vectors_dir, exist_ok=True)
        logger.info(
            "Query vector dump enabled: dir=%s only_refined=%s max_gb=%s",
            save_query_vectors_dir,
            bool(save_query_vectors_only_refined),
            float(args.save_query_vectors_max_gb),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_rows = list(_iter_jsonl(args.verified_test_jsonl))
    full_count = len(test_rows)
    if float(args.eval_limit_ratio) < 1.0:
        ratio = float(max(0.0, min(1.0, args.eval_limit_ratio)))
        keep_n = max(1, int(round(len(test_rows) * ratio)))
        rng = random.Random(int(args.seed))
        sel = sorted(rng.sample(range(len(test_rows)), keep_n))
        test_rows = [test_rows[i] for i in sel]
        logger.info(f"Applied eval_limit_ratio={ratio:.3f}: {full_count} -> {len(test_rows)}")
    if int(args.max_samples) > 0 and int(args.max_samples) < len(test_rows):
        test_rows = test_rows[: int(args.max_samples)]
        logger.info(f"Applied max_samples={int(args.max_samples)}: now {len(test_rows)} rows")

    test_row_indices = list(range(len(test_rows)))
    resume_skip_paths = _split_csv_paths(args.resume_skip_jsonl)
    if resume_skip_paths:
        pre_done_indices, files_found, rows_loaded, bad_lines = _load_done_indices_from_jsonl_paths(resume_skip_paths)
        logger.info(
            "Loaded resume-skip indices: paths=%d files_found=%d rows=%d bad_lines=%d unique_indices=%d",
            len(resume_skip_paths),
            int(files_found),
            int(rows_loaded),
            int(bad_lines),
            int(len(pre_done_indices)),
        )
        if pre_done_indices:
            before_n = len(test_rows)
            kept_pairs = [(ri, row) for ri, row in zip(test_row_indices, test_rows) if int(ri) not in pre_done_indices]
            test_row_indices = [int(ri) for ri, _ in kept_pairs]
            test_rows = [row for _, row in kept_pairs]
            logger.info("Applied resume pre-skip before sharding: %d -> %d rows", int(before_n), int(len(test_rows)))

    if int(args.num_shards) > 1:
        before_n = len(test_rows)
        kept_pairs = [
            (ri, row)
            for local_pos, (ri, row) in enumerate(zip(test_row_indices, test_rows))
            if int(local_pos) % int(args.num_shards) == int(args.shard_id)
        ]
        test_row_indices = [int(ri) for ri, _ in kept_pairs]
        test_rows = [row for _, row in kept_pairs]
        logger.info(
            "Applied shard split: shard=%d/%d, %d -> %d rows",
            int(args.shard_id),
            int(args.num_shards),
            int(before_n),
            int(len(test_rows)),
        )

    if not test_rows:
        logger.warning("No test rows assigned after filtering/resume-skip/sharding.")
    logger.info(f"Loaded verified test rows: {len(test_rows)}")

    query_embeddings = np.asarray(np.load(args.query_embeddings_path), dtype=np.float32)
    query_embeddings = _l2_norm_rows(query_embeddings)
    query_map = _load_query_meta(args.query_meta_path)
    video_embeddings = np.asarray(np.load(args.video_embeddings_path), dtype=np.float32)
    video_embeddings = _l2_norm_rows(video_embeddings)
    retrieval_dim = int(video_embeddings.shape[1])

    with open(args.video_docid2row_path, "r", encoding="utf-8") as f:
        docid2row = {str(k): int(v) for k, v in json.load(f).items()}
    row2doc = _invert_docid2row(docid2row)
    video_emb_t = torch.from_numpy(video_embeddings).to(device=device, dtype=torch.float32)

    rerank_topk = _parse_rerank_topk(args.rerank_topk, max_docs=video_embeddings.shape[0])
    if rerank_topk is not None:
        raise ValueError(
            "Temporal multi-turn eval currently supports only --rerank_topk all. "
            "Set RERANKS=all (default)."
        )

    logger.info(
        f"Loaded embeddings: query={query_embeddings.shape}, video={video_embeddings.shape}, docids={len(docid2row)}"
    )

    resolved_video_meta_path = _resolve_video_meta_path(
        args.video_meta_path, args.video_root
    )
    video_meta_index = (
        load_video_meta_index(resolved_video_meta_path)
        if resolved_video_meta_path
        else {}
    )
    logger.info(
        "Video meta path: %s",
        resolved_video_meta_path if resolved_video_meta_path else "(disabled)",
    )

    processor_path = _resolve_processor_path(args.model_path, args.processor_path)
    logger.info(f"Using processor path: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, padding_side="left")

    refine_token = _resolve_refine_token(args.refine_token)
    refine_rollout_depth = _resolve_rollout_depth(int(args.refine_token_count))
    profile_rollout_depths = _parse_profile_depths(
        str(args.profile_rollout_depths or ""),
        max_depth=int(refine_rollout_depth),
    )
    refine_tokens = [refine_token]
    missing_tokens = [tok for tok in refine_tokens if tok not in processor.tokenizer.get_vocab()]
    if missing_tokens:
        processor.tokenizer.add_tokens(missing_tokens, special_tokens=True)
        logger.info(f"Added refine tokens: {missing_tokens}")
    refine_token_ids = [processor.tokenizer.convert_tokens_to_ids(tok) for tok in refine_tokens]
    logger.info(
        "Refine token config: "
        f"token={refine_token}, token_id={refine_token_ids[0]}, "
        f"rollout_depth={refine_rollout_depth}"
    )
    if profile_rollout_depths:
        logger.info("Profile rollout depths: %s", profile_rollout_depths)

    system_prompt = _make_system_prompt(refine_token)

    model_kwargs = {}
    if device.type == "cuda" and bool(args.bf16):
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    except Exception as exc:
        if "attn_implementation" in model_kwargs:
            logger.warning(f"flash_attention_2 load failed: {exc}; retry without it.")
            model_kwargs.pop("attn_implementation", None)
            model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
        else:
            raise

    model.resize_token_embeddings(len(processor.tokenizer))
    hidden_size, hidden_size_src = _resolve_text_hidden_size(model, default=retrieval_dim)
    if not hasattr(model, "refine_projector"):
        model.refine_projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, retrieval_dim),
            torch.nn.LayerNorm(retrieval_dim),
            torch.nn.GELU(),
            torch.nn.Linear(retrieval_dim, retrieval_dim),
        )
    if not hasattr(model, "refine_gate"):
        model.refine_gate = torch.nn.Linear(hidden_size, 1)
    created_roll_in = False
    if not hasattr(model, "refine_latent_input_projector"):
        model.refine_latent_input_projector = _make_latent_input_projector(retrieval_dim, hidden_size)
        created_roll_in = True
    if created_roll_in:
        _init_partial_identity_linear(model.refine_latent_input_projector[1])
    logger.info(
        f"Refine module init: hidden_size={hidden_size} (source={hidden_size_src}), retrieval_dim={retrieval_dim}"
    )

    query_embedder_model = None
    query_embedder_tokenizer = None
    need_query_embedder = bool(args.use_query_embedder_path) or bool(args.use_updated_query)
    query_hidden = retrieval_dim
    if need_query_embedder:
        logger.info(
            "Loading query embedder: %s (use_query_embedder_path=%s, use_updated_query=%s)",
            args.query_embedder_model_path,
            bool(args.use_query_embedder_path),
            bool(args.use_updated_query),
        )
        q_proc = AutoProcessor.from_pretrained(args.query_embedder_model_path, padding_side="left")
        query_embedder_tokenizer = q_proc.tokenizer
        q_kwargs = dict(model_kwargs)
        try:
            query_embedder_model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.query_embedder_model_path, **q_kwargs
            )
        except Exception as exc:
            if "attn_implementation" in q_kwargs:
                logger.warning(f"Query embedder flash_attention_2 load failed: {exc}; retry without it.")
                q_kwargs.pop("attn_implementation", None)
                query_embedder_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    args.query_embedder_model_path, **q_kwargs
                )
            else:
                raise

        query_hidden, query_hidden_src = _resolve_text_hidden_size(query_embedder_model, default=retrieval_dim)
        if not hasattr(model, "query_embedder_head"):
            if query_hidden != retrieval_dim:
                model_dtype = next(model.parameters()).dtype
                model.query_embedder_head = torch.nn.Linear(query_hidden, retrieval_dim).to(
                    device=device, dtype=model_dtype
                )
                logger.info(
                    "Initialized query_embedder_head from query embedder config "
                    f"({query_hidden} -> {retrieval_dim}, source={query_hidden_src})"
                )
            else:
                model.query_embedder_head = torch.nn.Identity()
                logger.info(
                    "query_embedder_head is Identity "
                    f"(query_hidden={query_hidden}, source={query_hidden_src})"
                )
        query_embedder_model.to(device)
        query_embedder_model.eval()

    created_append_in = False
    if not hasattr(model, "refine_append_input_projector"):
        model.refine_append_input_projector = _make_latent_input_projector(retrieval_dim, query_hidden)
        created_append_in = True
    if created_append_in:
        _init_partial_identity_linear(model.refine_append_input_projector[1])

    loaded_refine = _maybe_load_refine_weights(model, args.model_path)
    if not loaded_refine and hasattr(model, "refine_latent_input_projector") and hidden_size == retrieval_dim:
        # Backward-compat: old checkpoints without this module can keep legacy no-op behavior.
        model.refine_latent_input_projector = None
    if not loaded_refine and hasattr(model, "refine_append_input_projector") and query_hidden == retrieval_dim:
        model.refine_append_input_projector = None
    model.to(device)
    model.eval()

    use_updated_query = bool(args.use_updated_query)
    if use_updated_query:
        if query_embedder_model is None or query_embedder_tokenizer is None:
            raise RuntimeError(
                "--use_updated_query=True requires a loadable query embedder "
                "(set --query_embedder_model_path or keep checkpoint/query_embedder)."
            )
        logger.info(
            "use_updated_query=True: initial q_orig vectors will be re-encoded from query text."
        )
    updated_query_cache: Dict[int, torch.Tensor] = {}

    use_vllm = bool(args.use_vllm)
    vllm_engine = None
    vllm_sampling_params = None
    vllm_runtime_model_path = ""
    vllm_eval_batch_size = 1
    if use_vllm:
        if not _VLLM_AVAILABLE:
            raise RuntimeError("use_vllm=True but vLLM is not installed in this environment.")
        if device.type != "cuda":
            raise RuntimeError("use_vllm=True requires CUDA device.")
        vllm_model_path = str(args.vllm_model_path or "").strip()
        if not vllm_model_path:
            vllm_model_path = _prepare_vllm_compatible_model_dir(args.model_path)
        vllm_runtime_model_path = str(vllm_model_path)
        logger.info("Using vLLM model path: %s", vllm_model_path)
        _patch_vllm_qwen3_loader_for_soft_refine()
        vllm_max_model_len = int(args.model_max_length) + int(args.max_new_tokens)
        logger.info(
            "Initializing vLLM backend: tp=%d, mem_util=%.2f, max_num_seqs=%d, max_model_len=%d",
            int(args.vllm_tensor_parallel_size),
            float(args.vllm_gpu_memory_utilization),
            int(args.vllm_max_num_seqs),
            int(vllm_max_model_len),
        )
        vllm_engine = LLM(
            model=vllm_model_path,
            tensor_parallel_size=max(1, int(args.vllm_tensor_parallel_size)),
            gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
            max_num_seqs=max(1, int(args.vllm_max_num_seqs)),
            max_model_len=int(vllm_max_model_len),
            dtype="bfloat16" if bool(args.bf16) else "float16",
            disable_custom_all_reduce=bool(args.vllm_disable_custom_all_reduce),
            enforce_eager=bool(args.vllm_enforce_eager),
        )
        vllm_sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=int(args.max_new_tokens),
            repetition_penalty=1.0,
        )
        vllm_eval_batch_size = int(args.vllm_eval_batch_size)
        if vllm_eval_batch_size <= 0:
            vllm_eval_batch_size = int(args.vllm_max_num_seqs)
        vllm_eval_batch_size = max(1, int(vllm_eval_batch_size))
        logger.info(
            "vLLM eval batching: eval_batch_size=%d (max_num_seqs=%d)",
            int(vllm_eval_batch_size),
            int(args.vllm_max_num_seqs),
        )
        logger.info("vLLM backend ready.")

    skip = {
        "query_not_found_in_meta": 0,
        "pos_doc_missing": 0,
        "video_file_missing": 0,
        "gt_time_invalid": 0,
    }
    video_meta_stats: Counter = Counter()

    orig_ranks: List[int] = []
    final_ranks: List[int] = []
    turn1_iou_r1: List[float] = []
    final_iou_r1: List[float] = []
    turn1_iou_matched_only: List[float] = []
    final_iou_matched_only: List[float] = []
    answer_counts: Counter = Counter()
    stop_reason_counts: Counter = Counter()
    matched_by_turn: Counter = Counter()
    refine_counters: Counter = Counter()
    policy_ranks_by_turn: Dict[int, List[int]] = defaultdict(list)
    strict_ranks_by_turn: Dict[int, List[int]] = defaultdict(list)
    policy_iou_r1_by_turn: Dict[int, List[float]] = defaultdict(list)
    strict_iou_r1_by_turn: Dict[int, List[float]] = defaultdict(list)
    policy_iou_matched_by_turn: Dict[int, List[float]] = defaultdict(list)
    strict_iou_matched_by_turn: Dict[int, List[float]] = defaultdict(list)
    query_vector_dump_saved = 0
    query_vector_dump_failed = 0
    query_vector_dump_skipped_filter = 0
    query_vector_dump_skipped_cap = 0
    query_vector_dump_bytes = 0

    streamed_detail_rows = 0
    done_indices: set[int] = set()
    output_jsonl = str(args.output_jsonl or "").strip()
    jsonl_writer = None

    if output_jsonl:
        out_dir = os.path.dirname(output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        resume_enabled = bool(args.resume_from_jsonl and os.path.exists(output_jsonl))
        if resume_enabled:
            existing_by_index: Dict[int, dict] = {}
            bad_lines = 0
            with open(output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row_obj = json.loads(line)
                    except Exception:
                        bad_lines += 1
                        continue
                    idx_val = row_obj.get("index")
                    if isinstance(idx_val, int):
                        existing_by_index[idx_val] = row_obj
            for idx_val in sorted(existing_by_index.keys()):
                row_obj = existing_by_index[idx_val]
                done_indices.add(int(idx_val))
                _accumulate_summary_from_detail(
                    row=row_obj,
                    orig_ranks=orig_ranks,
                    final_ranks=final_ranks,
                    turn1_iou_r1=turn1_iou_r1,
                    final_iou_r1=final_iou_r1,
                    turn1_iou_matched_only=turn1_iou_matched_only,
                    final_iou_matched_only=final_iou_matched_only,
                    answer_counts=answer_counts,
                    stop_reason_counts=stop_reason_counts,
                    matched_by_turn=matched_by_turn,
                    refine_counters=refine_counters,
                    max_turn=int(args.max_turn),
                    policy_ranks_by_turn=policy_ranks_by_turn,
                    strict_ranks_by_turn=strict_ranks_by_turn,
                    policy_iou_r1_by_turn=policy_iou_r1_by_turn,
                    strict_iou_r1_by_turn=strict_iou_r1_by_turn,
                    policy_iou_matched_by_turn=policy_iou_matched_by_turn,
                    strict_iou_matched_by_turn=strict_iou_matched_by_turn,
                )
            streamed_detail_rows = len(existing_by_index)
            logger.info(
                f"Resume enabled from existing jsonl: {output_jsonl} "
                f"(rows={streamed_detail_rows}, bad_lines={bad_lines})"
            )
            jsonl_writer = open(output_jsonl, "a", encoding="utf-8", buffering=1)
        else:
            jsonl_writer = open(output_jsonl, "w", encoding="utf-8", buffering=1)
        logger.info(f"Streaming detail jsonl: {output_jsonl}")

    loop_start = time.perf_counter()
    progress_interval_sec = float(args.progress_interval_sec)
    last_progress_log_at = loop_start
    use_tqdm = bool(args.use_tqdm) and _TQDM_AVAILABLE
    if bool(args.use_tqdm) and not _TQDM_AVAILABLE:
        logger.warning("use_tqdm=True but tqdm is not installed; fallback to logger progress.")
    pbar = (
        _tqdm(
            total=len(test_rows),
            desc="temporal_eval",
            unit="sample",
            dynamic_ncols=True,
            mininterval=float(args.tqdm_mininterval),
        )
        if use_tqdm
        else None
    )

    def _maybe_log_progress(done: int, force: bool = False) -> None:
        nonlocal last_progress_log_at
        if use_tqdm:
            return
        now = time.perf_counter()
        if not force and (now - last_progress_log_at) < progress_interval_sec:
            return
        elapsed = now - loop_start
        done_n = int(max(0, done))
        total_n = int(len(test_rows))
        remain = max(0, total_n - done_n)
        rate = done_n / max(elapsed, 1e-9)
        eta_sec = remain / max(rate, 1e-9)
        logger.info(
            "progress "
            f"{done_n}/{total_n} ({100.0 * done_n / max(1, total_n):.1f}%) | "
            f"remaining={remain} eta={eta_sec:.1f}s elapsed={elapsed:.1f}s | "
            f"valid={len(final_ranks)} matched={answer_counts.get('matched', 0)} "
            f"not_matched={answer_counts.get('not_matched', 0)} "
            f"unknown={answer_counts.get('unknown', 0)}"
        )
        last_progress_log_at = now

    use_vllm_batch = bool(use_vllm and int(vllm_eval_batch_size) > 1)
    if use_vllm_batch:
        logger.info("Using batched vLLM eval loop (batch_size=%d).", int(vllm_eval_batch_size))

    def _update_progress_after_index(idx_val: int) -> None:
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "valid": len(final_ranks),
                    "matched": answer_counts.get("matched", 0),
                    "not_matched": answer_counts.get("not_matched", 0),
                    "unknown": answer_counts.get("unknown", 0),
                },
                refresh=False,
            )
        _maybe_log_progress(done=int(idx_val), force=(int(idx_val) == len(test_rows)))

    def _finalize_eval_row(
        *,
        row_index: int,
        qid: str,
        query_text: str,
        gt_pos_doc_id: str,
        gt_span: Tuple[float, float],
        orig_rank: int,
        turn_details: List[dict],
        final_rank: int,
        final_top1_doc: str,
        final_top1_is_gt: bool,
        final_answer: str,
        final_time_parse_ok: bool,
        final_pred_start,
        final_pred_end,
        final_iou_raw: float,
        final_iou_r1_val: float,
        matched_turn: int,
        stop_reason: str,
        used_refine_any: bool,
        refine_token_generated_turns: int,
        refine_applied_turns: int,
        q_orig_initial: Optional[torch.Tensor] = None,
        q_policy_turn_vectors: Optional[Dict[int, torch.Tensor]] = None,
        q_profile_turn_vectors: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
        gt_video_vec: Optional[torch.Tensor] = None,
    ) -> None:
        nonlocal streamed_detail_rows
        nonlocal query_vector_dump_saved, query_vector_dump_failed
        nonlocal query_vector_dump_skipped_filter, query_vector_dump_skipped_cap, query_vector_dump_bytes

        _accumulate_turn_metrics_from_policy_turns(
            policy_turns=turn_details,
            max_turn=int(args.max_turn),
            policy_ranks_by_turn=policy_ranks_by_turn,
            strict_ranks_by_turn=strict_ranks_by_turn,
            policy_iou_r1_by_turn=policy_iou_r1_by_turn,
            strict_iou_r1_by_turn=strict_iou_r1_by_turn,
            policy_iou_matched_by_turn=policy_iou_matched_by_turn,
            strict_iou_matched_by_turn=strict_iou_matched_by_turn,
        )

        turn1 = turn_details[0]
        turn1_iou_r1.append(float(turn1["iou_r1"]))
        if str(turn1["answer"]) == "matched" and bool(turn1["time_parse_ok"]):
            turn1_iou_matched_only.append(float(turn1["iou_raw"]))

        final_iou_r1.append(float(final_iou_r1_val))
        if str(final_answer) == "matched" and bool(final_time_parse_ok):
            final_iou_matched_only.append(float(final_iou_raw))

        orig_ranks.append(int(orig_rank))
        final_ranks.append(int(final_rank))
        answer_counts[str(final_answer)] += 1
        stop_reason_counts[str(stop_reason)] += 1
        if int(matched_turn) > 0:
            matched_by_turn[str(int(matched_turn))] += 1
        if bool(used_refine_any):
            refine_counters["samples_with_refine"] += 1
        refine_counters["refine_token_generated_turns"] += int(refine_token_generated_turns)
        refine_counters["refine_applied_turns"] += int(refine_applied_turns)

        if jsonl_writer is not None:
            detail_row = {
                "index": int(row_index),
                "qid": str(qid),
                "query": query_text,
                "gt_pos_doc_id": str(gt_pos_doc_id),
                "gt_start": float(gt_span[0]),
                "gt_end": float(gt_span[1]),
                "orig_rank": int(orig_rank),
                "turn1_rank": int(turn1["policy_rank"]),
                "turn1_rank_before_refine": int(turn1["rank_before_refine"]),
                "turn1_top1_doc_id": str(turn1["top1_doc_id"]),
                "turn1_top1_is_gt": bool(turn1["top1_is_gt"]),
                "turn1_policy_top1_doc_id": str(turn1["policy_top1_doc_id"]),
                "turn1_policy_top1_is_gt": bool(turn1["policy_top1_is_gt"]),
                "turn1_answer": str(turn1["answer"]),
                "turn1_time_parse_ok": bool(turn1["time_parse_ok"]),
                "turn1_pred_start": turn1["pred_start"],
                "turn1_pred_end": turn1["pred_end"],
                "turn1_iou_raw": float(turn1["iou_raw"]),
                "turn1_iou_r1": float(turn1["iou_r1"]),
                "final_rank": int(final_rank),
                "final_top1_doc_id": str(final_top1_doc),
                "final_top1_is_gt": bool(final_top1_is_gt),
                "final_answer": str(final_answer),
                "final_time_parse_ok": bool(final_time_parse_ok),
                "final_pred_start": final_pred_start,
                "final_pred_end": final_pred_end,
                "final_iou_raw": float(final_iou_raw),
                "final_iou_r1": float(final_iou_r1_val),
                "matched_turn": int(matched_turn),
                "stop_turn": int(len(turn_details)),
                "stop_reason": str(stop_reason),
                "used_refine_any": bool(used_refine_any),
                "refine_token_generated_turns": int(refine_token_generated_turns),
                "refine_applied_turns": int(refine_applied_turns),
                "max_turn": int(args.max_turn),
                "turns": turn_details,
            }
            jsonl_writer.write(json.dumps(detail_row, ensure_ascii=False) + "\n")
            jsonl_writer.flush()
            streamed_detail_rows += 1

        if save_query_vectors_enabled:
            if bool(save_query_vectors_only_refined) and (not bool(used_refine_any)):
                query_vector_dump_skipped_filter += 1
                return
            if save_query_vectors_max_bytes > 0 and int(query_vector_dump_bytes) >= int(save_query_vectors_max_bytes):
                query_vector_dump_skipped_cap += 1
                return
            try:
                saved_bytes = _save_query_vector_artifact(
                    save_dir=save_query_vectors_dir,
                    row_index=int(row_index),
                    qid=str(qid),
                    gt_pos_doc_id=str(gt_pos_doc_id),
                    query_text=str(query_text),
                    q_orig_initial=q_orig_initial,
                    gt_video_vec=gt_video_vec,
                    q_policy_turn_vectors=q_policy_turn_vectors,
                    q_profile_turn_vectors=q_profile_turn_vectors,
                    turn_details=turn_details,
                )
                if int(saved_bytes) > 0:
                    query_vector_dump_saved += 1
                    query_vector_dump_bytes += int(saved_bytes)
                else:
                    query_vector_dump_failed += 1
            except Exception as exc:
                logger.warning("Failed saving query vector artifact for index=%s: %s", row_index, exc)
                query_vector_dump_failed += 1

    try:
        if use_vllm_batch:

            def _run_vllm_batch(batch_items: List[Tuple[int, dict, int]]) -> None:
                states: List[dict] = []
                for _, row, row_index in batch_items:
                    query_text = str(row.get("fig_desc", "")).strip()
                    if not query_text:
                        skip["query_not_found_in_meta"] += 1
                        continue

                    meta_item = query_map.get(query_text)
                    if meta_item is None:
                        skip["query_not_found_in_meta"] += 1
                        continue

                    q_row, qid, pos_doc_meta = meta_item
                    gt_pos_doc_id = str(row.get("video", "")).strip() or str(pos_doc_meta)
                    pos_row = docid2row.get(gt_pos_doc_id)
                    if pos_row is None:
                        pos_row = docid2row.get(str(pos_doc_meta))
                        if pos_row is None:
                            skip["pos_doc_missing"] += 1
                            continue
                        gt_pos_doc_id = str(pos_doc_meta)

                    gt_time = row.get("time", None)
                    if not isinstance(gt_time, (list, tuple)) or len(gt_time) < 2:
                        skip["gt_time_invalid"] += 1
                        continue
                    gt_start = _parse_float(gt_time[0])
                    gt_end = _parse_float(gt_time[1])
                    if gt_start is None or gt_end is None:
                        skip["gt_time_invalid"] += 1
                        continue
                    if gt_end < gt_start:
                        gt_start, gt_end = gt_end, gt_start
                    gt_span = (float(gt_start), float(gt_end))

                    duration = _safe_float(row.get("duration", None), default=float("nan"))
                    if not np.isfinite(duration):
                        duration = None

                    q_orig = _resolve_initial_query_vector(
                        use_updated_query=use_updated_query,
                        query_row=int(q_row),
                        query_text=query_text,
                        query_embeddings=query_embeddings,
                        updated_query_cache=updated_query_cache,
                        query_embedder_model=query_embedder_model,
                        query_embedder_tokenizer=query_embedder_tokenizer,
                        query_embedder_head=getattr(model, "query_embedder_head", None),
                        qfinal_pooling=args.qfinal_pooling,
                        query_embedder_max_length=int(args.query_embedder_max_length),
                        query_embedder_input_prefix=str(args.query_embedder_input_prefix or ""),
                        retrieval_dim=int(retrieval_dim),
                        device=device,
                    )
                    sims_current = torch.mv(video_emb_t, q_orig)
                    orig_rank = _compute_rank_from_scores(sims_current, int(pos_row))

                    states.append(
                        {
                            "row_index": int(row_index),
                            "qid": str(qid),
                            "query_text": query_text,
                            "gt_pos_doc_id": str(gt_pos_doc_id),
                            "gt_span": gt_span,
                            "duration": duration,
                            "pos_row": int(pos_row),
                            "q_orig": q_orig,
                            "sims_current": sims_current,
                            "orig_rank": int(orig_rank),
                            "turn_details": [],
                            "used_refine_any": False,
                            "refine_token_generated_turns": 0,
                            "refine_applied_turns": 0,
                            "matched_turn": 0,
                            "stop_reason": "max_turn_reached",
                            "final_rank": int(orig_rank),
                            "final_top1_doc": "",
                            "final_answer": "unknown",
                            "final_time_parse_ok": False,
                            "final_pred_start": None,
                            "final_pred_end": None,
                            "final_iou_raw": 0.0,
                            "final_iou_r1_val": 0.0,
                            "final_top1_is_gt": False,
                            "video_missing": False,
                            "done": False,
                            "q_orig_initial": (
                                q_orig.detach().to(device="cpu", dtype=torch.float32)
                                if save_query_vectors_enabled
                                else None
                            ),
                            "q_policy_turn_vectors": {} if save_query_vectors_enabled else None,
                            "q_profile_turn_vectors": {} if save_query_vectors_enabled else None,
                            "gt_video_vec": (
                                video_emb_t[int(pos_row)].detach().to(device="cpu", dtype=torch.float32)
                                if save_query_vectors_enabled
                                else None
                            ),
                        }
                    )

                if not states:
                    return

                max_turn = int(args.max_turn)
                for turn in range(1, max_turn + 1):
                    active_state_indices: List[int] = []
                    vllm_inputs: List[Dict[str, object]] = []

                    for state_idx, state in enumerate(states):
                        if bool(state["done"]):
                            continue

                        sims_current = state["sims_current"]
                        pos_row = int(state["pos_row"])
                        gt_pos_doc_id = str(state["gt_pos_doc_id"])

                        top1_row = int(torch.argmax(sims_current).item())
                        top1_doc = row2doc[top1_row]
                        current_rank = _compute_rank_from_scores(sims_current, pos_row)
                        top1_is_gt = bool(str(top1_doc) == gt_pos_doc_id)

                        video_path = _resolve_video_path(args.video_root, top1_doc)
                        if video_path is None:
                            state["video_missing"] = True
                            state["done"] = True
                            continue

                        video_meta = None
                        if str(video_path).lower().endswith((".npy", ".npz")):
                            video_meta_stats["npy_or_npz_inputs"] += 1
                            video_meta = resolve_video_meta_for_video_path(video_path, video_meta_index)
                            if isinstance(video_meta, dict) and video_meta:
                                video_meta_stats["meta_payload_hits"] += 1
                            else:
                                video_meta_stats["meta_payload_misses"] += 1

                        generation_inputs = _build_generation_inputs(
                            processor=processor,
                            system_prompt=system_prompt,
                            query_text=str(state["query_text"]),
                            video_path=video_path,
                            video_meta=video_meta,
                            model_max_length=args.model_max_length,
                            image_patch_size=args.image_patch_size,
                            video_min_pixels=args.video_min_pixels,
                            video_max_pixels=args.video_max_pixels,
                            video_total_pixels=args.video_total_pixels,
                        )
                        state["turn_cache"] = {
                            "generation_inputs": generation_inputs,
                            "current_rank": int(current_rank),
                            "top1_doc": str(top1_doc),
                            "top1_is_gt": bool(top1_is_gt),
                        }
                        vllm_inputs.append(_build_vllm_input_from_generation_inputs(generation_inputs))
                        active_state_indices.append(int(state_idx))

                    if not vllm_inputs:
                        break

                    outputs = vllm_engine.generate(vllm_inputs, sampling_params=vllm_sampling_params, use_tqdm=False)

                    for out_idx, state_idx in enumerate(active_state_indices):
                        state = states[int(state_idx)]
                        turn_cache = state.pop("turn_cache", {})
                        generation_inputs = turn_cache.get("generation_inputs")
                        current_rank = int(turn_cache.get("current_rank", 0))
                        top1_doc = str(turn_cache.get("top1_doc", ""))
                        top1_is_gt = bool(turn_cache.get("top1_is_gt", False))

                        out_obj = outputs[out_idx] if out_idx < len(outputs) else None
                        token_ids: List[int] = []
                        if out_obj is not None and getattr(out_obj, "outputs", None):
                            token_ids = list(getattr(out_obj.outputs[0], "token_ids", []) or [])
                        completion_text = _decode_completion_from_token_ids(processor.tokenizer, token_ids)
                        if token_ids:
                            completion_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                        else:
                            completion_ids = torch.empty((1, 0), dtype=torch.long, device=device)

                        answer = _parse_answer(completion_text)
                        pred_span = _parse_temporal_span(completion_text, duration=state["duration"])
                        time_parse_ok = pred_span is not None
                        pred_start = float(pred_span[0]) if pred_span is not None else None
                        pred_end = float(pred_span[1]) if pred_span is not None else None

                        iou_raw = 0.0
                        if answer == "matched" and time_parse_ok:
                            iou_raw = _compute_iou(state["gt_span"], pred_span)
                        iou_r1 = iou_raw if (answer == "matched" and time_parse_ok and top1_is_gt) else 0.0

                        had_refine_token = False
                        used_refine_this_turn = False
                        qfinal_mode = "none"
                        rollout_debug = _summarize_rollout_latents(None)
                        qfinal_debug = {
                            "forward_mode": "none",
                            "input_ids_is_none": None,
                            "inputs_embeds_shape": [],
                            "full_attention_shape": [],
                        }
                        policy_rank = int(current_rank)
                        policy_top1_doc = str(top1_doc)
                        policy_top1_is_gt = bool(top1_is_gt)
                        q_policy_vec = state["q_orig"]
                        profile_rank_by_depth: Dict[str, int] = {}
                        profile_top1_doc_by_depth: Dict[str, str] = {}
                        profile_top1_is_gt_by_depth: Dict[str, bool] = {}
                        q_profile_vec_by_depth: Dict[int, torch.Tensor] = {}

                        if answer == "matched":
                            state["matched_turn"] = int(turn)
                            state["stop_reason"] = "matched"
                            state["done"] = True
                        else:
                            update = None
                            rollout_model_inputs = None
                            if completion_ids.numel() > 0:
                                comp_mask = torch.zeros_like(completion_ids[0], dtype=torch.bool)
                                for tok_id in refine_token_ids:
                                    comp_mask |= completion_ids[0].eq(int(tok_id))
                                had_refine_token = bool(comp_mask.any().item())
                            else:
                                had_refine_token = False

                            if had_refine_token and generation_inputs is not None:
                                model_inputs = _build_generation_batch_from_inputs(
                                    processor=processor,
                                    generation_inputs=generation_inputs,
                                    model_max_length=args.model_max_length,
                                )
                                model_inputs = _move_batch_to_device(model_inputs, device=device)
                                prompt_len = int(model_inputs["input_ids"].shape[1])
                                full_ids = torch.cat([model_inputs["input_ids"], completion_ids], dim=1)
                                update, _ = _extract_update_from_ids(
                                    model=model,
                                    model_inputs=model_inputs,
                                    full_ids=full_ids,
                                    prompt_len=prompt_len,
                                    completion_ids=completion_ids,
                                    refine_token_ids=refine_token_ids,
                                    use_refine_gate=bool(args.use_refine_gate),
                                )
                                last_comp_ref_pos = _find_last_refine_pos_in_completion(
                                    completion_ids=completion_ids,
                                    refine_token_ids=refine_token_ids,
                                )
                                rollout_full_ids = full_ids
                                if last_comp_ref_pos is not None:
                                    rollout_full_ids = _truncate_full_ids_to_last_refine(
                                        full_ids=full_ids,
                                        prompt_len=prompt_len,
                                        last_comp_ref_pos=last_comp_ref_pos,
                                    )
                                # [legacy full-completion hidden-state fallback]
                                # rollout_full_ids = full_ids
                                rollout_model_inputs = _build_rollout_inputs_with_full_ids(
                                    model_inputs=model_inputs,
                                    full_ids=rollout_full_ids,
                                )

                            if had_refine_token and update is not None:
                                state["refine_token_generated_turns"] += 1
                                rollout_update = _rollout_refine_latents(
                                    model=model,
                                    first_update=update,
                                    rollout_depth=refine_rollout_depth,
                                    row_model_inputs=rollout_model_inputs,
                                    refine_token_ids=refine_token_ids,
                                    use_refine_gate=bool(args.use_refine_gate),
                                    qfinal_pooling=args.qfinal_pooling,
                                )
                                rollout_debug = _summarize_rollout_latents(rollout_update)
                                q_next, qfinal_mode, qfinal_debug = _build_q_final(
                                    q_orig=state["q_orig"],
                                    update=rollout_update,
                                    query_text=state["query_text"],
                                    use_query_embedder_path=bool(args.use_query_embedder_path),
                                    query_embedder_model=query_embedder_model,
                                    query_embedder_tokenizer=query_embedder_tokenizer,
                                    query_embedder_head=getattr(model, "query_embedder_head", None),
                                    append_input_projector=getattr(
                                        model, "refine_append_input_projector", None
                                    ),
                                    latent_input_projector=getattr(
                                        model, "refine_latent_input_projector", None
                                    ),
                                    qfinal_pooling=args.qfinal_pooling,
                                    qfinal_normalize=bool(args.qfinal_normalize),
                                    query_embedder_max_length=int(args.query_embedder_max_length),
                                    query_embedder_input_prefix=str(args.query_embedder_input_prefix or ""),
                                )
                                sims_next = torch.mv(video_emb_t, q_next)
                                policy_rank = _compute_rank_from_scores(sims_next, int(state["pos_row"]))
                                policy_top1_row = int(torch.argmax(sims_next).item())
                                policy_top1_doc = row2doc[policy_top1_row]
                                policy_top1_is_gt = bool(str(policy_top1_doc) == str(state["gt_pos_doc_id"]))
                                q_policy_vec = q_next
                                if profile_rollout_depths:
                                    rollout_actual = int(rollout_update.size(0))
                                    for pd in profile_rollout_depths:
                                        pd_i = int(pd)
                                        if pd_i < 1 or pd_i > rollout_actual:
                                            continue
                                        if pd_i == rollout_actual:
                                            q_pd = q_next
                                            sims_pd = sims_next
                                        else:
                                            q_pd, _, _ = _build_q_final(
                                                q_orig=state["q_orig"],
                                                update=rollout_update[:pd_i],
                                                query_text=state["query_text"],
                                                use_query_embedder_path=bool(args.use_query_embedder_path),
                                                query_embedder_model=query_embedder_model,
                                                query_embedder_tokenizer=query_embedder_tokenizer,
                                                query_embedder_head=getattr(model, "query_embedder_head", None),
                                                append_input_projector=getattr(
                                                    model, "refine_append_input_projector", None
                                                ),
                                                latent_input_projector=getattr(
                                                    model, "refine_latent_input_projector", None
                                                ),
                                                qfinal_pooling=args.qfinal_pooling,
                                                qfinal_normalize=bool(args.qfinal_normalize),
                                                query_embedder_max_length=int(args.query_embedder_max_length),
                                                query_embedder_input_prefix=str(args.query_embedder_input_prefix or ""),
                                            )
                                            sims_pd = torch.mv(video_emb_t, q_pd)
                                        rank_pd = _compute_rank_from_scores(sims_pd, int(state["pos_row"]))
                                        top1_row_pd = int(torch.argmax(sims_pd).item())
                                        top1_doc_pd = row2doc[top1_row_pd]
                                        top1_is_gt_pd = bool(str(top1_doc_pd) == str(state["gt_pos_doc_id"]))
                                        profile_rank_by_depth[str(pd_i)] = int(rank_pd)
                                        profile_top1_doc_by_depth[str(pd_i)] = str(top1_doc_pd)
                                        profile_top1_is_gt_by_depth[str(pd_i)] = bool(top1_is_gt_pd)
                                        if save_query_vectors_enabled:
                                            q_profile_vec_by_depth[pd_i] = q_pd.detach().to(
                                                device="cpu",
                                                dtype=torch.float32,
                                            )
                                used_refine_this_turn = True
                                state["used_refine_any"] = True
                                state["refine_applied_turns"] += 1
                                if turn < max_turn:
                                    state["q_orig"] = q_next
                                    state["sims_current"] = sims_next
                                    state["stop_reason"] = "continue_refine"
                                    state["done"] = False
                                else:
                                    state["stop_reason"] = "max_turn_reached_with_refine"
                                    state["done"] = True
                            else:
                                if had_refine_token and update is None:
                                    state["stop_reason"] = "refine_token_no_update"
                                elif answer == "not_matched":
                                    state["stop_reason"] = "not_matched_no_refine_token"
                                else:
                                    state["stop_reason"] = "unknown_no_refine_token"
                                state["done"] = True

                        if save_query_vectors_enabled and isinstance(state.get("q_policy_turn_vectors"), dict):
                            state["q_policy_turn_vectors"][int(turn)] = (
                                q_policy_vec.detach().to(device="cpu", dtype=torch.float32)
                                if torch.is_tensor(q_policy_vec)
                                else None
                            )
                        if save_query_vectors_enabled and isinstance(state.get("q_profile_turn_vectors"), dict):
                            state["q_profile_turn_vectors"][int(turn)] = q_profile_vec_by_depth

                        turn_detail = {
                            "turn": int(turn),
                            "rank": int(current_rank),
                            "rank_before_refine": int(current_rank),
                            "policy_rank": int(policy_rank),
                            "top1_doc_id": str(top1_doc),
                            "top1_is_gt": bool(top1_is_gt),
                            "policy_top1_doc_id": str(policy_top1_doc),
                            "policy_top1_is_gt": bool(policy_top1_is_gt),
                            "answer": str(answer),
                            "time_parse_ok": bool(time_parse_ok),
                            "pred_start": pred_start,
                            "pred_end": pred_end,
                            "iou_raw": float(iou_raw),
                            "iou_r1": float(iou_r1),
                            "refine_token_generated": bool(had_refine_token),
                            "refine_applied": bool(used_refine_this_turn),
                            "qfinal_mode": qfinal_mode,
                            "rollout_depth_config": int(refine_rollout_depth),
                            "rollout_depth_actual": int(rollout_debug["rollout_depth_actual"]),
                            "rollout_shape": rollout_debug["rollout_shape"],
                            "rollout_latent_norms": rollout_debug["rollout_latent_norms"],
                            "rollout_latent_cos_prev": rollout_debug["rollout_latent_cos_prev"],
                            "rollout_latent_head": rollout_debug["rollout_latent_head"],
                            "latent_injected_to_llm": bool(
                                used_refine_this_turn and qfinal_debug.get("forward_mode") == "inputs_embeds"
                            ),
                            "qfinal_forward_mode": str(qfinal_debug.get("forward_mode", "none")),
                            "qfinal_input_ids_is_none": qfinal_debug.get("input_ids_is_none", None),
                            "qfinal_inputs_embeds_shape": qfinal_debug.get("inputs_embeds_shape", []),
                            "qfinal_full_attention_shape": qfinal_debug.get("full_attention_shape", []),
                            "profile_policy_rank_by_depth": profile_rank_by_depth,
                            "profile_policy_top1_doc_by_depth": profile_top1_doc_by_depth,
                            "profile_policy_top1_is_gt_by_depth": profile_top1_is_gt_by_depth,
                            "raw_output": completion_text,
                        }
                        state["turn_details"].append(turn_detail)

                        state["final_rank"] = int(policy_rank)
                        state["final_top1_doc"] = str(policy_top1_doc)
                        state["final_answer"] = str(answer)
                        state["final_time_parse_ok"] = bool(time_parse_ok)
                        state["final_pred_start"] = pred_start
                        state["final_pred_end"] = pred_end
                        state["final_iou_raw"] = float(iou_raw)
                        state["final_iou_r1_val"] = float(iou_r1)
                        state["final_top1_is_gt"] = bool(policy_top1_is_gt)

                        if answer == "matched":
                            state["done"] = True
                        elif not had_refine_token:
                            state["done"] = True
                        elif turn >= max_turn:
                            state["done"] = True

                    if all(bool(s["done"]) for s in states):
                        break

                for state in states:
                    if bool(state["video_missing"]):
                        skip["video_file_missing"] += 1
                        continue
                    if not state["turn_details"]:
                        continue

                    _finalize_eval_row(
                        row_index=int(state["row_index"]),
                        qid=str(state["qid"]),
                        query_text=str(state["query_text"]),
                        gt_pos_doc_id=str(state["gt_pos_doc_id"]),
                        gt_span=state["gt_span"],
                        orig_rank=int(state["orig_rank"]),
                        turn_details=state["turn_details"],
                        final_rank=int(state["final_rank"]),
                        final_top1_doc=str(state["final_top1_doc"]),
                        final_top1_is_gt=bool(state["final_top1_is_gt"]),
                        final_answer=str(state["final_answer"]),
                        final_time_parse_ok=bool(state["final_time_parse_ok"]),
                        final_pred_start=state["final_pred_start"],
                        final_pred_end=state["final_pred_end"],
                        final_iou_raw=float(state["final_iou_raw"]),
                        final_iou_r1_val=float(state["final_iou_r1_val"]),
                        matched_turn=int(state["matched_turn"]),
                        stop_reason=str(state["stop_reason"]),
                        used_refine_any=bool(state["used_refine_any"]),
                        refine_token_generated_turns=int(state["refine_token_generated_turns"]),
                        refine_applied_turns=int(state["refine_applied_turns"]),
                        q_orig_initial=state.get("q_orig_initial"),
                        q_policy_turn_vectors=state.get("q_policy_turn_vectors"),
                        q_profile_turn_vectors=state.get("q_profile_turn_vectors"),
                        gt_video_vec=state.get("gt_video_vec"),
                    )

            pending_batch: List[Tuple[int, dict, int]] = []
            for idx, (row_index, row) in enumerate(zip(test_row_indices, test_rows), start=1):
                if row_index in done_indices:
                    _update_progress_after_index(idx)
                    continue

                pending_batch.append((idx, row, row_index))
                if len(pending_batch) >= int(vllm_eval_batch_size):
                    _run_vllm_batch(pending_batch)
                    for done_idx, _, _ in pending_batch:
                        _update_progress_after_index(done_idx)
                    pending_batch = []

            if pending_batch:
                _run_vllm_batch(pending_batch)
                for done_idx, _, _ in pending_batch:
                    _update_progress_after_index(done_idx)
        else:
            for idx, (row_index, row) in enumerate(zip(test_row_indices, test_rows), start=1):
                try:
                    if row_index in done_indices:
                        continue

                    query_text = str(row.get("fig_desc", "")).strip()
                    if not query_text:
                        skip["query_not_found_in_meta"] += 1
                        continue

                    meta_item = query_map.get(query_text)
                    if meta_item is None:
                        skip["query_not_found_in_meta"] += 1
                        continue

                    q_row, qid, pos_doc_meta = meta_item
                    gt_pos_doc_id = str(row.get("video", "")).strip() or str(pos_doc_meta)
                    pos_row = docid2row.get(gt_pos_doc_id)
                    if pos_row is None:
                        pos_row = docid2row.get(str(pos_doc_meta))
                        if pos_row is None:
                            skip["pos_doc_missing"] += 1
                            continue
                        gt_pos_doc_id = str(pos_doc_meta)

                    gt_time = row.get("time", None)
                    if not isinstance(gt_time, (list, tuple)) or len(gt_time) < 2:
                        skip["gt_time_invalid"] += 1
                        continue
                    gt_start = _parse_float(gt_time[0])
                    gt_end = _parse_float(gt_time[1])
                    if gt_start is None or gt_end is None:
                        skip["gt_time_invalid"] += 1
                        continue
                    if gt_end < gt_start:
                        gt_start, gt_end = gt_end, gt_start
                    gt_span = (float(gt_start), float(gt_end))
                    duration = _safe_float(row.get("duration", None), default=float("nan"))
                    if not np.isfinite(duration):
                        duration = None

                    q_orig = _resolve_initial_query_vector(
                        use_updated_query=use_updated_query,
                        query_row=int(q_row),
                        query_text=query_text,
                        query_embeddings=query_embeddings,
                        updated_query_cache=updated_query_cache,
                        query_embedder_model=query_embedder_model,
                        query_embedder_tokenizer=query_embedder_tokenizer,
                        query_embedder_head=getattr(model, "query_embedder_head", None),
                        qfinal_pooling=args.qfinal_pooling,
                        query_embedder_max_length=int(args.query_embedder_max_length),
                        query_embedder_input_prefix=str(args.query_embedder_input_prefix or ""),
                        retrieval_dim=int(retrieval_dim),
                        device=device,
                    )
                    sims_current = torch.mv(video_emb_t, q_orig)
                    orig_rank = _compute_rank_from_scores(sims_current, int(pos_row))
                    q_orig_initial = (
                        q_orig.detach().to(device="cpu", dtype=torch.float32)
                        if save_query_vectors_enabled
                        else None
                    )
                    q_policy_turn_vectors: Optional[Dict[int, torch.Tensor]] = (
                        {} if save_query_vectors_enabled else None
                    )
                    q_profile_turn_vectors: Optional[Dict[int, Dict[int, torch.Tensor]]] = (
                        {} if save_query_vectors_enabled else None
                    )
                    gt_video_vec = (
                        video_emb_t[int(pos_row)].detach().to(device="cpu", dtype=torch.float32)
                        if save_query_vectors_enabled
                        else None
                    )

                    turn_details = []
                    used_refine_any = False
                    refine_token_generated_turns = 0
                    refine_applied_turns = 0
                    matched_turn = 0
                    stop_reason = "max_turn_reached"

                    final_rank = int(orig_rank)
                    final_top1_doc = ""
                    final_answer = "unknown"
                    final_time_parse_ok = False
                    final_pred_start = None
                    final_pred_end = None
                    final_iou_raw = 0.0
                    final_iou_r1_val = 0.0
                    final_top1_is_gt = False

                    video_missing = False

                    for turn in range(1, int(args.max_turn) + 1):
                        top1_row = int(torch.argmax(sims_current).item())
                        top1_doc = row2doc[top1_row]
                        current_rank = _compute_rank_from_scores(sims_current, int(pos_row))
                        top1_is_gt = bool(str(top1_doc) == str(gt_pos_doc_id))

                        video_path = _resolve_video_path(args.video_root, top1_doc)
                        if video_path is None:
                            video_missing = True
                            break
                        video_meta = None
                        if str(video_path).lower().endswith((".npy", ".npz")):
                            video_meta_stats["npy_or_npz_inputs"] += 1
                            video_meta = resolve_video_meta_for_video_path(video_path, video_meta_index)
                            if isinstance(video_meta, dict) and video_meta:
                                video_meta_stats["meta_payload_hits"] += 1
                            else:
                                video_meta_stats["meta_payload_misses"] += 1

                        generation_inputs = _build_generation_inputs(
                            processor=processor,
                            system_prompt=system_prompt,
                            query_text=query_text,
                            video_path=video_path,
                            video_meta=video_meta,
                            model_max_length=args.model_max_length,
                            image_patch_size=args.image_patch_size,
                            video_min_pixels=args.video_min_pixels,
                            video_max_pixels=args.video_max_pixels,
                            video_total_pixels=args.video_total_pixels,
                        )
                        model_inputs = None
                        prompt_len = 0
                        full_ids = None
                        if use_vllm:
                            vllm_input = _build_vllm_input_from_generation_inputs(generation_inputs)
                            outputs = vllm_engine.generate(
                                [vllm_input], sampling_params=vllm_sampling_params, use_tqdm=False
                            )
                            token_ids: List[int] = []
                            if outputs and getattr(outputs[0], "outputs", None):
                                token_ids = list(getattr(outputs[0].outputs[0], "token_ids", []) or [])
                            completion_text = _decode_completion_from_token_ids(processor.tokenizer, token_ids)
                            if token_ids:
                                completion_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                            else:
                                completion_ids = torch.empty((1, 0), dtype=torch.long, device=device)
                        else:
                            model_inputs = _build_generation_batch_from_inputs(
                                processor=processor,
                                generation_inputs=generation_inputs,
                                model_max_length=args.model_max_length,
                            )
                            model_inputs = _move_batch_to_device(model_inputs, device=device)
                            with torch.no_grad():
                                full_ids = model.generate(
                                    **model_inputs,
                                    max_new_tokens=int(args.max_new_tokens),
                                    do_sample=False,
                                )
                            prompt_len = int(model_inputs["input_ids"].shape[1])
                            completion_ids = full_ids[:, prompt_len:]
                            completion_text = processor.tokenizer.batch_decode(
                                completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                            )[0]

                        answer = _parse_answer(completion_text)
                        pred_span = _parse_temporal_span(completion_text, duration=duration)
                        time_parse_ok = pred_span is not None
                        pred_start = float(pred_span[0]) if pred_span is not None else None
                        pred_end = float(pred_span[1]) if pred_span is not None else None

                        iou_raw = 0.0
                        if answer == "matched" and time_parse_ok:
                            iou_raw = _compute_iou(gt_span, pred_span)
                        iou_r1 = iou_raw if (answer == "matched" and time_parse_ok and top1_is_gt) else 0.0

                        had_refine_token = False
                        used_refine_this_turn = False
                        qfinal_mode = "none"
                        rollout_debug = _summarize_rollout_latents(None)
                        qfinal_debug = {
                            "forward_mode": "none",
                            "input_ids_is_none": None,
                            "inputs_embeds_shape": [],
                            "full_attention_shape": [],
                        }
                        policy_rank = int(current_rank)
                        policy_top1_doc = str(top1_doc)
                        policy_top1_is_gt = bool(top1_is_gt)
                        q_policy_vec = q_orig
                        profile_rank_by_depth: Dict[str, int] = {}
                        profile_top1_doc_by_depth: Dict[str, str] = {}
                        profile_top1_is_gt_by_depth: Dict[str, bool] = {}
                        q_profile_vec_by_depth: Dict[int, torch.Tensor] = {}

                        if answer == "matched":
                            matched_turn = turn
                            stop_reason = "matched"
                        else:
                            update = None
                            rollout_model_inputs = model_inputs
                            if use_vllm:
                                if completion_ids.numel() > 0:
                                    comp_mask = torch.zeros_like(completion_ids[0], dtype=torch.bool)
                                    for tok_id in refine_token_ids:
                                        comp_mask |= completion_ids[0].eq(int(tok_id))
                                    had_refine_token = bool(comp_mask.any().item())
                                else:
                                    had_refine_token = False
                                if had_refine_token:
                                    if model_inputs is None:
                                        model_inputs = _build_generation_batch_from_inputs(
                                            processor=processor,
                                            generation_inputs=generation_inputs,
                                            model_max_length=args.model_max_length,
                                        )
                                        model_inputs = _move_batch_to_device(model_inputs, device=device)
                                    prompt_len = int(model_inputs["input_ids"].shape[1])
                                    full_ids = torch.cat([model_inputs["input_ids"], completion_ids], dim=1)
                                    update, _ = _extract_update_from_ids(
                                        model=model,
                                        model_inputs=model_inputs,
                                        full_ids=full_ids,
                                        prompt_len=prompt_len,
                                        completion_ids=completion_ids,
                                        refine_token_ids=refine_token_ids,
                                        use_refine_gate=bool(args.use_refine_gate),
                                    )
                                    last_comp_ref_pos = _find_last_refine_pos_in_completion(
                                        completion_ids=completion_ids,
                                        refine_token_ids=refine_token_ids,
                                    )
                                    rollout_full_ids = full_ids
                                    if last_comp_ref_pos is not None:
                                        rollout_full_ids = _truncate_full_ids_to_last_refine(
                                            full_ids=full_ids,
                                            prompt_len=prompt_len,
                                            last_comp_ref_pos=last_comp_ref_pos,
                                        )
                                    # [legacy full-completion hidden-state fallback]
                                    # rollout_full_ids = full_ids
                                    rollout_model_inputs = _build_rollout_inputs_with_full_ids(
                                        model_inputs=model_inputs,
                                        full_ids=rollout_full_ids,
                                    )
                            else:
                                update, had_refine_token = _extract_update_from_ids(
                                    model=model,
                                    model_inputs=model_inputs,
                                    full_ids=full_ids,
                                    prompt_len=prompt_len,
                                    completion_ids=completion_ids,
                                    refine_token_ids=refine_token_ids,
                                    use_refine_gate=bool(args.use_refine_gate),
                                )
                                if had_refine_token and full_ids is not None and model_inputs is not None:
                                    last_comp_ref_pos = _find_last_refine_pos_in_completion(
                                        completion_ids=completion_ids,
                                        refine_token_ids=refine_token_ids,
                                    )
                                    rollout_full_ids = full_ids
                                    if last_comp_ref_pos is not None:
                                        rollout_full_ids = _truncate_full_ids_to_last_refine(
                                            full_ids=full_ids,
                                            prompt_len=prompt_len,
                                            last_comp_ref_pos=last_comp_ref_pos,
                                        )
                                    # [legacy full-completion hidden-state fallback]
                                    # rollout_full_ids = full_ids
                                    rollout_model_inputs = _build_rollout_inputs_with_full_ids(
                                        model_inputs=model_inputs,
                                        full_ids=rollout_full_ids,
                                    )

                            if had_refine_token and update is not None:
                                refine_token_generated_turns += 1
                                rollout_update = _rollout_refine_latents(
                                    model=model,
                                    first_update=update,
                                    rollout_depth=refine_rollout_depth,
                                    row_model_inputs=rollout_model_inputs,
                                    refine_token_ids=refine_token_ids,
                                    use_refine_gate=bool(args.use_refine_gate),
                                    qfinal_pooling=args.qfinal_pooling,
                                )
                                rollout_debug = _summarize_rollout_latents(rollout_update)
                                q_next, qfinal_mode, qfinal_debug = _build_q_final(
                                    q_orig=q_orig,
                                    update=rollout_update,
                                    query_text=query_text,
                                    use_query_embedder_path=bool(args.use_query_embedder_path),
                                    query_embedder_model=query_embedder_model,
                                    query_embedder_tokenizer=query_embedder_tokenizer,
                                    query_embedder_head=getattr(model, "query_embedder_head", None),
                                    append_input_projector=getattr(
                                        model, "refine_append_input_projector", None
                                    ),
                                    latent_input_projector=getattr(
                                        model, "refine_latent_input_projector", None
                                    ),
                                    qfinal_pooling=args.qfinal_pooling,
                                    qfinal_normalize=bool(args.qfinal_normalize),
                                    query_embedder_max_length=int(args.query_embedder_max_length),
                                    query_embedder_input_prefix=str(args.query_embedder_input_prefix or ""),
                                )
                                sims_next = torch.mv(video_emb_t, q_next)
                                policy_rank = _compute_rank_from_scores(sims_next, int(pos_row))
                                policy_top1_row = int(torch.argmax(sims_next).item())
                                policy_top1_doc = row2doc[policy_top1_row]
                                policy_top1_is_gt = bool(str(policy_top1_doc) == str(gt_pos_doc_id))
                                q_policy_vec = q_next
                                if profile_rollout_depths:
                                    rollout_actual = int(rollout_update.size(0))
                                    for pd in profile_rollout_depths:
                                        pd_i = int(pd)
                                        if pd_i < 1 or pd_i > rollout_actual:
                                            continue
                                        if pd_i == rollout_actual:
                                            q_pd = q_next
                                            sims_pd = sims_next
                                        else:
                                            q_pd, _, _ = _build_q_final(
                                                q_orig=q_orig,
                                                update=rollout_update[:pd_i],
                                                query_text=query_text,
                                                use_query_embedder_path=bool(args.use_query_embedder_path),
                                                query_embedder_model=query_embedder_model,
                                                query_embedder_tokenizer=query_embedder_tokenizer,
                                                query_embedder_head=getattr(model, "query_embedder_head", None),
                                                append_input_projector=getattr(
                                                    model, "refine_append_input_projector", None
                                                ),
                                                latent_input_projector=getattr(
                                                    model, "refine_latent_input_projector", None
                                                ),
                                                qfinal_pooling=args.qfinal_pooling,
                                                qfinal_normalize=bool(args.qfinal_normalize),
                                                query_embedder_max_length=int(args.query_embedder_max_length),
                                                query_embedder_input_prefix=str(args.query_embedder_input_prefix or ""),
                                            )
                                            sims_pd = torch.mv(video_emb_t, q_pd)
                                        rank_pd = _compute_rank_from_scores(sims_pd, int(pos_row))
                                        top1_row_pd = int(torch.argmax(sims_pd).item())
                                        top1_doc_pd = row2doc[top1_row_pd]
                                        top1_is_gt_pd = bool(str(top1_doc_pd) == str(gt_pos_doc_id))
                                        profile_rank_by_depth[str(pd_i)] = int(rank_pd)
                                        profile_top1_doc_by_depth[str(pd_i)] = str(top1_doc_pd)
                                        profile_top1_is_gt_by_depth[str(pd_i)] = bool(top1_is_gt_pd)
                                        if save_query_vectors_enabled:
                                            q_profile_vec_by_depth[pd_i] = q_pd.detach().to(
                                                device="cpu",
                                                dtype=torch.float32,
                                            )
                                used_refine_this_turn = True
                                used_refine_any = True
                                refine_applied_turns += 1
                                if turn < int(args.max_turn):
                                    q_orig = q_next
                                    sims_current = sims_next
                                    stop_reason = "continue_refine"
                                else:
                                    stop_reason = "max_turn_reached_with_refine"
                            else:
                                if had_refine_token and update is None:
                                    stop_reason = "refine_token_no_update"
                                elif answer == "not_matched":
                                    stop_reason = "not_matched_no_refine_token"
                                else:
                                    stop_reason = "unknown_no_refine_token"

                        if save_query_vectors_enabled and isinstance(q_policy_turn_vectors, dict):
                            q_policy_turn_vectors[int(turn)] = (
                                q_policy_vec.detach().to(device="cpu", dtype=torch.float32)
                                if torch.is_tensor(q_policy_vec)
                                else None
                            )
                        if save_query_vectors_enabled and isinstance(q_profile_turn_vectors, dict):
                            q_profile_turn_vectors[int(turn)] = q_profile_vec_by_depth

                        turn_detail = {
                            "turn": int(turn),
                            "rank": int(current_rank),
                            "rank_before_refine": int(current_rank),
                            "policy_rank": int(policy_rank),
                            "top1_doc_id": str(top1_doc),
                            "top1_is_gt": bool(top1_is_gt),
                            "policy_top1_doc_id": str(policy_top1_doc),
                            "policy_top1_is_gt": bool(policy_top1_is_gt),
                            "answer": str(answer),
                            "time_parse_ok": bool(time_parse_ok),
                            "pred_start": pred_start,
                            "pred_end": pred_end,
                            "iou_raw": float(iou_raw),
                            "iou_r1": float(iou_r1),
                            "refine_token_generated": bool(had_refine_token),
                            "refine_applied": bool(used_refine_this_turn),
                            "qfinal_mode": qfinal_mode,
                            "rollout_depth_config": int(refine_rollout_depth),
                            "rollout_depth_actual": int(rollout_debug["rollout_depth_actual"]),
                            "rollout_shape": rollout_debug["rollout_shape"],
                            "rollout_latent_norms": rollout_debug["rollout_latent_norms"],
                            "rollout_latent_cos_prev": rollout_debug["rollout_latent_cos_prev"],
                            "rollout_latent_head": rollout_debug["rollout_latent_head"],
                            "latent_injected_to_llm": bool(
                                used_refine_this_turn and qfinal_debug.get("forward_mode") == "inputs_embeds"
                            ),
                            "qfinal_forward_mode": str(qfinal_debug.get("forward_mode", "none")),
                            "qfinal_input_ids_is_none": qfinal_debug.get("input_ids_is_none", None),
                            "qfinal_inputs_embeds_shape": qfinal_debug.get("inputs_embeds_shape", []),
                            "qfinal_full_attention_shape": qfinal_debug.get("full_attention_shape", []),
                            "profile_policy_rank_by_depth": profile_rank_by_depth,
                            "profile_policy_top1_doc_by_depth": profile_top1_doc_by_depth,
                            "profile_policy_top1_is_gt_by_depth": profile_top1_is_gt_by_depth,
                            "raw_output": completion_text,
                        }
                        turn_details.append(turn_detail)

                        final_rank = int(policy_rank)
                        final_top1_doc = str(policy_top1_doc)
                        final_answer = str(answer)
                        final_time_parse_ok = bool(time_parse_ok)
                        final_pred_start = pred_start
                        final_pred_end = pred_end
                        final_iou_raw = float(iou_raw)
                        final_iou_r1_val = float(iou_r1)
                        final_top1_is_gt = bool(policy_top1_is_gt)

                        if answer == "matched":
                            break
                        if not had_refine_token:
                            break
                        if turn >= int(args.max_turn):
                            break

                    if video_missing:
                        skip["video_file_missing"] += 1
                        continue
                    if not turn_details:
                        continue

                    _finalize_eval_row(
                        row_index=int(row_index),
                        qid=str(qid),
                        query_text=query_text,
                        gt_pos_doc_id=str(gt_pos_doc_id),
                        gt_span=gt_span,
                        orig_rank=int(orig_rank),
                        turn_details=turn_details,
                        final_rank=int(final_rank),
                        final_top1_doc=str(final_top1_doc),
                        final_top1_is_gt=bool(final_top1_is_gt),
                        final_answer=str(final_answer),
                        final_time_parse_ok=bool(final_time_parse_ok),
                        final_pred_start=final_pred_start,
                        final_pred_end=final_pred_end,
                        final_iou_raw=float(final_iou_raw),
                        final_iou_r1_val=float(final_iou_r1_val),
                        matched_turn=int(matched_turn),
                        stop_reason=str(stop_reason),
                        used_refine_any=bool(used_refine_any),
                        refine_token_generated_turns=int(refine_token_generated_turns),
                        refine_applied_turns=int(refine_applied_turns),
                        q_orig_initial=q_orig_initial,
                        q_policy_turn_vectors=q_policy_turn_vectors,
                        q_profile_turn_vectors=q_profile_turn_vectors,
                        gt_video_vec=gt_video_vec,
                    )

                finally:
                    _update_progress_after_index(idx)
    finally:
        if pbar is not None:
            pbar.close()

    if not final_ranks and not policy_ranks_by_turn.get(int(args.max_turn), []):
        if len(test_rows) > 0:
            raise RuntimeError("No valid rows were evaluated.")
        logger.warning("No valid rows to summarize for this shard (empty assignment).")

    max_k = video_embeddings.shape[0]
    ks = [int(min(k, max_k)) for k in ks]
    rows_valid = len(policy_ranks_by_turn.get(int(args.max_turn), []))
    if rows_valid <= 0:
        rows_valid = len(final_ranks)

    policy_retrieval_by_turn: Dict[str, Dict[str, float]] = {}
    strict_retrieval_by_turn: Dict[str, Dict[str, float]] = {}
    policy_temporal_by_turn: Dict[str, Dict[str, float]] = {}
    strict_temporal_by_turn: Dict[str, Dict[str, float]] = {}
    for turn in range(1, int(args.max_turn) + 1):
        policy_key = f"policy@{turn}"
        strict_key = f"turn{turn}"
        policy_retrieval_by_turn[policy_key] = _metrics_from_ranks(policy_ranks_by_turn.get(turn, []), ks)
        strict_retrieval_by_turn[strict_key] = _metrics_from_ranks(strict_ranks_by_turn.get(turn, []), ks)
        policy_temporal_by_turn[policy_key] = _temporal_metrics(
            policy_iou_r1_by_turn.get(turn, []),
            policy_iou_matched_by_turn.get(turn, []),
            iou_thresholds,
        )
        strict_temporal_by_turn[strict_key] = _temporal_metrics(
            strict_iou_r1_by_turn.get(turn, []),
            strict_iou_matched_by_turn.get(turn, []),
            iou_thresholds,
        )

    turn1_policy = policy_retrieval_by_turn.get("policy@1", _metrics_from_ranks([], ks))
    final_policy = policy_retrieval_by_turn.get(f"policy@{int(args.max_turn)}", _metrics_from_ranks([], ks))
    turn1_temporal_policy = policy_temporal_by_turn.get("policy@1", _temporal_metrics([], [], iou_thresholds))
    final_temporal_policy = policy_temporal_by_turn.get(
        f"policy@{int(args.max_turn)}", _temporal_metrics([], [], iou_thresholds)
    )

    result = {
        "model_path": args.model_path,
        "verified_test_jsonl": args.verified_test_jsonl,
        "video_root": args.video_root,
        "video_meta_path": resolved_video_meta_path,
        "video_meta_stats": {
            "npy_or_npz_inputs": int(video_meta_stats.get("npy_or_npz_inputs", 0)),
            "meta_payload_hits": int(video_meta_stats.get("meta_payload_hits", 0)),
            "meta_payload_misses": int(video_meta_stats.get("meta_payload_misses", 0)),
        },
        "rows_total": int(len(test_rows)),
        "rows_total_before_limit": int(full_count),
        "rows_valid": int(rows_valid),
        "skip": {k: int(v) for k, v in skip.items()},
        "settings": {
            "temporal": bool(args.temporal),
            "max_turn": int(args.max_turn),
            "iou_thresholds": [float(x) for x in iou_thresholds],
            "use_vllm": bool(args.use_vllm),
            "vllm_model_path": vllm_runtime_model_path if bool(args.use_vllm) else "",
            "vllm_eval_batch_size": int(vllm_eval_batch_size) if bool(args.use_vllm) else 0,
            "use_vllm_batch_mode": bool(use_vllm_batch),
            "num_shards": int(args.num_shards),
            "shard_id": int(args.shard_id),
            "resume_skip_jsonl": _split_csv_paths(args.resume_skip_jsonl),
            "use_refine_gate": bool(args.use_refine_gate),
            "use_query_embedder_path": bool(args.use_query_embedder_path),
            "use_updated_query": bool(args.use_updated_query),
            "refine_rollout_depth": int(refine_rollout_depth),
            "query_embedder_input_prefix": str(args.query_embedder_input_prefix or ""),
            "qfinal_pooling": args.qfinal_pooling,
            "qfinal_normalize": bool(args.qfinal_normalize),
            "profile_rollout_depths": [int(x) for x in profile_rollout_depths],
            "save_query_vectors": bool(save_query_vectors_enabled),
            "save_query_vectors_dir": str(save_query_vectors_dir or ""),
            "save_query_vectors_only_refined": bool(save_query_vectors_only_refined),
            "save_query_vectors_max_gb": float(args.save_query_vectors_max_gb),
            "topk": ks,
            "max_new_tokens": int(args.max_new_tokens),
        },
        "query_vector_dump": {
            "enabled": bool(save_query_vectors_enabled),
            "dir": str(save_query_vectors_dir or ""),
            "saved_rows": int(query_vector_dump_saved),
            "failed_rows": int(query_vector_dump_failed),
            "skipped_filter_rows": int(query_vector_dump_skipped_filter),
            "skipped_cap_rows": int(query_vector_dump_skipped_cap),
            "saved_bytes": int(query_vector_dump_bytes),
        },
        "answer_counts": {k: int(v) for k, v in answer_counts.items()},
        "stop_reason_counts": {k: int(v) for k, v in stop_reason_counts.items()},
        "matched_by_turn": {k: int(v) for k, v in matched_by_turn.items()},
        "refine": {
            "samples_with_refine": int(refine_counters.get("samples_with_refine", 0)),
            "refine_token_generated_turns": int(refine_counters.get("refine_token_generated_turns", 0)),
            "refine_applied_turns": int(refine_counters.get("refine_applied_turns", 0)),
        },
        "retrieval": {
            "orig_query": _metrics_from_ranks(orig_ranks, ks),
            "turn1": turn1_policy,
            "final": final_policy,
            "delta_final_minus_turn1": {},
        },
        "retrieval_policy_by_turn": policy_retrieval_by_turn,
        "retrieval_strict_by_turn": strict_retrieval_by_turn,
        "temporal": {
            "turn1": turn1_temporal_policy,
            "final": final_temporal_policy,
            "delta_final_minus_turn1": {},
        },
        "temporal_policy_by_turn": policy_temporal_by_turn,
        "temporal_strict_by_turn": strict_temporal_by_turn,
    }

    for key in [f"R@{k}" for k in ks] + ["mrr", "mean_rank", "median_rank"]:
        if key in result["retrieval"]["turn1"] and key in result["retrieval"]["final"]:
            result["retrieval"]["delta_final_minus_turn1"][key] = float(
                result["retrieval"]["final"][key] - result["retrieval"]["turn1"][key]
            )

    for key in [f"IoU@{th:.1f}@R1" for th in iou_thresholds] + ["mIoU@R1", "mIoU_matched_only"]:
        if key in result["temporal"]["turn1"] and key in result["temporal"]["final"]:
            result["temporal"]["delta_final_minus_turn1"][key] = float(
                result["temporal"]["final"][key] - result["temporal"]["turn1"][key]
            )

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary json: {output_json}")

    if jsonl_writer is not None:
        jsonl_writer.close()
        logger.info(f"Saved detail jsonl: {output_jsonl} ({streamed_detail_rows} rows, streamed)")
    if save_query_vectors_enabled:
        logger.info(
            "Saved query vectors: dir=%s saved=%d failed=%d skipped_filter=%d skipped_cap=%d bytes=%d",
            save_query_vectors_dir,
            int(query_vector_dump_saved),
            int(query_vector_dump_failed),
            int(query_vector_dump_skipped_filter),
            int(query_vector_dump_skipped_cap),
            int(query_vector_dump_bytes),
        )

    logger.info(
        "Final summary | "
        f"Retrieval R@1: {result['retrieval']['final'].get('R@1', float('nan')):.4f} "
        f"(turn1 {result['retrieval']['turn1'].get('R@1', float('nan')):.4f}) | "
        f"Retrieval MRR: {result['retrieval']['final'].get('mrr', float('nan')):.4f} "
        f"(turn1 {result['retrieval']['turn1'].get('mrr', float('nan')):.4f}) | "
        f"Temporal IoU@0.5@R1: {result['temporal']['final'].get('IoU@0.5@R1', float('nan')):.4f} "
        f"(turn1 {result['temporal']['turn1'].get('IoU@0.5@R1', float('nan')):.4f}) | "
        f"Temporal mIoU@R1: {result['temporal']['final'].get('mIoU@R1', float('nan')):.4f} "
        f"(turn1 {result['temporal']['turn1'].get('mIoU@R1', float('nan')):.4f})"
    )


if __name__ == "__main__":
    main()
