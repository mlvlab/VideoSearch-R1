#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor


def _setup_import_paths() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3]
    videosearch_root = repo_root / "videosearch_r1"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(videosearch_root))


_setup_import_paths()

from model.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration  # noqa: E402
from model.qwen_vl_utils.vision_process import process_vision_info  # noqa: E402
from utils.video_metadata import (  # noqa: E402
    load_video_meta_index,
    resolve_video_meta_for_video_path,
)


logger = logging.getLogger("eval_verified_test_rerank")

SYSTEM_PROMPT_TEMPLATE = (
    "You are a video retrieval assistant. Your task is to analyze a retrieved video against the user query. "
    "Inside <think>...</think>, perform a step-by-step analysis comparing the query's requirements with the visual "
    "evidence in the video. Identify any missing or incorrect elements. Output the final decision strictly as "
    "<answer>matched</answer> or <answer>not_matched</answer>. If the decision is <answer>not_matched</answer>, "
    "you must append the special token(s) {refine_suffix} at the very end to initiate a latent query update. "
    "Do not invent details beyond what is visible and be concise."
)

ANSWER_RE = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def _build_refine_tokens(refine_token: str, refine_token_count: int) -> List[str]:
    token = str(refine_token or "<REFINE>").strip()
    if not token:
        token = "<REFINE>"
    count = int(refine_token_count)
    if count <= 1:
        return [token]
    if token.startswith("<") and token.endswith(">") and len(token) >= 3:
        inner = token[1:-1].strip()
        if inner:
            return [f"<{inner}_{idx}>" for idx in range(1, count + 1)]
    return [f"{token}_{idx}" for idx in range(1, count + 1)]


def _resolve_refine_tokens(tokenizer, refine_token: str, refine_token_count: int) -> List[str]:
    if int(refine_token_count) > 0:
        return _build_refine_tokens(refine_token, int(refine_token_count))

    base = str(refine_token or "<REFINE>").strip() or "<REFINE>"
    numbered: List[Tuple[int, str]] = []
    if base.startswith("<") and base.endswith(">") and len(base) >= 3:
        inner = re.escape(base[1:-1].strip())
        pat = re.compile(rf"^<{inner}_(\d+)>$")
        for tok in tokenizer.get_vocab().keys():
            m = pat.match(str(tok))
            if m:
                numbered.append((int(m.group(1)), str(tok)))
    if numbered:
        numbered.sort(key=lambda x: x[0])
        return [tok for _, tok in numbered]
    return [base]


def _make_system_prompt(refine_tokens: List[str]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(refine_suffix="".join(refine_tokens))


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


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _l2_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _compute_rank_from_scores(scores: torch.Tensor, pos_row: int) -> int:
    pos_score = scores[pos_row]
    return int((scores > pos_score).sum().item() + 1)


def _metrics_from_ranks(ranks: List[int], ks: List[int]) -> Dict[str, float]:
    arr = np.asarray(ranks, dtype=np.int64)
    out: Dict[str, float] = {
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


def _update_counts_from_detail_row(counts: Dict[str, int], row: dict) -> None:
    answer = str(row.get("answer", "")).strip().lower()
    had_refine = bool(row.get("refine_token_generated", False))
    used_refine = bool(row.get("used_refine", False))
    if answer == "matched":
        counts["matched"] += 1
    elif answer == "not_matched":
        counts["not_matched"] += 1
        if not had_refine:
            counts["not_matched_no_refine_token"] += 1
    else:
        counts["unknown_answer"] += 1
        if not had_refine:
            counts["unknown_no_refine_token"] += 1
    if used_refine:
        counts["refine_applied"] += 1
    if had_refine:
        counts["refine_token_generated"] += 1


def _invert_docid2row(docid2row: Dict[str, int]) -> List[str]:
    n = max(docid2row.values()) + 1
    row2doc = [""] * n
    for doc, row in docid2row.items():
        if 0 <= int(row) < n:
            row2doc[int(row)] = str(doc)
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
                model.query_embedder_head = torch.nn.Linear(
                    in_dim, out_dim, bias=(q_b is not None)
                )
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
                dropped.append(
                    f"{k}:shape_ckpt={tuple(v.shape)}!=model={tuple(target.shape)}"
                )
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
        # direct (id may already contain extension)
        cands.append(os.path.join(root, vid))
        # legacy npy subtree
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
    # key: exact query text, value: (row_idx, qid, pos_doc_id)
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


def _build_generation_batch(
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
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=image_patch_size,
    )
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=int(model_max_length),
        padding=True,
        **video_kwargs,
    )
    return batch


def _extract_update_from_ids(
    model,
    model_inputs: Dict[str, torch.Tensor],
    full_ids: torch.Tensor,
    prompt_len: int,
    completion_ids: torch.Tensor,
    refine_token_ids: List[int],
    use_refine_gate: bool,
) -> Tuple[Optional[torch.Tensor], bool]:
    # Detect refine tokens ONLY in generated completion (not in prompt/system text).
    comp_mask = torch.zeros_like(completion_ids[0], dtype=torch.bool)
    for tok_id in refine_token_ids:
        comp_mask |= completion_ids[0].eq(int(tok_id))
    comp_ref_pos = torch.nonzero(comp_mask, as_tuple=False).squeeze(1)
    had_refine = bool(comp_ref_pos.numel() > 0)
    if not had_refine:
        return None, False
    run_ids = full_ids

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
        hidden = outputs.hidden_states[-1]  # [1, T, H]

    pos_in_full = comp_ref_pos + int(prompt_len)
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
):
    if update.dim() == 1:
        update = update.unsqueeze(0)
    if update.dim() != 2:
        raise ValueError(f"update must be 1D/2D tensor, got shape={tuple(update.shape)}")

    if bool(use_query_embedder_path):
        if query_embedder_model is None or query_embedder_tokenizer is None:
            raise RuntimeError("use_query_embedder_path=True but query embedder is not initialized.")
        q_embedder = query_embedder_model
        q_embedder.eval()
        tokenized = query_embedder_tokenizer(
            [query_text],
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
        latent_token = update_for_llm.to(dtype=query_token_embeds.dtype).unsqueeze(0)
        inputs_embeds = torch.cat([query_token_embeds, latent_token], dim=1)
        latent_attn = torch.ones((1, int(update.size(0))), device=device, dtype=attention_mask.dtype)
        full_attention = torch.cat([attention_mask, latent_attn], dim=1)

        with torch.no_grad():
            outputs = q_embedder.language_model(
                input_ids=None,
                attention_mask=full_attention,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state.to(dtype=torch.float32)

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

    if bool(qfinal_normalize):
        q_final = F.normalize(q_final, p=2, dim=-1)
    return q_final, mode


def _default_output_path(model_path: str) -> str:
    ckpt_name = os.path.basename(os.path.normpath(model_path))
    parent = os.path.dirname(os.path.normpath(model_path))
    logs_dir = os.path.join(parent, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"verified_test_rerank_{ckpt_name}.json")


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
    # Last fallback.
    return "Qwen/Qwen3-VL-2B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Strict test evaluation with generated <answer> decision on top1 video. "
            "If not_matched, extract latent at <REFINE>, build q_final, and rerank."
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
        default="data/activitynet/test/video_npy_with_meta",
        help=(
            "Root for video files. Supports npy/npz and raw video "
            "(mp4/mkv/webm/avi/mov/m4v)."
        ),
    )
    p.add_argument(
        "--video_meta_path",
        type=str,
        default="data/activitynet/test/video_npy_with_meta/meta.jsonl",
        help="Optional video meta jsonl path. If empty, auto-detect from video_root.",
    )
    p.add_argument(
        "--query_embeddings_path",
        type=str,
        default="data/activitynet/test/query_embedding/query_embeddings.test.npy",
    )
    p.add_argument(
        "--query_meta_path",
        type=str,
        default="data/activitynet/test/query_embedding/query_meta.test.jsonl",
    )
    p.add_argument(
        "--video_embeddings_path",
        type=str,
        default="data/activitynet/test/video_embedding_1fps/segment_embeds.npy",
    )
    p.add_argument(
        "--video_docid2row_path",
        type=str,
        default="data/activitynet/test/video_embedding_1fps/docid2row.json",
    )
    p.add_argument("--refine_token", type=str, default="<REFINE>")
    p.add_argument(
        "--refine_token_count",
        type=int,
        default=0,
        help="Number of refine tokens. <=0 enables auto-detect from tokenizer vocab.",
    )
    p.add_argument("--use_refine_gate", type=_str2bool, default=True)
    p.add_argument("--use_query_embedder_path", type=_str2bool, default=True)
    p.add_argument("--query_embedder_model_path", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    p.add_argument("--qfinal_pooling", type=str, default="latent_last", choices=["latent_last", "mean"])
    p.add_argument("--qfinal_normalize", type=_str2bool, default=True)
    p.add_argument("--query_embedder_max_length", type=int, default=128)
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--topk", type=str, default="1,5,10,100")
    p.add_argument(
        "--rerank_topk",
        type=str,
        default="all",
        help="Refine rerank scope: 'all' (default) or positive integer (e.g., 10).",
    )
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--eval_limit_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", type=_str2bool, default=True)
    p.add_argument("--image_patch_size", type=int, default=16)
    p.add_argument("--video_min_pixels", type=int, default=128 * 28 * 28)
    p.add_argument("--video_max_pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--video_total_pixels", type=int, default=115200 * 28 * 28)
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--output_jsonl", type=str, default="")
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
    ks = sorted({max(1, int(x.strip())) for x in str(args.topk).split(",") if x.strip()})
    output_json = args.output_json.strip() or _default_output_path(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load test rows.
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
    if not test_rows:
        raise RuntimeError("No test rows loaded.")
    logger.info(f"Loaded verified test rows: {len(test_rows)}")

    # Retrieval resources.
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
    logger.info(
        f"Rerank mode: {'all' if rerank_topk is None else f'top{rerank_topk}'}"
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

    # Load VLM model.
    processor_path = _resolve_processor_path(args.model_path, args.processor_path)
    logger.info(f"Using processor path: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, padding_side="left")
    refine_tokens = _resolve_refine_tokens(
        tokenizer=processor.tokenizer,
        refine_token=args.refine_token,
        refine_token_count=int(args.refine_token_count),
    )
    missing_tokens = [tok for tok in refine_tokens if tok not in processor.tokenizer.get_vocab()]
    if missing_tokens:
        processor.tokenizer.add_tokens(missing_tokens, special_tokens=True)
        logger.info(f"Added refine tokens: {missing_tokens}")
    refine_token_ids = [processor.tokenizer.convert_tokens_to_ids(tok) for tok in refine_tokens]
    logger.info(
        f"Refine tokens resolved: tokens={refine_tokens}, ids={refine_token_ids}, "
        f"requested_count={args.refine_token_count}"
    )
    system_prompt = _make_system_prompt(refine_tokens)

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
    created_append_in = False
    if not hasattr(model, "refine_latent_input_projector"):
        model.refine_latent_input_projector = torch.nn.Sequential(
            torch.nn.LayerNorm(retrieval_dim),
            torch.nn.Linear(retrieval_dim, retrieval_dim),
        )
        created_roll_in = True
    if not hasattr(model, "refine_append_input_projector"):
        model.refine_append_input_projector = torch.nn.Sequential(
            torch.nn.LayerNorm(retrieval_dim),
            torch.nn.Linear(retrieval_dim, retrieval_dim),
        )
        created_append_in = True
    with torch.no_grad():
        if created_roll_in:
            torch.nn.init.eye_(model.refine_latent_input_projector[1].weight)
            torch.nn.init.zeros_(model.refine_latent_input_projector[1].bias)
        if created_append_in:
            torch.nn.init.eye_(model.refine_append_input_projector[1].weight)
            torch.nn.init.zeros_(model.refine_append_input_projector[1].bias)
    logger.info(
        f"Refine module init: hidden_size={hidden_size} (source={hidden_size_src}), "
        f"retrieval_dim={retrieval_dim}"
    )
    loaded_refine = _maybe_load_refine_weights(model, args.model_path)
    if not loaded_refine and hasattr(model, "refine_latent_input_projector"):
        # Backward-compat: old checkpoints without this module should keep legacy behavior.
        model.refine_latent_input_projector = None
    if not loaded_refine and hasattr(model, "refine_append_input_projector"):
        model.refine_append_input_projector = None
    model.to(device)
    model.eval()

    # Query embedder branch.
    query_embedder_model = None
    query_embedder_tokenizer = None
    if bool(args.use_query_embedder_path):
        logger.info(f"Loading query embedder: {args.query_embedder_model_path}")
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
        query_hidden, query_hidden_src = _resolve_text_hidden_size(
            query_embedder_model, default=retrieval_dim
        )
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

    # Iterate.
    skip = {
        "query_not_found_in_meta": 0,
        "pos_doc_missing": 0,
        "video_file_missing": 0,
    }
    counts = {
        "matched": 0,
        "not_matched": 0,
        "unknown_answer": 0,
        "refine_applied": 0,
        "refine_token_generated": 0,
        "not_matched_no_refine_token": 0,
        "unknown_no_refine_token": 0,
    }

    orig_ranks: List[int] = []
    final_ranks: List[int] = []
    refined_only_ranks: List[int] = []
    streamed_detail_rows = 0
    done_indices: set[int] = set()
    output_jsonl = str(args.output_jsonl or "").strip()
    jsonl_writer = None
    if output_jsonl:
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
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
                if isinstance(row_obj.get("orig_rank"), int):
                    orig_ranks.append(int(row_obj["orig_rank"]))
                if isinstance(row_obj.get("final_rank"), int):
                    final_ranks.append(int(row_obj["final_rank"]))
                    if bool(row_obj.get("used_refine", False)):
                        refined_only_ranks.append(int(row_obj["final_rank"]))
                _update_counts_from_detail_row(counts, row_obj)
            streamed_detail_rows = len(existing_by_index)
            logger.info(
                f"Resume enabled from existing jsonl: {output_jsonl} "
                f"(rows={streamed_detail_rows}, bad_lines={bad_lines})"
            )
            if os.path.exists(output_json):
                try:
                    with open(output_json, "r", encoding="utf-8") as f:
                        old_summary = json.load(f)
                    old_skip = old_summary.get("skip", {})
                    for k in skip.keys():
                        if isinstance(old_skip.get(k), int):
                            skip[k] = int(old_skip[k])
                except Exception as exc:
                    logger.warning(f"Failed to load previous summary for skip counts: {exc}")
            jsonl_writer = open(output_jsonl, "a", encoding="utf-8", buffering=1)
        else:
            # Line-buffered writer so each sample is visible immediately via tail -f.
            jsonl_writer = open(output_jsonl, "w", encoding="utf-8", buffering=1)
        logger.info(f"Streaming detail jsonl: {output_jsonl}")
    loop_start = time.perf_counter()
    progress_interval = max(1, len(test_rows) // 20)

    for idx, row in enumerate(test_rows, start=1):
        row_index = idx - 1
        if row_index in done_indices:
            if idx % progress_interval == 0 or idx == len(test_rows):
                elapsed = time.perf_counter() - loop_start
                done = idx
                rate = done / max(elapsed, 1e-9)
                remain = len(test_rows) - done
                eta_sec = remain / max(rate, 1e-9)
                logger.info(
                    "progress "
                    f"{done}/{len(test_rows)} ({100.0 * done / max(1, len(test_rows)):.1f}%) | "
                    f"matched={counts['matched']} not_matched={counts['not_matched']} "
                    f"unknown={counts['unknown_answer']} | "
                    f"elapsed={elapsed:.1f}s eta={eta_sec:.1f}s"
                )
            continue

        query_text = str(row.get("fig_desc", "")).strip()
        if not query_text:
            skip["query_not_found_in_meta"] += 1
            continue
        meta_item = query_map.get(query_text)
        if meta_item is None:
            skip["query_not_found_in_meta"] += 1
            continue
        q_row, qid, pos_doc_id = meta_item
        pos_row = docid2row.get(str(pos_doc_id))
        if pos_row is None:
            skip["pos_doc_missing"] += 1
            continue

        q_orig = torch.from_numpy(query_embeddings[int(q_row)]).to(device=device, dtype=torch.float32)
        # First retrieval (top1) from original query embedding.
        sims_orig = torch.mv(video_emb_t, q_orig)
        orig_top1_row = int(torch.argmax(sims_orig).item())
        orig_top1_doc = row2doc[orig_top1_row]
        orig_rank = _compute_rank_from_scores(sims_orig, int(pos_row))
        orig_ranks.append(orig_rank)

        video_path = _resolve_video_path(args.video_root, orig_top1_doc)
        if video_path is None:
            skip["video_file_missing"] += 1
            continue
        video_meta = None
        if str(video_path).lower().endswith((".npy", ".npz")):
            video_meta = resolve_video_meta_for_video_path(video_path, video_meta_index)

        # Generate answer with top1 + query.
        batch = _build_generation_batch(
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
        model_inputs = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                model_inputs[k] = v.to(device, non_blocking=True)
            else:
                model_inputs[k] = v

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
            )
        prompt_len = int(model_inputs["input_ids"].shape[1])
        completion_ids = generated_ids[:, prompt_len:]
        completion_text = processor.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        answer = _parse_answer(completion_text)

        final_rank = orig_rank
        final_top1_doc = orig_top1_doc
        qfinal_mode = "original_top1_accept"
        used_refine = False
        had_refine_token = False
        orig_top1_rank_after_final = 1
        neg_push_down = 0

        if answer == "matched":
            counts["matched"] += 1
        else:
            if answer == "not_matched":
                counts["not_matched"] += 1
            else:
                counts["unknown_answer"] += 1
            # Strict mode: apply refine only when model actually generated <REFINE>.
            update, had_refine_token = _extract_update_from_ids(
                model=model,
                model_inputs=model_inputs,
                full_ids=generated_ids,
                prompt_len=prompt_len,
                completion_ids=completion_ids,
                refine_token_ids=refine_token_ids,
                use_refine_gate=bool(args.use_refine_gate),
            )
            if had_refine_token:
                used_refine = True
                counts["refine_applied"] += 1
                counts["refine_token_generated"] += 1
                q_final, qfinal_mode = _build_q_final(
                    q_orig=q_orig,
                    update=update,
                    query_text=query_text,
                    use_query_embedder_path=bool(args.use_query_embedder_path),
                    query_embedder_model=query_embedder_model,
                    query_embedder_tokenizer=query_embedder_tokenizer,
                    query_embedder_head=getattr(model, "query_embedder_head", None),
                    append_input_projector=getattr(model, "refine_append_input_projector", None),
                    latent_input_projector=getattr(model, "refine_latent_input_projector", None),
                    qfinal_pooling=args.qfinal_pooling,
                    qfinal_normalize=bool(args.qfinal_normalize),
                    query_embedder_max_length=int(args.query_embedder_max_length),
                )
                if rerank_topk is None:
                    sims_final = torch.mv(video_emb_t, q_final)
                    final_top1_row = int(torch.argmax(sims_final).item())
                    final_top1_doc = row2doc[final_top1_row]
                    final_rank = _compute_rank_from_scores(sims_final, int(pos_row))
                    orig_top1_rank_after_final = _compute_rank_from_scores(sims_final, int(orig_top1_row))
                    neg_push_down = int(orig_top1_rank_after_final - 1)
                else:
                    k = int(rerank_topk)
                    topk_rows = torch.topk(
                        sims_orig, k=k, largest=True, sorted=True
                    ).indices
                    topk_emb = video_emb_t.index_select(0, topk_rows)
                    topk_scores_final = torch.mv(topk_emb, q_final)
                    topk_order = torch.argsort(topk_scores_final, descending=True)
                    reranked_rows = topk_rows.index_select(0, topk_order)

                    final_top1_row = int(reranked_rows[0].item())
                    final_top1_doc = row2doc[final_top1_row]

                    if int(orig_rank) > k:
                        final_rank = int(orig_rank)
                    else:
                        pos_in_head = torch.nonzero(
                            reranked_rows.eq(int(pos_row)), as_tuple=False
                        )
                        if pos_in_head.numel() > 0:
                            final_rank = int(pos_in_head[0].item() + 1)
                        else:
                            final_rank = int(orig_rank)

                    orig_top1_in_head = torch.nonzero(
                        reranked_rows.eq(int(orig_top1_row)), as_tuple=False
                    )
                    if orig_top1_in_head.numel() > 0:
                        orig_top1_rank_after_final = int(orig_top1_in_head[0].item() + 1)
                    else:
                        orig_top1_rank_after_final = 1
                    neg_push_down = int(orig_top1_rank_after_final - 1)
                    qfinal_mode = f"{qfinal_mode}_top{k}"
                refined_only_ranks.append(final_rank)
            else:
                if answer == "not_matched":
                    counts["not_matched_no_refine_token"] += 1
                    qfinal_mode = "not_matched_no_refine_token"
                else:
                    counts["unknown_no_refine_token"] += 1
                    qfinal_mode = "unknown_no_refine_token"

        final_ranks.append(final_rank)

        if jsonl_writer is not None:
            detail_row = {
                "index": idx - 1,
                "qid": qid,
                "query": query_text,
                "gt_pos_doc_id": pos_doc_id,
                "orig_top1_doc_id": orig_top1_doc,
                "orig_top1_is_negative": bool(orig_top1_doc != str(pos_doc_id)),
                "orig_rank": int(orig_rank),
                "answer": answer,
                "raw_output": completion_text,
                "used_refine": bool(used_refine),
                "qfinal_mode": qfinal_mode,
                "refine_token_generated": bool(had_refine_token),
                "orig_top1_rank_after_final": int(orig_top1_rank_after_final),
                "orig_top1_push_down": int(neg_push_down),
                "final_top1_doc_id": final_top1_doc,
                "final_rank": int(final_rank),
                "rerank_topk": "all" if rerank_topk is None else int(rerank_topk),
            }
            jsonl_writer.write(json.dumps(detail_row, ensure_ascii=False) + "\n")
            jsonl_writer.flush()
            streamed_detail_rows += 1

        if idx % progress_interval == 0 or idx == len(test_rows):
            elapsed = time.perf_counter() - loop_start
            done = idx
            rate = done / max(elapsed, 1e-9)
            remain = len(test_rows) - done
            eta_sec = remain / max(rate, 1e-9)
            logger.info(
                "progress "
                f"{done}/{len(test_rows)} ({100.0 * done / max(1, len(test_rows)):.1f}%) | "
                f"matched={counts['matched']} not_matched={counts['not_matched']} "
                f"unknown={counts['unknown_answer']} | "
                f"elapsed={elapsed:.1f}s eta={eta_sec:.1f}s"
            )

    if not final_ranks:
        raise RuntimeError("No valid rows were evaluated.")

    max_k = video_embeddings.shape[0]
    ks = [int(min(k, max_k)) for k in ks]

    result = {
        "model_path": args.model_path,
        "verified_test_jsonl": args.verified_test_jsonl,
        "video_root": args.video_root,
        "video_meta_path": resolved_video_meta_path,
        "rerank_topk": "all" if rerank_topk is None else int(rerank_topk),
        "rows_total": int(len(test_rows)),
        "rows_total_before_limit": int(full_count),
        "rows_valid": int(len(final_ranks)),
        "skip": {k: int(v) for k, v in skip.items()},
        "answer_counts": {k: int(v) for k, v in counts.items()},
        "settings": {
            "use_refine_gate": bool(args.use_refine_gate),
            "use_query_embedder_path": bool(args.use_query_embedder_path),
            "qfinal_pooling": args.qfinal_pooling,
            "qfinal_normalize": bool(args.qfinal_normalize),
            "rerank_topk": "all" if rerank_topk is None else int(rerank_topk),
            "topk": ks,
            "max_new_tokens": int(args.max_new_tokens),
        },
        "original": _metrics_from_ranks(orig_ranks, ks),
        "final": _metrics_from_ranks(final_ranks, ks),
        "refined_only": _metrics_from_ranks(refined_only_ranks, ks) if refined_only_ranks else {},
        "delta_final_minus_original": {},
    }
    for k in [f"R@{kk}" for kk in ks] + ["mrr", "mean_rank", "median_rank"]:
        if k in result["original"] and k in result["final"]:
            result["delta_final_minus_original"][k] = float(result["final"][k] - result["original"][k])

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary json: {output_json}")

    if jsonl_writer is not None:
        jsonl_writer.close()
        logger.info(f"Saved detail jsonl: {output_jsonl} ({streamed_detail_rows} rows, streamed)")

    logger.info(
        "Final summary | "
        f"R@1: {result['final'].get('R@1', float('nan')):.4f} "
        f"(orig {result['original'].get('R@1', float('nan')):.4f}) | "
        f"R@10: {result['final'].get('R@10', float('nan')):.4f} "
        f"(orig {result['original'].get('R@10', float('nan')):.4f}) | "
        f"MRR: {result['final'].get('mrr', float('nan')):.4f} "
        f"(orig {result['original'].get('mrr', float('nan')):.4f})"
    )


if __name__ == "__main__":
    main()
