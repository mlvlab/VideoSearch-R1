#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from transformers import AutoProcessor


def _setup_import_paths() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3]
    videosearch_root = repo_root / "videosearch_r1"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(videosearch_root))


_setup_import_paths()

from model.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration  # noqa: E402
from trainer.sft_share_gpt_trainer import ShareGPTSFTCollator  # noqa: E402
from utils.data_sft_share_gpt import ShareGPTDataArguments, build_sharegpt_dataset  # noqa: E402


logger = logging.getLogger("eval_refine_recall")


def _default_activitynet_path(*parts: str) -> str:
    return os.path.join(os.environ.get("VIDEOSEARCH_DATA_ROOT", "./data"), "activitynet", *parts)


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


def _load_query_meta_maps(path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    qid_to_row: Dict[str, int] = {}
    qid_to_pos: Dict[str, str] = {}
    for row_idx, row in enumerate(_iter_jsonl(path)):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        qid_to_row[qid] = row_idx
        pos_doc = str(row.get("pos_doc_id", "")).strip()
        if pos_doc:
            qid_to_pos[qid] = pos_doc
    return qid_to_row, qid_to_pos


def _is_negative_sample(sample: dict) -> bool:
    if not isinstance(sample, dict):
        return False
    response = str(sample.get("response", ""))
    if "<answer>not_matched</answer>" in response:
        return True
    meta = sample.get("meta", {})
    if isinstance(meta, dict):
        gold_label = str(meta.get("gold_label", "")).strip().lower()
        if gold_label in {"neg", "negative", "not_matched", "not matched", "mismatch"}:
            return True
    return False


def _filter_only_neg_dataset(dataset):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        source_indices = list(dataset.indices)
    else:
        base_dataset = dataset
        source_indices = list(range(len(dataset)))

    keep_indices = []
    for idx in source_indices:
        sample = base_dataset[idx]
        if _is_negative_sample(sample):
            keep_indices.append(idx)

    return Subset(base_dataset, keep_indices), len(source_indices), len(keep_indices)


def _maybe_load_refine_weights(model: torch.nn.Module, model_path: str) -> bool:
    if not os.path.isdir(model_path):
        return False
    has_refine = hasattr(model, "refine_projector") and hasattr(model, "refine_gate")
    if not has_refine:
        return False

    prefix = ("refine_projector.", "refine_gate.")
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
    logger.info("No refine weights found in checkpoint; using initialized refine modules.")
    return False


def _l2_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _compute_rank_metrics(
    q_mat: np.ndarray,
    video_mat: np.ndarray,
    pos_rows: np.ndarray,
    topk_max: int,
) -> Dict[str, float]:
    sims = np.matmul(q_mat, video_mat.T)
    pos_scores = sims[np.arange(sims.shape[0]), pos_rows]
    ranks = (sims > pos_scores[:, None]).sum(axis=1) + 1

    topk_max = int(max(1, min(topk_max, video_mat.shape[0])))
    out: Dict[str, float] = {
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    for k in range(1, topk_max + 1):
        out[f"R@{k}"] = float(np.mean(ranks <= k))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Compare retrieval R@1..100 for q_orig vs q_orig+delta.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset_info", type=str, default="sft_data/data_config.yaml")
    p.add_argument("--dataset_name", type=str, default="ActivityNet-Oneturn")
    p.add_argument("--model_max_length", type=int, default=46384)
    p.add_argument("--image_min_pixels", type=int, default=3136)
    p.add_argument("--image_max_pixels", type=int, default=12845056)
    p.add_argument("--video_min_pixels", type=int, default=100352)
    p.add_argument("--video_max_pixels", type=int, default=602112)
    p.add_argument("--video_total_pixels", type=int, default=90316800)
    p.add_argument("--max_frames", type=int, default=40)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--eval_split_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only_neg", type=_str2bool, default=True)
    p.add_argument("--refine_token", type=str, default="<REFINE>")
    p.add_argument(
        "--refine_token_count",
        type=int,
        default=0,
        help="Number of refine tokens. <=0 enables auto-detect from tokenizer vocab.",
    )
    p.add_argument("--use_refine_gate", type=_str2bool, default=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--topk_max", type=int, default=100)
    p.add_argument("--loss_type", type=str, default="all_assistant")
    p.add_argument("--bf16", type=_str2bool, default=True)

    p.add_argument(
        "--query_embeddings_path",
        type=str,
        default=_default_activitynet_path("train", "query_embedding", "query_embeddings.train.npy"),
    )
    p.add_argument(
        "--query_meta_path",
        type=str,
        default=_default_activitynet_path("train", "query_embedding", "query_meta.train.jsonl"),
    )
    p.add_argument(
        "--video_embeddings_path",
        type=str,
        default=_default_activitynet_path("train", "video_embedding_1fps", "segment_embeds.npy"),
    )
    p.add_argument(
        "--video_docid2row_path",
        type=str,
        default=_default_activitynet_path("train", "video_embedding_1fps", "docid2row.json"),
    )
    p.add_argument("--output_json", type=str, default="")
    return p.parse_args()


def _default_output_path(model_path: str) -> str:
    ckpt_name = os.path.basename(os.path.normpath(model_path))
    parent = os.path.dirname(os.path.normpath(model_path))
    logs_dir = os.path.join(parent, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"refine_recall_{ckpt_name}.json")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(args.model_path)
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
    hidden_size, hidden_size_src = _resolve_text_hidden_size(model, default=2048)
    model.refine_projector = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 2048),
        torch.nn.LayerNorm(2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 2048),
    )
    model.refine_gate = torch.nn.Linear(hidden_size, 1)
    logger.info(f"Refine module init: hidden_size={hidden_size} (source={hidden_size_src})")
    _maybe_load_refine_weights(model, args.model_path)
    model.eval()
    model.to(device)

    data_args = ShareGPTDataArguments(
        dataset_info=args.dataset_info,
        dataset_name=[args.dataset_name],
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_total_pixels=args.video_total_pixels,
        max_frames=args.max_frames,
        fps=args.fps,
    )
    full_dataset = build_sharegpt_dataset(data_args)
    if bool(args.only_neg):
        full_dataset, before_n, after_n = _filter_only_neg_dataset(full_dataset)
        if after_n == 0:
            raise ValueError("only_neg=True but no negative samples were found.")
        logger.info(f"Applied only_neg filter: {before_n} -> {after_n}")

    total = len(full_dataset)
    if total < 2:
        raise ValueError("Dataset too small for 8:2 split.")
    ratio = float(max(0.01, min(0.99, args.eval_split_ratio)))
    eval_size = max(1, int(round(total * ratio)))
    train_size = total - eval_size
    if train_size <= 0:
        train_size = total - 1
        eval_size = 1
    split_gen = torch.Generator().manual_seed(int(args.seed))
    _, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=split_gen)
    logger.info(
        f"Eval split: total={total} train={train_size} eval={eval_size} "
        f"(ratio={ratio:.3f})"
    )

    collator = ShareGPTSFTCollator(
        processor=processor,
        loss_type=args.loss_type,
        max_length=args.model_max_length,
        image_patch_size=16,
        use_video_metadata=True,
    )

    def safe_collate(examples):
        return collator([copy.deepcopy(ex) for ex in examples])

    loader = DataLoader(
        eval_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=safe_collate,
        pin_memory=(device.type == "cuda"),
    )

    query_embeddings = np.load(args.query_embeddings_path, mmap_mode="r")
    qid_to_query_row, qid_to_pos_video = _load_query_meta_maps(args.query_meta_path)
    video_embeddings = np.asarray(np.load(args.video_embeddings_path), dtype=np.float32)
    video_embeddings = _l2_norm_rows(video_embeddings)
    with open(args.video_docid2row_path, "r", encoding="utf-8") as f:
        video_id_to_row = {str(k): int(v) for k, v in json.load(f).items()}

    orig_vecs: List[np.ndarray] = []
    refine_vecs: List[np.ndarray] = []
    pos_rows: List[int] = []
    used_qids: List[str] = []

    skip = {
        "qid_missing": 0,
        "pos_missing": 0,
        "refine_token_missing": 0,
    }

    with torch.no_grad():
        for step, batch in enumerate(loader):
            qids = batch.pop("qids", [])
            _ = batch.pop("metas", None)

            model_inputs = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    model_inputs[k] = v.to(device, non_blocking=True)
                else:
                    model_inputs[k] = v
            model_inputs["output_hidden_states"] = True

            outputs = model(**model_inputs)
            input_ids = model_inputs["input_ids"]
            last_hidden = outputs.hidden_states[-1]
            refine_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tok_id in refine_token_ids:
                refine_mask |= input_ids.eq(int(tok_id))

            bs = input_ids.size(0)
            for row in range(bs):
                qid = str(qids[row]).strip() if row < len(qids) else ""
                q_row = qid_to_query_row.get(qid)
                if q_row is None:
                    skip["qid_missing"] += 1
                    continue
                pos_vid = str(qid_to_pos_video.get(qid, "")).strip()
                pos_row = video_id_to_row.get(pos_vid)
                if pos_row is None:
                    skip["pos_missing"] += 1
                    continue
                pos_list = torch.nonzero(refine_mask[row], as_tuple=False).squeeze(1)
                if pos_list.numel() == 0:
                    skip["refine_token_missing"] += 1
                    continue

                try:
                    projector_dtype = next(model.refine_projector.parameters()).dtype
                except StopIteration:
                    projector_dtype = last_hidden.dtype
                h_ref = last_hidden[row, pos_list]
                delta = model.refine_projector(h_ref.to(dtype=projector_dtype)).to(dtype=torch.float32)
                if bool(args.use_refine_gate):
                    try:
                        gate_dtype = next(model.refine_gate.parameters()).dtype
                    except StopIteration:
                        gate_dtype = projector_dtype
                    alpha = torch.sigmoid(model.refine_gate(h_ref.to(dtype=gate_dtype))).to(dtype=torch.float32)
                    update = delta * alpha
                else:
                    update = delta

                q_orig = np.asarray(query_embeddings[q_row], dtype=np.float32)
                q_orig = q_orig / max(float(np.linalg.norm(q_orig)), 1e-12)
                q_orig_t = torch.from_numpy(q_orig).to(device=device, dtype=torch.float32)
                q_final_t = F.normalize(q_orig_t + update.mean(dim=0), p=2, dim=-1)

                orig_vecs.append(q_orig.astype(np.float32))
                refine_vecs.append(q_final_t.detach().cpu().numpy().astype(np.float32))
                pos_rows.append(int(pos_row))
                used_qids.append(qid)

            if (step + 1) % 20 == 0:
                logger.info(f"Processed {step + 1} eval batches...")

    if not orig_vecs:
        raise RuntimeError("No valid eval rows found for recall evaluation.")

    q_orig_mat = _l2_norm_rows(np.stack(orig_vecs, axis=0).astype(np.float32))
    q_refine_mat = _l2_norm_rows(np.stack(refine_vecs, axis=0).astype(np.float32))
    pos_rows_np = np.asarray(pos_rows, dtype=np.int64)

    orig_metrics = _compute_rank_metrics(
        q_mat=q_orig_mat,
        video_mat=video_embeddings,
        pos_rows=pos_rows_np,
        topk_max=args.topk_max,
    )
    refine_metrics = _compute_rank_metrics(
        q_mat=q_refine_mat,
        video_mat=video_embeddings,
        pos_rows=pos_rows_np,
        topk_max=args.topk_max,
    )

    topk_max = int(max(1, min(args.topk_max, video_embeddings.shape[0])))
    improve = {
        f"R@{k}": float(refine_metrics[f"R@{k}"] - orig_metrics[f"R@{k}"])
        for k in range(1, topk_max + 1)
    }
    improve["mrr"] = float(refine_metrics["mrr"] - orig_metrics["mrr"])
    improve["mean_rank"] = float(refine_metrics["mean_rank"] - orig_metrics["mean_rank"])
    improve["median_rank"] = float(refine_metrics["median_rank"] - orig_metrics["median_rank"])

    out = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "only_neg": bool(args.only_neg),
        "use_refine_gate": bool(args.use_refine_gate),
        "refine_tokens": refine_tokens,
        "topk_max": topk_max,
        "counts": {
            "eval_total_rows": int(len(eval_dataset)),
            "valid_rows": int(len(orig_vecs)),
            "skip_qid_missing": int(skip["qid_missing"]),
            "skip_pos_missing": int(skip["pos_missing"]),
            "skip_refine_token_missing": int(skip["refine_token_missing"]),
        },
        "original": orig_metrics,
        "refined": refine_metrics,
        "delta": improve,
    }

    out_path = args.output_json.strip() or _default_output_path(args.model_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved recall comparison to {out_path}")
    logger.info(
        "Summary: "
        f"R@1 orig={orig_metrics['R@1']:.4f} -> refined={refine_metrics['R@1']:.4f}, "
        f"R@10 orig={orig_metrics['R@10']:.4f} -> refined={refine_metrics['R@10']:.4f}, "
        f"R@100 orig={orig_metrics[f'R@{topk_max}']:.4f} -> refined={refine_metrics[f'R@{topk_max}']:.4f}"
    )


if __name__ == "__main__":
    main()
