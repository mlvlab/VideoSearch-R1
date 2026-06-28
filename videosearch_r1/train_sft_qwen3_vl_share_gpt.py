# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import datetime
import logging
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import Subset, random_split
from transformers import AutoProcessor, HfArgumentParser, TrainerCallback
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import SFTTrainer, SFTConfig

from model.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration
from model.monkey_patch import apply_qwen3_vl_monkey_patch
from trainer.sft_share_gpt_trainer import ShareGPTSFTCollator
from trainer.sft_soft_refine_trainer import SoftRefineTrainer, EvalDetailLoggerCallback
from utils.arguments import ModelArguments
from utils.data_sft_share_gpt import ShareGPTDataArguments, build_sharegpt_dataset

logger = logging.getLogger("train_sft_sharegpt")
_DATA_ROOT = os.environ.get("VIDEOSEARCH_DATA_ROOT", "data")
_ACTIVITYNET_ROOT = os.path.join(_DATA_ROOT, "activitynet")


@dataclass
class SoftRefineArguments:
    enable_retrieval_optimization: bool = field(
        default=False,
        metadata={"help": "Enable retrieval optimization loss for <REFINE> samples."},
    )
    refine_token: str = field(
        default="<REFINE>",
        metadata={"help": "Special token that triggers retrieval refinement loss."},
    )
    refine_token_count: int = field(
        default=1,
        metadata={
            "help": (
                "Refine latent rollout depth. Special token registration remains single "
                "(refine_token only); values >1 enable deeper latent update rollout."
            )
        },
    )
    retrieval_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for retrieval InfoNCE loss."},
    )
    retrieval_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature for retrieval InfoNCE loss."},
    )
    negative_pool_size: int = field(
        default=32,
        metadata={"help": "Number of negative videos per sample for retrieval loss."},
    )
    retrieval_ignore_ambiguous_negatives: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, remove near-positive / false-negative candidates from "
                "retrieval loss using q_orig similarity margin."
            )
        },
    )
    ambiguous_negative_margin: float = field(
        default=0.02,
        metadata={
            "help": (
                "Treat negatives with q_orig similarity within this margin of the "
                "positive as ambiguous and exclude them from retrieval loss."
            )
        },
    )
    strict_negative_topk: int = field(
        default=0,
        metadata={
            "help": (
                "If >0, keep only the top-K strict negatives by q_final similarity "
                "for retrieval loss (0 keeps all strict negatives)."
            )
        },
    )
    projector_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for refine projector/gate."},
    )
    use_refine_gate: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use q_final=normalize(q_orig + alpha*delta). "
                "If False, disable gate and use q_final=normalize(q_orig + delta)."
            )
        },
    )
    zero_init_refine: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, zero-init projector last layer and gate (only when refine "
                "weights are not loaded from checkpoint)."
            )
        },
    )
    retrieval_on_eval: bool = field(
        default=False,
        metadata={"help": "If True, also apply retrieval loss during eval."},
    )
    only_neg: bool = field(
        default=False,
        metadata={"help": "If True, keep only not_matched/negative samples in dataset."},
    )
    eval_split_ratio: float = field(
        default=0.2,
        metadata={"help": "Eval split ratio when eval strategy is not 'no'."},
    )
    eval_recall_on_eval: bool = field(
        default=False,
        metadata={"help": "If True, compute retrieval recall (R@k) during each trainer.evaluate()."},
    )
    eval_recall_ks: str = field(
        default="1,5,10",
        metadata={"help": "Comma-separated K list for eval recall metrics. Example: 1,5,10"},
    )
    eval_detail_dir: str = field(
        default="logs",
        metadata={"help": "Relative directory under output_dir to write eval detail jsonl logs."},
    )
    query_embeddings_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "train/query_embedding/query_embeddings.train.npy"),
        metadata={"help": "Precomputed query embeddings npy path."},
    )
    query_meta_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "train/query_embedding/query_meta.train.jsonl"),
        metadata={"help": "Query meta jsonl path (qid -> row / pos_doc_id)."},
    )
    video_embeddings_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "train/video_embedding_1fps/segment_embeds.npy"),
        metadata={"help": "Precomputed video embeddings npy path."},
    )
    video_docid2row_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "train/video_embedding_1fps/docid2row.json"),
        metadata={"help": "video doc_id -> row index map json path."},
    )
    hard_negatives_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "train/hard_negatives.json"),
        metadata={"help": "qid -> hard negatives json path."},
    )
    hard_negative_refresh_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Refresh hard negatives online every N optimizer steps during training "
                "(0 disables online refresh, 1 refreshes every step)."
            )
        },
    )
    use_query_embedder_path: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, build q_final with a separate embedding model from "
                "concat([query_tokens, latent_token])."
            )
        },
    )
    query_embedder_model_path: str = field(
        default=os.environ.get("VIDEOSEARCH_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"),
        metadata={"help": "Model path for query embedder branch."},
    )
    qfinal_pooling: str = field(
        default="latent_last",
        metadata={"help": "Pooling for query-embedder branch: latent_last|mean"},
    )
    qfinal_normalize: bool = field(
        default=True,
        metadata={"help": "If True, L2-normalize q_final before retrieval logits."},
    )
    tune_query_embedder: bool = field(
        default=False,
        metadata={"help": "If True, unfreeze and train query embedder backbone."},
    )
    query_embedder_lr: float = field(
        default=5e-6,
        metadata={"help": "Learning rate for query embedder when tune_query_embedder=True."},
    )
    query_embedder_max_length: int = field(
        default=128,
        metadata={"help": "Max token length for query text in query embedder branch."},
    )
    eval_only: bool = field(
        default=False,
        metadata={"help": "Run evaluation only (no training)."},
    )
    external_eval_on_gpu0: bool = field(
        default=False,
        metadata={"help": "Launch eval-only subprocess on a dedicated GPU at each save step."},
    )
    external_eval_gpu: str = field(
        default="0",
        metadata={"help": "CUDA_VISIBLE_DEVICES value for external eval subprocess."},
    )
    external_eval_report_to: str = field(
        default="none",
        metadata={"help": "report_to value for external eval subprocess."},
    )
    external_eval_mode: str = field(
        default="standard",
        metadata={"help": "External eval mode: standard|verified_test_rerank"},
    )
    external_eval_verified_test_jsonl: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "raw_annotation/test.jsonl"),
        metadata={"help": "verified_test jsonl path for verified_test_rerank mode."},
    )
    external_eval_video_root: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "test/video_npy_with_meta"),
        metadata={"help": "Video npy root for verified_test_rerank mode."},
    )
    external_eval_video_meta_path: str = field(
        default="",
        metadata={"help": "Optional video meta jsonl path for verified_test_rerank mode."},
    )
    external_eval_query_embeddings_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "test/query_embedding/query_embeddings.test.npy"),
        metadata={"help": "Query embeddings path for verified_test_rerank mode."},
    )
    external_eval_query_meta_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "test/query_embedding/query_meta.test.jsonl"),
        metadata={"help": "Query meta path for verified_test_rerank mode."},
    )
    external_eval_video_embeddings_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "test/video_embedding_1fps/segment_embeds.npy"),
        metadata={"help": "Video embeddings path for verified_test_rerank mode."},
    )
    external_eval_video_docid2row_path: str = field(
        default=os.path.join(_ACTIVITYNET_ROOT, "test/video_embedding_1fps/docid2row.json"),
        metadata={"help": "Video docid2row path for verified_test_rerank mode."},
    )
    external_eval_topk: str = field(
        default="1,5,10,100",
        metadata={"help": "Top-k list for verified_test_rerank mode."},
    )
    external_eval_max_samples: int = field(
        default=0,
        metadata={"help": "Optional max samples for verified_test_rerank mode (0 = all)."},
    )
    external_eval_limit_ratio: float = field(
        default=1.0,
        metadata={"help": "Optional subset ratio for verified_test_rerank mode (0~1)."},
    )
    external_eval_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Generation max_new_tokens for verified_test_rerank mode."},
    )
    external_eval_output_jsonl: bool = field(
        default=True,
        metadata={"help": "Write per-sample jsonl for verified_test_rerank mode."},
    )
    retrieval_debug: bool = field(
        default=False,
        metadata={"help": "Enable detailed retrieval-loss debug logging."},
    )
    retrieval_debug_max_logs: int = field(
        default=500,
        metadata={"help": "Maximum number of debug rows to log."},
    )
    retrieval_debug_jsonl: str = field(
        default="logs/retrieval_debug.jsonl",
        metadata={"help": "Path (relative to output_dir) for retrieval debug jsonl."},
    )


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_query_meta_maps(path: str):
    qid_to_row: Dict[str, int] = {}
    qid_to_pos: Dict[str, str] = {}
    qid_to_query_text: Dict[str, str] = {}
    for row_idx, row in enumerate(_iter_jsonl(path)):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        qid_to_row[qid] = row_idx
        pos_doc = str(row.get("pos_doc_id", "")).strip()
        if pos_doc:
            qid_to_pos[qid] = pos_doc
        query_text = str(row.get("query", "")).strip()
        if query_text:
            qid_to_query_text[qid] = query_text
    return qid_to_row, qid_to_pos, qid_to_query_text


def _extract_query_from_user_content(content: object) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                v = item.get("text")
                if isinstance(v, str):
                    chunks.append(v)
            elif isinstance(item, str):
                chunks.append(item)
        text = "\n".join(chunks)
    else:
        return ""
    text = text.strip()
    if not text:
        return ""
    m = re.search(r'Query:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"Query:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip().strip('"')
    return ""


def _fill_query_text_map_from_dataset(dataset, qid_to_query_text: Dict[str, str]) -> int:
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        source_indices = list(dataset.indices)
    else:
        base_dataset = dataset
        source_indices = list(range(len(dataset)))

    added = 0
    for idx in source_indices:
        sample = base_dataset[idx]
        if not isinstance(sample, dict):
            continue
        qid = str(sample.get("qid", "")).strip()
        if not qid:
            continue
        if qid in qid_to_query_text and qid_to_query_text[qid]:
            continue
        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            continue
        query_text = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).strip().lower() != "user":
                continue
            query_text = _extract_query_from_user_content(msg.get("content"))
            if query_text:
                break
        if query_text:
            qid_to_query_text[qid] = query_text
            added += 1
    return added


def _bool_str(v: bool) -> str:
    return "True" if bool(v) else "False"


def _safe_arg(v):
    if isinstance(v, bool):
        return _bool_str(v)
    return str(v)


def _build_refine_tokens(refine_token: str, refine_token_count: int) -> List[str]:
    token = str(refine_token or "<REFINE>").strip()
    if not token:
        token = "<REFINE>"
    # Keep refine token registration strictly single-token.
    return [token]


def _resolve_text_hidden_size(model: torch.nn.Module, default: int = 2048) -> Tuple[int, str]:
    """
    Resolve text hidden size robustly across Qwen3-VL model scales/config variants.
    Prefer text_config.hidden_size when available, then fallback to other fields.
    """
    candidates: List[Tuple[str, Optional[object]]] = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", None)
        candidates.append(("config.text_config.hidden_size", getattr(text_cfg, "hidden_size", None)))
        candidates.append(("config.hidden_size", getattr(cfg, "hidden_size", None)))
    lm_head = getattr(model, "lm_head", None)
    candidates.append(("model.lm_head.in_features", getattr(lm_head, "in_features", None)))

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


def _resolve_retrieval_dim(
    query_embeddings_path: str,
    video_embeddings_path: str,
    default: int = 2048,
) -> int:
    dims: List[Tuple[str, int]] = []
    for label, path in (
        ("query", str(query_embeddings_path or "").strip()),
        ("video", str(video_embeddings_path or "").strip()),
    ):
        if not path or not os.path.exists(path):
            continue
        arr = np.load(path, mmap_mode="r")
        if getattr(arr, "ndim", 0) < 2:
            raise ValueError(f"{label} embeddings at {path} must be 2D, got shape={getattr(arr, 'shape', None)}")
        dims.append((label, int(arr.shape[1])))
    if not dims:
        return int(default)
    dim_values = {dim for _, dim in dims}
    if len(dim_values) != 1:
        detail = ", ".join(f"{label}={dim}" for label, dim in dims)
        raise ValueError(f"Retrieval embedding dims mismatch: {detail}")
    return int(dims[0][1])


def _save_query_embedder_for_checkpoint(
    query_embedder_model: Optional[torch.nn.Module],
    query_embedder_processor,
    checkpoint_dir: str,
    logger_: Optional[logging.Logger] = None,
    log_prefix: str = "[QueryEmbedder]",
) -> str:
    if query_embedder_model is None or query_embedder_processor is None:
        return ""
    q_dir = os.path.join(checkpoint_dir, "query_embedder")
    try:
        os.makedirs(q_dir, exist_ok=True)
        model_to_save = query_embedder_model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        model_to_save.save_pretrained(q_dir)
        query_embedder_processor.save_pretrained(q_dir)
        return q_dir
    except Exception as exc:
        if logger_ is not None:
            logger_.warning(f"{log_prefix} failed to save query embedder at {q_dir}: {exc}")
        return ""


class QueryEmbedderCheckpointCallback(TrainerCallback):
    """Save query embedder into each trainer checkpoint directory."""

    def __init__(
        self,
        use_query_embedder_path: bool,
        query_embedder_model: Optional[torch.nn.Module],
        query_embedder_processor,
        logger_: logging.Logger,
    ):
        self.use_query_embedder_path = bool(use_query_embedder_path)
        self.query_embedder_model = query_embedder_model
        self.query_embedder_processor = query_embedder_processor
        self.logger = logger_
        self._last_saved_step = -1

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if not self.use_query_embedder_path:
            return
        if self.query_embedder_model is None or self.query_embedder_processor is None:
            return
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step == self._last_saved_step:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(checkpoint_dir):
            self.logger.warning(
                "[QueryEmbedderCheckpoint] checkpoint not found at save event: "
                f"step={step}, path={checkpoint_dir}"
            )
            return
        saved_dir = _save_query_embedder_for_checkpoint(
            query_embedder_model=self.query_embedder_model,
            query_embedder_processor=self.query_embedder_processor,
            checkpoint_dir=checkpoint_dir,
            logger_=self.logger,
            log_prefix="[QueryEmbedderCheckpoint]",
        )
        if saved_dir:
            self._last_saved_step = step
            self.logger.info(
                f"[QueryEmbedderCheckpoint] saved query embedder at step={step}: {saved_dir}"
            )


class ExternalGpuEvalCallback(TrainerCallback):
    """Run eval-only subprocess on a separate GPU for each saved checkpoint."""

    def __init__(
        self,
        script_path: str,
        model_args: ModelArguments,
        data_args: ShareGPTDataArguments,
        training_args: SFTConfig,
        soft_args: SoftRefineArguments,
        query_embedder_model: Optional[torch.nn.Module],
        query_embedder_processor,
        logger_: logging.Logger,
    ):
        self.script_path = script_path
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.soft_args = soft_args
        self.query_embedder_model = query_embedder_model
        self.query_embedder_processor = query_embedder_processor
        self.logger = logger_
        self._last_eval_step = -1
        self._running_proc: Optional[subprocess.Popen] = None
        self._running_step: int = -1
        self._running_mode: str = ""
        self._running_out_json: str = ""
        self._running_out_jsonl: str = ""

    def _save_query_embedder_for_checkpoint(self, checkpoint_dir: str) -> str:
        if not bool(self.soft_args.use_query_embedder_path):
            return ""
        return _save_query_embedder_for_checkpoint(
            query_embedder_model=self.query_embedder_model,
            query_embedder_processor=self.query_embedder_processor,
            checkpoint_dir=checkpoint_dir,
            logger_=self.logger,
            log_prefix="[ExternalEval]",
        )

    def _summarize_verified_test_json(
        self, step: int, mode: str, out_json: str, out_jsonl: str
    ) -> None:
        if mode != "verified_test_rerank" or not out_json or not os.path.exists(out_json):
            return
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            fin = obj.get("final", {})
            ori = obj.get("original", {})
            self.logger.info(
                "[ExternalEvalSummary] "
                + json.dumps(
                    {
                        "step": int(step),
                        "mode": mode,
                        "rows_valid": int(obj.get("rows_valid", 0)),
                        "final_R@1": fin.get("R@1", None),
                        "final_R@5": fin.get("R@5", None),
                        "final_R@10": fin.get("R@10", None),
                        "final_R@100": fin.get("R@100", None),
                        "final_mrr": fin.get("mrr", None),
                        "orig_R@1": ori.get("R@1", None),
                        "orig_mrr": ori.get("mrr", None),
                        "json": out_json,
                        "jsonl": out_jsonl if out_jsonl and os.path.exists(out_jsonl) else "",
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            self.logger.warning(f"[ExternalEval] failed to summarize json output: {exc}")

    def _poll_running_eval(self) -> None:
        if self._running_proc is None:
            return
        rc = self._running_proc.poll()
        if rc is None:
            return

        step = self._running_step
        mode = self._running_mode
        out_json = self._running_out_json
        out_jsonl = self._running_out_jsonl
        self._running_proc = None
        self._running_step = -1
        self._running_mode = ""
        self._running_out_json = ""
        self._running_out_jsonl = ""

        if rc != 0:
            self.logger.warning(f"[ExternalEval] failed at step={step} (exit={rc})")
            return
        self.logger.info(f"[ExternalEval] done step={step}")
        self._summarize_verified_test_json(step=step, mode=mode, out_json=out_json, out_jsonl=out_jsonl)

    def _build_verified_test_cmd(
        self, checkpoint_dir: str, output_dir: str
    ) -> tuple[List[str], str, str]:
        repo_root = os.path.dirname(os.path.dirname(self.script_path))
        eval_script = os.path.join(repo_root, "scripts", "internal", "stage1", "eval_verified_test_rerank.py")
        ckpt_name = os.path.basename(os.path.normpath(checkpoint_dir))
        logs_dir = os.path.join(output_dir, self.soft_args.eval_detail_dir)
        os.makedirs(logs_dir, exist_ok=True)
        out_json = os.path.join(logs_dir, f"external_verified_test_{ckpt_name}.json")
        out_jsonl = os.path.join(logs_dir, f"external_verified_test_{ckpt_name}.jsonl")
        query_embedder_path = self.soft_args.query_embedder_model_path
        ckpt_query_embedder = os.path.join(checkpoint_dir, "query_embedder")
        if os.path.isdir(ckpt_query_embedder):
            query_embedder_path = ckpt_query_embedder

        cmd = [
            sys.executable,
            eval_script,
            "--model_path",
            checkpoint_dir,
            "--processor_path",
            checkpoint_dir,
            "--verified_test_jsonl",
            self.soft_args.external_eval_verified_test_jsonl,
            "--video_root",
            self.soft_args.external_eval_video_root,
            "--video_meta_path",
            self.soft_args.external_eval_video_meta_path,
            "--query_embeddings_path",
            self.soft_args.external_eval_query_embeddings_path,
            "--query_meta_path",
            self.soft_args.external_eval_query_meta_path,
            "--video_embeddings_path",
            self.soft_args.external_eval_video_embeddings_path,
            "--video_docid2row_path",
            self.soft_args.external_eval_video_docid2row_path,
            "--refine_token",
            self.soft_args.refine_token,
            "--refine_token_count",
            _safe_arg(self.soft_args.refine_token_count),
            "--use_refine_gate",
            _safe_arg(self.soft_args.use_refine_gate),
            "--use_query_embedder_path",
            _safe_arg(self.soft_args.use_query_embedder_path),
            "--query_embedder_model_path",
            query_embedder_path,
            "--qfinal_pooling",
            self.soft_args.qfinal_pooling,
            "--qfinal_normalize",
            _safe_arg(self.soft_args.qfinal_normalize),
            "--query_embedder_max_length",
            _safe_arg(self.soft_args.query_embedder_max_length),
            "--model_max_length",
            _safe_arg(self.model_args.model_max_length),
            "--max_new_tokens",
            _safe_arg(self.soft_args.external_eval_max_new_tokens),
            "--topk",
            self.soft_args.external_eval_topk,
            "--max_samples",
            _safe_arg(self.soft_args.external_eval_max_samples),
            "--eval_limit_ratio",
            _safe_arg(self.soft_args.external_eval_limit_ratio),
            "--seed",
            _safe_arg(self.training_args.seed),
            "--bf16",
            _safe_arg(self.training_args.bf16),
            "--output_json",
            out_json,
        ]
        if self.soft_args.external_eval_output_jsonl:
            cmd.extend(["--output_jsonl", out_jsonl])
        return cmd, out_json, out_jsonl

    def _build_eval_cmd(self, checkpoint_dir: str, output_dir: str) -> List[str]:
        query_embedder_path = self.soft_args.query_embedder_model_path
        ckpt_query_embedder = os.path.join(checkpoint_dir, "query_embedder")
        if os.path.isdir(ckpt_query_embedder):
            query_embedder_path = ckpt_query_embedder
        cmd: List[str] = [
            "torchrun",
            "--nproc_per_node=1",
            self.script_path,
            "--model_path",
            checkpoint_dir,
            "--output_dir",
            output_dir,
            "--dataset_info",
            self.data_args.dataset_info,
            "--dataset_name",
            *list(self.data_args.dataset_name or []),
            "--image_min_pixels",
            _safe_arg(self.data_args.image_min_pixels),
            "--image_max_pixels",
            _safe_arg(self.data_args.image_max_pixels),
            "--video_min_pixels",
            _safe_arg(self.data_args.video_min_pixels),
            "--video_max_pixels",
            _safe_arg(self.data_args.video_max_pixels),
            "--video_total_pixels",
            _safe_arg(self.data_args.video_total_pixels),
            "--max_frames",
            _safe_arg(self.data_args.max_frames),
            "--fps",
            _safe_arg(self.data_args.fps),
            "--video_root_override",
            _safe_arg(getattr(self.data_args, "video_root_override", "")),
            "--video_meta_path",
            _safe_arg(getattr(self.data_args, "video_meta_path", "")),
            "--model_max_length",
            _safe_arg(self.model_args.model_max_length),
            "--tune_mm_llm",
            _safe_arg(self.model_args.tune_mm_llm),
            "--tune_mm_mlp",
            _safe_arg(self.model_args.tune_mm_mlp),
            "--tune_mm_vision",
            _safe_arg(self.model_args.tune_mm_vision),
            "--per_device_eval_batch_size",
            "1",
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--num_train_epochs",
            "1",
            "--save_strategy",
            "no",
            "--eval_strategy",
            "no",
            "--logging_steps",
            _safe_arg(max(int(self.training_args.logging_steps), 1)),
            "--report_to",
            self.soft_args.external_eval_report_to,
            "--bf16",
            _safe_arg(self.training_args.bf16),
            "--fp16",
            _safe_arg(self.training_args.fp16),
            "--tf32",
            _safe_arg(self.training_args.tf32),
            "--seed",
            _safe_arg(self.training_args.seed),
            "--enable_retrieval_optimization",
            _safe_arg(self.soft_args.enable_retrieval_optimization),
            "--retrieval_loss_weight",
            _safe_arg(self.soft_args.retrieval_loss_weight),
            "--retrieval_temperature",
            _safe_arg(self.soft_args.retrieval_temperature),
            "--negative_pool_size",
            _safe_arg(self.soft_args.negative_pool_size),
            "--retrieval_ignore_ambiguous_negatives",
            _safe_arg(self.soft_args.retrieval_ignore_ambiguous_negatives),
            "--ambiguous_negative_margin",
            _safe_arg(self.soft_args.ambiguous_negative_margin),
            "--strict_negative_topk",
            _safe_arg(self.soft_args.strict_negative_topk),
            "--projector_lr",
            _safe_arg(self.soft_args.projector_lr),
            "--use_refine_gate",
            _safe_arg(self.soft_args.use_refine_gate),
            "--zero_init_refine",
            _safe_arg(self.soft_args.zero_init_refine),
            "--retrieval_on_eval",
            "True",
            "--only_neg",
            _safe_arg(self.soft_args.only_neg),
            "--eval_split_ratio",
            _safe_arg(self.soft_args.eval_split_ratio),
            "--eval_recall_on_eval",
            _safe_arg(self.soft_args.eval_recall_on_eval),
            "--eval_recall_ks",
            self.soft_args.eval_recall_ks,
            "--eval_detail_dir",
            self.soft_args.eval_detail_dir,
            "--refine_token",
            self.soft_args.refine_token,
            "--refine_token_count",
            _safe_arg(self.soft_args.refine_token_count),
            "--query_embeddings_path",
            self.soft_args.query_embeddings_path,
            "--query_meta_path",
            self.soft_args.query_meta_path,
            "--video_embeddings_path",
            self.soft_args.video_embeddings_path,
            "--video_docid2row_path",
            self.soft_args.video_docid2row_path,
            "--hard_negatives_path",
            self.soft_args.hard_negatives_path,
            "--hard_negative_refresh_steps",
            _safe_arg(self.soft_args.hard_negative_refresh_steps),
            "--use_query_embedder_path",
            _safe_arg(self.soft_args.use_query_embedder_path),
            "--query_embedder_model_path",
            query_embedder_path,
            "--qfinal_pooling",
            self.soft_args.qfinal_pooling,
            "--qfinal_normalize",
            _safe_arg(self.soft_args.qfinal_normalize),
            "--tune_query_embedder",
            _safe_arg(self.soft_args.tune_query_embedder),
            "--query_embedder_lr",
            _safe_arg(self.soft_args.query_embedder_lr),
            "--query_embedder_max_length",
            _safe_arg(self.soft_args.query_embedder_max_length),
            "--retrieval_debug",
            _safe_arg(self.soft_args.retrieval_debug),
            "--retrieval_debug_max_logs",
            _safe_arg(self.soft_args.retrieval_debug_max_logs),
            "--retrieval_debug_jsonl",
            self.soft_args.retrieval_debug_jsonl,
            "--eval_only",
            "True",
        ]
        if self.model_args.apply_monkey_patch:
            cmd.extend(["--apply_monkey_patch", str(self.model_args.apply_monkey_patch)])
        return cmd

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if not self.soft_args.external_eval_on_gpu0:
            return

        self._poll_running_eval()

        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step == self._last_eval_step:
            return

        if self._running_proc is not None:
            self.logger.warning(
                f"[ExternalEval] previous eval still running at step={self._running_step}; "
                f"skip launching new eval for checkpoint-{step}"
            )
            self._last_eval_step = step
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(checkpoint_dir):
            self.logger.warning(
                f"[ExternalEval] checkpoint not found, skip step={step}: {checkpoint_dir}"
            )
            return
        saved_query_embedder = self._save_query_embedder_for_checkpoint(checkpoint_dir)
        if saved_query_embedder:
            self.logger.info(f"[ExternalEval] saved tuned query embedder: {saved_query_embedder}")

        mode = str(self.soft_args.external_eval_mode or "standard").strip().lower()
        out_json = ""
        out_jsonl = ""
        if mode == "verified_test_rerank":
            cmd, out_json, out_jsonl = self._build_verified_test_cmd(
                checkpoint_dir=checkpoint_dir, output_dir=args.output_dir
            )
        else:
            cmd = self._build_eval_cmd(checkpoint_dir=checkpoint_dir, output_dir=args.output_dir)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.soft_args.external_eval_gpu)
        env["WANDB_DISABLED"] = "true"
        env["TOKENIZERS_PARALLELISM"] = "false"

        self.logger.info(
            f"[ExternalEval] step={step} on CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}"
        )
        self.logger.info("[ExternalEval] " + " ".join(cmd))
        self._running_proc = subprocess.Popen(cmd, env=env)
        self._running_step = step
        self._running_mode = mode
        self._running_out_json = out_json
        self._running_out_jsonl = out_jsonl
        self.logger.info(
            f"[ExternalEval] launched async pid={self._running_proc.pid} for checkpoint-{step}"
        )
        self._last_eval_step = step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if not self.soft_args.external_eval_on_gpu0:
            return
        self._poll_running_eval()

    def on_train_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if not self.soft_args.external_eval_on_gpu0:
            return
        self._poll_running_eval()
        if self._running_proc is not None:
            self.logger.warning(
                f"[ExternalEval] still running at train end (step={self._running_step}, "
                f"pid={self._running_proc.pid}). It will continue in background."
            )


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


def _maybe_load_refine_weights(
    model: torch.nn.Module, model_path: str, logger_: logging.Logger
) -> bool:
    """Load soft-refine auxiliary weights from checkpoint directory when present."""
    if not os.path.isdir(model_path):
        return False
    has_refine = hasattr(model, "refine_projector") and hasattr(model, "refine_gate")
    if not has_refine:
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

    # 1) Try single-file safetensors / bin first.
    try:
        safetensor_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensor_path):
            from safetensors.torch import load_file

            _collect_from_state_dict(load_file(safetensor_path, device="cpu"))
    except Exception as exc:
        logger_.warning(f"Failed loading model.safetensors for refine weights: {exc}")

    if not refine_state:
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            try:
                _collect_from_state_dict(torch.load(bin_path, map_location="cpu"))
            except Exception as exc:
                logger_.warning(f"Failed loading pytorch_model.bin for refine weights: {exc}")

    # 2) Sharded checkpoint fallback via index file.
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
                shard_files = {
                    shard for k, shard in weight_map.items() if k.startswith(prefix)
                }
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
                logger_.warning(f"Failed loading sharded refine weights from {index_path}: {exc}")

    if refine_state:
        missing, unexpected = model.load_state_dict(refine_state, strict=False)
        logger_.info(
            "Loaded refine weights from checkpoint "
            f"({len(refine_state)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return True
    else:
        logger_.info("No refine weights found in checkpoint; using initialized refine modules.")
        return False


def set_model(model_args, model, logger):
    if model_args.tune_mm_vision:
        for p in model.visual.parameters():
            p.requires_grad = True
    else:
        for p in model.visual.parameters():
            p.requires_grad = False
    logger.info(f"tune_mm_vision: {model_args.tune_mm_vision}")

    if model_args.tune_mm_mlp:
        for p in model.visual.merger.parameters():
            p.requires_grad = True
        for p in model.visual.deepstack_merger_list.parameters():
            p.requires_grad = True
    else:
        for p in model.visual.merger.parameters():
            p.requires_grad = False
        for p in model.visual.deepstack_merger_list.parameters():
            p.requires_grad = False
    logger.info(f"tune_mm_mlp: {model_args.tune_mm_mlp}")

    if model_args.tune_mm_llm:
        for p in model.language_model.parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for p in model.language_model.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
    logger.info(f"tune_mm_llm: {model_args.tune_mm_llm}")

    if is_deepspeed_zero3_enabled():

        def numel(p):
            return p.ds_numel if hasattr(p, "ds_numel") else p.numel()

    else:

        def numel(p):
            return p.numel()

    total_params = sum(numel(p) for p in model.parameters())
    trainable_params = sum(numel(p) for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")


def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))

    parser = HfArgumentParser(
        (ModelArguments, ShareGPTDataArguments, SFTConfig, SoftRefineArguments)
    )
    model_args, data_args, training_args, soft_args = parser.parse_args_into_dataclasses()

    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logger.info(f"Global Rank: {global_rank}, Local Rank: {local_rank}")
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Soft refine args: {soft_args}")

    if model_args.apply_monkey_patch:
        apply_qwen3_vl_monkey_patch(model_args.apply_monkey_patch)
        logger.info(f"Qwen3-VL patch applied, mode: {model_args.apply_monkey_patch}")

    processor = AutoProcessor.from_pretrained(
        model_args.model_path, padding_side="left"
    )
    refine_tokens = _build_refine_tokens(
        refine_token=soft_args.refine_token,
        refine_token_count=1,
    )
    refine_rollout_depth = int(max(1, soft_args.refine_token_count))
    missing_tokens = [tok for tok in refine_tokens if tok not in processor.tokenizer.get_vocab()]
    if missing_tokens:
        processor.tokenizer.add_tokens(missing_tokens, special_tokens=True)
        logger.info(f"Added special refine tokens: {missing_tokens}")
    refine_token_ids = [processor.tokenizer.convert_tokens_to_ids(tok) for tok in refine_tokens]
    logger.info(
        "Refine token config: "
        f"token={refine_tokens[0]}, token_id={refine_token_ids[0]}, "
        f"rollout_depth={refine_rollout_depth}"
    )

    attn_impl = str(os.environ.get("ATTN_IMPL", "flash_attention_2") or "").strip()
    if not attn_impl:
        attn_impl = "flash_attention_2"

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if training_args.bf16 else None,
    }
    if attn_impl.lower() not in {"none", "auto"}:
        model_kwargs["attn_implementation"] = attn_impl
    logger.info(
        "Model attention backend: "
        f"{model_kwargs.get('attn_implementation', 'transformers_default')}"
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_path, **model_kwargs
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    logger.info(f"Model loaded from {model_args.model_path}")
    query_embedder_model = None
    query_embedder_tokenizer = None
    query_embedder_processor = None

    if training_args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    if soft_args.enable_retrieval_optimization:
        retrieval_dim = _resolve_retrieval_dim(
            query_embeddings_path=soft_args.query_embeddings_path,
            video_embeddings_path=soft_args.video_embeddings_path,
            default=2048,
        )
        hidden_size, hidden_size_src = _resolve_text_hidden_size(model, default=retrieval_dim)
        model.refine_projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, retrieval_dim),
            torch.nn.LayerNorm(retrieval_dim),
            torch.nn.GELU(),
            torch.nn.Linear(retrieval_dim, retrieval_dim),
        )
        model.refine_gate = torch.nn.Linear(hidden_size, 1)
        # rollout 단계에서 latent를 query-embedder LLM 입력으로 넣기 직전에 사용하는 projector.
        model.refine_latent_input_projector = _make_latent_input_projector(retrieval_dim, hidden_size)
        _init_partial_identity_linear(model.refine_latent_input_projector[1])
        query_embedder_hidden = retrieval_dim

        if soft_args.use_query_embedder_path:
            logger.info(
                f"Loading query embedder from {soft_args.query_embedder_model_path} "
                f"(tune_query_embedder={soft_args.tune_query_embedder})"
            )
            query_embedder_processor = AutoProcessor.from_pretrained(
                soft_args.query_embedder_model_path, padding_side="left"
            )
            query_embedder_tokenizer = query_embedder_processor.tokenizer
            query_model_kwargs = dict(model_kwargs)
            try:
                query_embedder_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    soft_args.query_embedder_model_path, **query_model_kwargs
                )
            except Exception as exc:
                if "attn_implementation" in query_model_kwargs:
                    logger.warning(f"Query embedder flash_attention_2 load failed: {exc}; retry without it.")
                    query_model_kwargs.pop("attn_implementation", None)
                    query_embedder_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        soft_args.query_embedder_model_path, **query_model_kwargs
                    )
                else:
                    raise

            if hasattr(query_embedder_model, "config"):
                query_embedder_model.config.use_cache = False
            query_embedder_hidden, query_embedder_hidden_src = _resolve_text_hidden_size(
                query_embedder_model, default=retrieval_dim
            )
            if query_embedder_hidden != retrieval_dim:
                model.query_embedder_head = torch.nn.Linear(query_embedder_hidden, retrieval_dim)
                logger.info(
                    "query_embedder_head enabled for hidden projection "
                    f"({query_embedder_hidden} -> {retrieval_dim}, source={query_embedder_hidden_src})"
                )
            else:
                model.query_embedder_head = torch.nn.Identity()
                logger.info(
                    f"query_embedder_head is Identity (hidden size already {retrieval_dim}, "
                    f"source={query_embedder_hidden_src})."
                )

            for p in query_embedder_model.parameters():
                p.requires_grad = bool(soft_args.tune_query_embedder)
            if soft_args.tune_query_embedder:
                query_embedder_model.train()
            else:
                query_embedder_model.eval()
            query_embedder_model.to(
                torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            )
            # Register query embedder as a submodule when tuning so DeepSpeed/optimizer
            # can resolve param-name mappings from self.model.named_parameters().
            if soft_args.tune_query_embedder:
                model.query_embedder_model = query_embedder_model
                logger.info("Attached query_embedder_model to main model for distributed optimizer compatibility.")

        # q_final 계산 시 append되는 latent 전용 projector.
        # rollout용과 분리해서 단계별 역할을 분명하게 유지한다.
        model.refine_append_input_projector = _make_latent_input_projector(
            retrieval_dim, query_embedder_hidden
        )
        _init_partial_identity_linear(model.refine_append_input_projector[1])

        logger.info(
            "Retrieval optimization modules enabled "
            f"(hidden_size={hidden_size}, hidden_size_source={hidden_size_src}, retrieval_dim={retrieval_dim})"
        )
        loaded_refine = _maybe_load_refine_weights(model, model_args.model_path, logger)
        if soft_args.zero_init_refine:
            if loaded_refine:
                logger.info("zero_init_refine=True but skipped because refine weights were loaded.")
            else:
                torch.nn.init.zeros_(model.refine_projector[-1].weight)
                torch.nn.init.zeros_(model.refine_projector[-1].bias)
                torch.nn.init.zeros_(model.refine_gate.weight)
                torch.nn.init.zeros_(model.refine_gate.bias)
                logger.info(
                    "Applied zero-init to refine modules "
                    "(projector[-1].weight/bias, gate.weight/bias)."
                )
        if not soft_args.use_refine_gate:
            for p in model.refine_gate.parameters():
                p.requires_grad = False
            logger.info("Refine gate disabled by args: gate parameters are frozen.")

    set_model(model_args, model, logger)

    full_dataset = build_sharegpt_dataset(data_args)
    if soft_args.only_neg:
        filtered_dataset, before_n, after_n = _filter_only_neg_dataset(full_dataset)
        if after_n == 0:
            raise ValueError("only_neg=True but no negative samples were found.")
        full_dataset = filtered_dataset
        logger.info(f"Applied only_neg filter: {before_n} -> {after_n}")

    train_dataset = full_dataset
    eval_dataset = None

    # `training_args.eval_strategy` can be an enum (e.g., IntervalStrategy.NO).
    # Normalize with `.value` when present to avoid accidental eval split on "no".
    eval_mode = str(getattr(training_args.eval_strategy, "value", training_args.eval_strategy)).lower()
    need_eval_split = (eval_mode != "no") or bool(soft_args.eval_only)
    if need_eval_split:
        total = len(full_dataset)
        if total < 2:
            logger.warning("Dataset too small for eval split; disabling eval.")
            training_args.eval_strategy = "no"
            eval_mode = "no"
        else:
            ratio = float(max(0.01, min(0.99, soft_args.eval_split_ratio)))
            eval_size = max(1, int(round(total * ratio)))
            train_size = total - eval_size
            if train_size <= 0:
                train_size = total - 1
                eval_size = 1
            split_gen = torch.Generator().manual_seed(int(training_args.seed))
            train_dataset, eval_dataset = random_split(
                full_dataset,
                [train_size, eval_size],
                generator=split_gen,
            )
            logger.info(
                f"Dataset split for eval: total={total} train={train_size} eval={eval_size} "
                f"(ratio={ratio:.3f})"
            )
    if soft_args.eval_only and eval_dataset is None:
        raise ValueError("eval_only=True requires a valid eval dataset split.")

    loss_type = os.environ.get("LOSS_TYPE", "only_assistant")
    collator = ShareGPTSFTCollator(
        processor,
        loss_type=loss_type,
        max_length=model_args.model_max_length,
        image_patch_size=16,
        use_video_metadata=True,
    )

    callbacks = []
    if (
        soft_args.use_query_embedder_path
        and query_embedder_model is not None
        and query_embedder_processor is not None
        and not soft_args.eval_only
    ):
        callbacks.append(
            QueryEmbedderCheckpointCallback(
                use_query_embedder_path=soft_args.use_query_embedder_path,
                query_embedder_model=query_embedder_model,
                query_embedder_processor=query_embedder_processor,
                logger_=logger,
            )
        )
    if eval_mode != "no" and not soft_args.eval_only:
        callbacks.append(
            EvalDetailLoggerCallback(
                output_dir=os.path.join(training_args.output_dir, soft_args.eval_detail_dir)
            )
        )
    if soft_args.external_eval_on_gpu0 and not soft_args.eval_only:
        callbacks.append(
            ExternalGpuEvalCallback(
                script_path=os.path.abspath(__file__),
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                soft_args=soft_args,
                query_embedder_model=query_embedder_model,
                query_embedder_processor=query_embedder_processor,
                logger_=logger,
            )
        )

    if soft_args.enable_retrieval_optimization:
        for path in (
            soft_args.query_embeddings_path,
            soft_args.query_meta_path,
            soft_args.video_embeddings_path,
            soft_args.video_docid2row_path,
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing retrieval input file: {path}")

        query_embeddings = np.load(soft_args.query_embeddings_path, mmap_mode="r")
        qid_to_query_row, qid_to_pos_video, qid_to_query_text = _load_query_meta_maps(
            soft_args.query_meta_path
        )
        added_query_text = _fill_query_text_map_from_dataset(full_dataset, qid_to_query_text)
        video_embeddings = np.load(soft_args.video_embeddings_path, mmap_mode="r")
        with open(soft_args.video_docid2row_path, "r", encoding="utf-8") as f:
            video_id_to_row = {str(k): int(v) for k, v in json.load(f).items()}

        hard_negatives = {}
        if soft_args.hard_negatives_path and os.path.exists(soft_args.hard_negatives_path):
            with open(soft_args.hard_negatives_path, "r", encoding="utf-8") as f:
                raw_neg = json.load(f)
            if isinstance(raw_neg, dict):
                hard_negatives = {
                    str(k): [str(x) for x in v] for k, v in raw_neg.items() if isinstance(v, list)
                }

        logger.info(
            "Loaded retrieval resources: "
            f"query_embeddings={query_embeddings.shape}, "
            f"video_embeddings={video_embeddings.shape}, "
            f"hard_neg_qids={len(hard_negatives)}, "
            f"qid_to_query_text={len(qid_to_query_text)} (+{added_query_text} from dataset fallback)"
        )

        trainer = SoftRefineTrainer(
            model=model,
            processing_class=processor,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            refine_token_ids=refine_token_ids,
            query_embeddings=query_embeddings,
            qid_to_query_row=qid_to_query_row,
            qid_to_pos_video=qid_to_pos_video,
            qid_to_query_text=qid_to_query_text,
            video_embeddings=video_embeddings,
            video_id_to_row=video_id_to_row,
            hard_negatives=hard_negatives,
            hard_negative_refresh_steps=soft_args.hard_negative_refresh_steps,
            enable_retrieval_optimization=soft_args.enable_retrieval_optimization,
            retrieval_loss_weight=soft_args.retrieval_loss_weight,
            retrieval_temperature=soft_args.retrieval_temperature,
            negative_pool_size=soft_args.negative_pool_size,
            retrieval_ignore_ambiguous_negatives=soft_args.retrieval_ignore_ambiguous_negatives,
            ambiguous_negative_margin=soft_args.ambiguous_negative_margin,
            strict_negative_topk=soft_args.strict_negative_topk,
            projector_lr=soft_args.projector_lr,
            use_refine_gate=soft_args.use_refine_gate,
            retrieval_on_eval=soft_args.retrieval_on_eval,
            retrieval_debug=soft_args.retrieval_debug,
            retrieval_debug_max_logs=soft_args.retrieval_debug_max_logs,
            retrieval_debug_jsonl=soft_args.retrieval_debug_jsonl,
            use_query_embedder_path=soft_args.use_query_embedder_path,
            query_embedder_model=query_embedder_model,
            query_embedder_tokenizer=query_embedder_tokenizer,
            qfinal_pooling=soft_args.qfinal_pooling,
            qfinal_normalize=soft_args.qfinal_normalize,
            tune_query_embedder=soft_args.tune_query_embedder,
            query_embedder_lr=soft_args.query_embedder_lr,
            query_embedder_max_length=soft_args.query_embedder_max_length,
            refine_rollout_depth=refine_rollout_depth,
            random_seed=int(training_args.seed),
            eval_recall_on_eval=soft_args.eval_recall_on_eval,
            eval_recall_ks=soft_args.eval_recall_ks,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            processing_class=processor,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

    if soft_args.eval_only:
        metrics = trainer.evaluate()
        logger.info(f"Eval-only metrics: {metrics}")
        if trainer.accelerator.is_main_process:
            log_dir = os.path.join(training_args.output_dir, soft_args.eval_detail_dir)
            os.makedirs(log_dir, exist_ok=True)
            ckpt_name = os.path.basename(os.path.normpath(model_args.model_path))
            out_path = os.path.join(log_dir, f"external_eval_{ckpt_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved eval-only metrics to {out_path}")
        return
    else:
        trainer.train()
        logger.info("Training finished")

        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        if (
            soft_args.use_query_embedder_path
            and query_embedder_model is not None
            and query_embedder_processor is not None
        ):
            q_out_dir = os.path.join(training_args.output_dir, "query_embedder")
            os.makedirs(q_out_dir, exist_ok=True)
            model_to_save = query_embedder_model.module if hasattr(query_embedder_model, "module") else query_embedder_model
            model_to_save.save_pretrained(q_out_dir)
            query_embedder_processor.save_pretrained(q_out_dir)
            logger.info(f"Saved query embedder to {q_out_dir}")

        if trainer.accelerator.is_main_process:
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
