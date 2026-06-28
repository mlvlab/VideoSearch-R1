# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import copy
import json
import re
import sys
import time
import faulthandler
import unicodedata
from datetime import datetime
from collections import defaultdict
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import numpy as np
import transformers
import requests
import torch.nn.functional as F
from accelerate.utils import gather, set_seed
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models.utils import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import entropy_from_logits, pad, selective_log_softmax

from model.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration
from model.qwen_vl_utils.vision_process import cached_process_vision_info
from utils.arguments import GRPOConfig
from utils.video_metadata import (
    load_video_meta_index,
    resolve_video_meta_for_video_path,
)

try:
    import deepspeed
except Exception:
    deepspeed = None

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


def _patch_vllm_qwen3_loader_for_soft_refine(
    ignore_prefixes: tuple[str, ...] = (
        "refine_projector",
        "refine_gate",
        "query_embedder_head",
        "query_embedder_model",
        "refine_latent_input_projector",
        "refine_append_input_projector",
    ),
) -> None:
    """
    vLLM's Qwen3-VL loader fails on extra checkpoint keys used by our soft-refine
    branch (e.g. refine_projector/refine_gate). Patch once per process to ignore
    those unexpected prefixes during vLLM initial load.
    """
    if not is_vllm_available():
        return
    try:
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLForConditionalGeneration as _VLLMQwen3VLForConditionalGeneration,
        )
        from vllm.model_executor.models.utils import AutoWeightsLoader
    except Exception as exc:
        print(f"[vllm][warn] failed to import qwen3_vl loader for patch: {exc}")
        return

    model_cls = _VLLMQwen3VLForConditionalGeneration
    patch_flag = "_soft_refine_ignore_patch_done"
    if getattr(model_cls, patch_flag, False):
        return

    original_load_weights = model_cls.load_weights

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
    setattr(model_cls, "_soft_refine_original_load_weights", original_load_weights)
    setattr(model_cls, patch_flag, True)
    print(
        "[vllm] patched Qwen3-VL loader to ignore unexpected prefixes: "
        + ", ".join(ignore_prefixes)
    )


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_refine_tokens(refine_token: str, refine_token_count: int) -> list[str]:
    token = str(refine_token or "<REFINE>").strip()
    if not token:
        token = "<REFINE>"
    # GRPO refine path now uses a single token; rollout depth controls latent steps.
    return [token]


_GEN_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
_MM_TEXT_TOKEN_PATTERN = re.compile(
    r"<\|(?:video_pad|image_pad|video(?:_\d+)?|image(?:_\d+)?|vision_start|vision_end)\|>"
    r"|</?(?:video(?:_\d+)?|image(?:_\d+)?)>"
)


def _sanitize_generated_text(text: str) -> str:
    """
    Keep assistant generations text-safe for chat-template re-tokenization.
    Strip trailing decode artifacts and multimodal placeholder tokens that can
    break downstream processor assumptions.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    for stop_tok in _GEN_STOP_TOKENS:
        pos = cleaned.find(stop_tok)
        if pos >= 0:
            cleaned = cleaned[:pos]
    cleaned = _MM_TEXT_TOKEN_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("\x00", "")
    return cleaned.strip()


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


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (
            (self.num_samples // self.batch_size)
            * self.batch_size
            * self.mini_repeat_count
            * self.repeat_count
        )


def split_to_chunk(input_dict: dict, num_chunks: int) -> list[dict]:
    """
    Splits a dictionary into `num_chunks` equal parts. If the value is tensor, then split along the first dimension.
    If the value is a list, then split accordingly.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> z = [1, 2, 3, None, 5, 6]
    >>> input_dict = {"x": x, "y": y, "z": z}
    >>> split_to_chunk(input_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]]), "z": [1, 2]},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]]), "z": [3, None]},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]]), "z": [5, 6]},
    ]
    ```
    """
    first_tensor = next(v for v in input_dict.values() if isinstance(v, torch.Tensor))
    chunk_size = first_tensor.shape[0] // num_chunks

    splitted_dict = []
    for i in range(num_chunks):
        tmp_dict = {}
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                tmp_dict[k] = v[i * chunk_size : (i + 1) * chunk_size]
            elif isinstance(v, list):
                tmp_dict[k] = [
                    v[j] for j in range(i * chunk_size, (i + 1) * chunk_size)
                ]
            else:
                raise ValueError(f"Unsupported type {type(v)} for key {k}")
        splitted_dict.append(tmp_dict)
    return splitted_dict


def shuffle_sequence_dict(
    seq_dict: dict[str, Optional[Sequence]],
) -> dict[str, Optional[Sequence]]:
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v: Optional[Sequence]) -> Optional[Sequence]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def _extract_search_query(text: str) -> str:
    if not text:
        return ""
    pattern = r"<search>(.*?)</search>"
    hits = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if hits:
        return hits[-1].strip()
    start = text.lower().rfind("<search>")
    if start >= 0:
        return text[start + len("<search>") :].strip()
    return ""


def _extract_answer_tag(text: str) -> str:
    if not text:
        return ""
    match = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    val = match[-1].strip().lower().replace(" ", "_")
    if val in ("match", "matched"):
        return "matched"
    if val in ("not_match", "not_matched", "notmatch"):
        return "not_matched"
    return val


def _extract_start_end(text: str) -> tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    start_hits = re.findall(
        r"<start>\s*([-+]?\d+(?:\.\d+)?)\s*</start>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    end_hits = re.findall(
        r"<end>\s*([-+]?\d+(?:\.\d+)?)\s*</end>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not start_hits or not end_hits:
        return None, None
    try:
        return float(start_hits[-1]), float(end_hits[-1])
    except (TypeError, ValueError):
        return None, None


def _normalize_gt_span(raw: Any) -> tuple[Optional[float], Optional[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None, None
    try:
        gt_start = float(raw[0])
        gt_end = float(raw[1])
    except (TypeError, ValueError):
        return None, None
    return gt_start, gt_end


def _compute_temporal_iou(
    pred_start: Optional[float],
    pred_end: Optional[float],
    gt_start: Optional[float],
    gt_end: Optional[float],
) -> Optional[float]:
    if (
        pred_start is None
        or pred_end is None
        or gt_start is None
        or gt_end is None
    ):
        return None
    if pred_end <= pred_start or gt_end <= gt_start:
        return None
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return None
    return float(inter / union)


def _extract_search_instruction(text: str) -> str:
    if not text:
        return ""
    tag = "search_instruction"
    pattern = rf"<{tag}>(.*?)</{tag}>"
    hits = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if hits:
        instr = hits[-1].strip()
        if instr:
            return instr
    start = text.lower().rfind(f"<{tag}>")
    if start >= 0:
        return text[start + len(tag) + 2 :].strip()
    return ""


def _default_search_msg(query: str, think: str) -> str:
    if think:
        return f"<think>\n{think}\n</think>\n<search>\n{query}\n</search>"
    return f"<search>\n{query}\n</search>"


def _is_bad_search_query(query: str, min_chars: int, min_alnum: int) -> bool:
    if not query:
        return True
    if min_chars and len(query) < min_chars:
        return True
    if min_alnum:
        alnum_cnt = sum(1 for ch in query if ch.isalnum())
        if alnum_cnt < min_alnum:
            return True
    return False


def _build_env_content(
    video_path: str, cfg: dict[str, Any], video_meta: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    video_payload: dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "min_pixels": cfg["video_min_pixels"],
        "max_pixels": cfg["video_max_pixels"],
        "total_pixels": cfg["video_total_pixels"],
        "max_frames": cfg["max_frames"],
        "fps": cfg["fps"],
    }
    if isinstance(video_meta, dict) and video_meta:
        video_payload.update(video_meta)
    return [
        {"type": "text", "text": "<information>\nRetrieved video: "},
        video_payload,
        {"type": "text", "text": "\n</information>"},
    ]


def _resolve_retrieved_video_path(video_root: str, retrieved_id: str) -> str:
    root = str(video_root or "").strip()
    rid = str(retrieved_id or "").strip()
    if not root or not rid:
        return ""
    if os.path.isabs(rid) and os.path.exists(rid):
        return rid

    exts = [".npy", ".npz", ".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"]
    rid_base = os.path.basename(rid)
    rid_stem = os.path.splitext(rid_base)[0]
    id_variants = [rid, rid_base]
    if rid_stem:
        id_variants.append(rid_stem)
    cands: list[str] = [
        os.path.join(root, rid),
        os.path.join(root, "test_video_npy", rid),
    ]
    for rid_var in id_variants:
        cands.append(os.path.join(root, rid_var))
        cands.append(os.path.join(root, "test_video_npy", rid_var))
        for ext in exts:
            cands.append(os.path.join(root, f"{rid_var}{ext}"))
            cands.append(os.path.join(root, "test_video_npy", f"{rid_var}{ext}"))
    for p in cands:
        if os.path.exists(p):
            return p
    return ""


def _load_query_meta_by_query(path: str) -> dict[str, tuple[int, str, str]]:
    # key: exact query text, value: (row_idx, qid, pos_doc_id)
    out: dict[str, tuple[int, str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            qid = str(row.get("qid", "")).strip()
            pos_doc_id = str(row.get("pos_doc_id", "")).strip()
            out[query] = (row_idx, qid, pos_doc_id)
    return out


def _l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def _zero_init_disabled_ctx():
    if deepspeed is None:
        return nullcontext()
    zero_mod = getattr(deepspeed, "zero", None)
    if zero_mod is None or not hasattr(zero_mod, "Init"):
        return nullcontext()
    try:
        return zero_mod.Init(enabled=False)
    except Exception:
        return nullcontext()


def _embed_tokens_safely(embedding_module: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    try:
        return embedding_module(input_ids)
    except RuntimeError as exc:
        msg = str(exc)
        if "weight" not in msg or "2-D" not in msg:
            raise

        weight = getattr(embedding_module, "weight", None)
        if weight is None:
            raise

        gather_ctx = nullcontext()
        if deepspeed is not None:
            zero_mod = getattr(deepspeed, "zero", None)
            if zero_mod is not None and hasattr(zero_mod, "GatheredParameters"):
                try:
                    gather_ctx = zero_mod.GatheredParameters([weight], modifier_rank=None)
                except Exception:
                    gather_ctx = nullcontext()

        with gather_ctx:
            if getattr(weight, "ndim", 0) != 2:
                raise RuntimeError(
                    "Query embedder embedding weight is not 2-D even after gather: "
                    f"shape={tuple(weight.shape)}"
                ) from exc
            return F.embedding(
                input_ids,
                weight,
                padding_idx=getattr(embedding_module, "padding_idx", None),
                max_norm=getattr(embedding_module, "max_norm", None),
                norm_type=getattr(embedding_module, "norm_type", 2.0),
                scale_grad_by_freq=getattr(embedding_module, "scale_grad_by_freq", False),
                sparse=getattr(embedding_module, "sparse", False),
            )


def _module_has_zero3_partitioned_params(module: torch.nn.Module) -> bool:
    for p in module.parameters():
        if hasattr(p, "ds_id") or hasattr(p, "ds_status"):
            return True
    return False


def _gather_module_params_ctx(module: torch.nn.Module):
    if deepspeed is None:
        return nullcontext()
    zero_mod = getattr(deepspeed, "zero", None)
    if zero_mod is None or not hasattr(zero_mod, "GatheredParameters"):
        return nullcontext()
    params = [p for p in module.parameters() if p is not None]
    if not params:
        return nullcontext()
    # Gather all params for forward when module was born under ZeRO-3 partition context.
    return zero_mod.GatheredParameters(params, modifier_rank=None)


def identity(x):
    """Do we really need docs for this?"""
    return x


def split_visual_data(batch):
    """
    Splits and reorganizes the visual data (images and videos) in a batch into per-sample structures aligned
    with the message layout. It separates and groups `pixel_values`, `pixel_values_videos`, `image_grid_thw`,
    `video_grid_thw` into a list of tensors (could be None if no visual information
    in this sample), so that each sample has its own corresponding visual tensors instead of one large
    concatenated tensor across the batch.
    """
    # if only have text data, skip
    if "pixel_values" not in batch and "pixel_values_videos" not in batch:
        return batch

    # split the image pixel values
    if "image_grid_thw" in batch:
        image_lengths = batch["image_grid_thw"].prod(dim=1).tolist()
        if sum(image_lengths) != batch["pixel_values"].size(0):
            raise ValueError(
                f"Mismatch: sum(image_lengths) = {sum(image_lengths)} != pixel_values.size(0) = {batch['pixel_values'].size(0)}"
            )
        split_pixel_values = list(
            torch.split(batch["pixel_values"], image_lengths, dim=0)
        )  # [total, feature_dim]

    # split the video pixel values
    if "video_grid_thw" in batch:
        video_lengths = batch["video_grid_thw"].prod(dim=1).tolist()
        if sum(video_lengths) != batch["pixel_values_videos"].size(0):
            raise ValueError(
                f"Mismatch: sum(video_lengths) = {sum(video_lengths)} != pixel_values_videos.size(0) = {batch['pixel_values_videos'].size(0)}"
            )
        split_pixel_values_videos = list(
            torch.split(batch["pixel_values_videos"], video_lengths, dim=0)
        )  # [total, feature_dim]

    batch_pixel_values = []
    batch_image_grid_thw = []
    batch_pixel_values_videos = []
    batch_video_grid_thw = []
    image_idx = 0
    video_idx = 0

    for message in batch["messages"]:
        tmp_pixel_values = []
        tmp_image_grid_thw = []
        tmp_pixel_values_videos = []
        tmp_video_grid_thw = []

        for msg in message:
            if isinstance(msg["content"], list):
                for ele in msg["content"]:
                    if "image" in ele or "image_url" in ele:
                        tmp_pixel_values.append(split_pixel_values[image_idx])
                        tmp_image_grid_thw.append(batch["image_grid_thw"][image_idx])
                        image_idx += 1

                    if "video" in ele:
                        tmp_pixel_values_videos.append(
                            split_pixel_values_videos[video_idx]
                        )
                        tmp_video_grid_thw.append(batch["video_grid_thw"][video_idx])
                        video_idx += 1

        if len(tmp_pixel_values) > 0:
            tmp_pixel_values = torch.cat(tmp_pixel_values, dim=0)
            tmp_image_grid_thw = torch.stack(tmp_image_grid_thw, dim=0)
        batch_pixel_values.append(tmp_pixel_values)
        batch_image_grid_thw.append(tmp_image_grid_thw)

        if len(tmp_pixel_values_videos) > 0:
            tmp_pixel_values_videos = torch.cat(tmp_pixel_values_videos, dim=0)
            tmp_video_grid_thw = torch.stack(tmp_video_grid_thw, dim=0)
        batch_pixel_values_videos.append(tmp_pixel_values_videos)
        batch_video_grid_thw.append(tmp_video_grid_thw)

    # Note that we might have empty list such as in `pixel_values_videos` if there is no video in the batch.
    # We will pop them out in `unsplit_visual_data`.
    return {
        **batch,
        "pixel_values": batch_pixel_values,
        "image_grid_thw": batch_image_grid_thw,
        "pixel_values_videos": batch_pixel_values_videos,
        "video_grid_thw": batch_video_grid_thw,
    }


def unsplit_visual_data(batch):
    """
    Opposite of `split_visual_data`. Merges a list of tensors in `batch["pixel_values"]`and others
    back into a single tensor along the first dimension.
    """
    if "pixel_values" in batch:
        non_empty_pixel_values = [pv for pv in batch["pixel_values"] if len(pv) > 0]
        non_empty_image_grid_thw = [
            thw for thw in batch["image_grid_thw"] if len(thw) > 0
        ]
        assert len(non_empty_pixel_values) == len(non_empty_image_grid_thw)
        if non_empty_pixel_values:
            batch["pixel_values"] = torch.cat(non_empty_pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(non_empty_image_grid_thw, dim=0)
        else:
            batch.pop("pixel_values")
            batch.pop("image_grid_thw")

    if "pixel_values_videos" in batch:
        non_empty_pixel_values_videos = [
            pv for pv in batch["pixel_values_videos"] if len(pv) > 0
        ]
        non_empty_video_grid_thw = [
            thw for thw in batch["video_grid_thw"] if len(thw) > 0
        ]
        if non_empty_pixel_values_videos:
            batch["pixel_values_videos"] = torch.cat(
                non_empty_pixel_values_videos, dim=0
            )
            batch["video_grid_thw"] = torch.cat(non_empty_video_grid_thw, dim=0)
        else:
            batch.pop("pixel_values_videos")
            batch.pop("video_grid_thw")

    return batch


def get_high_entropy_mask(
    entropies: torch.Tensor, mask: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

    Args:
        entropies (`torch.Tensor`):
            Tensor of shape (batch_size, seq_len) with per-token entropy values.
        mask (`torch.Tensor`):
            Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
        threshold (`float`):
            Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

    Returns:
        `torch.Tensor`:
            Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold and
            `False` otherwise.
    """
    non_pad_entropies = entropies[mask.bool()].float()
    if non_pad_entropies.numel() == 0:
        return torch.zeros_like(entropies, dtype=torch.bool)
    entropy_threshold = torch.quantile(non_pad_entropies, threshold)
    masked_entropies = entropies * mask.float()
    entropy_mask = masked_entropies >= entropy_threshold
    return entropy_mask & mask.bool()  # ensure padding tokens are always masked out


class Qwen3_VL_GRPOVLLMTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Should be a [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. Can be either: a single reward function, or list of
            reward functions, where each item can independently be any of the above types.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        processing_class: Union[PreTrainedTokenizerBase, ProcessorMixin],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
    ):
        # Models
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._assistant_prefix_ids = tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        ).input_ids
        self._user_prefix_ids = tokenizer(
            "<|im_start|>user\n", add_special_tokens=False
        ).input_ids
        self._system_prefix_ids = tokenizer(
            "<|im_start|>system\n", add_special_tokens=False
        ).input_ids

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        self.reward_func_names = [func.__name__ for func in reward_funcs]

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        assert self.vllm_mode == "colocate", "Only colocate mode is supported on MAST."
        self.vllm_gpu_memory_utilization = (
            args.vllm_gpu_memory_utilization
        )  # only applies to colocation mode
        self.vllm_tensor_parallel_size = (
            args.vllm_tensor_parallel_size
        )  # only applies to colocation mode
        self.loss_type = args.loss_type
        self.importance_sampling_level = args.importance_sampling_level
        self.top_entropy_quantile = args.top_entropy_quantile

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon

        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        self._search_log_dir: Optional[str] = None
        self._reward_debug_steps_set: Optional[set[int]] = None
        self._stage_debug_enabled = str(
            os.environ.get("GRPO_STAGE_DEBUG", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._stage_debug_every = max(
            1, int(os.environ.get("GRPO_STAGE_DEBUG_EVERY", "1") or 1)
        )
        self._stage_watchdog_sec = max(
            0, int(os.environ.get("GRPO_STAGE_WATCHDOG_SEC", "0") or 0)
        )
        self._stage_watchdog_file = None
        self._stage_watchdog_enabled = False
        self._logps_no_chunk = str(
            os.environ.get("GRPO_LOGPS_NO_CHUNK", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Latent improve reward resources
        self._latent_enabled = bool(args.use_latent_improve_reward)
        self._refine_token = str(getattr(args, "refine_token", "<REFINE>"))
        self._refine_token_count = 1
        self._refine_tokens = _build_refine_tokens(self._refine_token, self._refine_token_count)
        self._refine_rollout_depth = int(max(1, getattr(args, "refine_rollout_depth", 1)))
        vocab = tokenizer.get_vocab()
        self._refine_token_ids = [
            int(tokenizer.convert_tokens_to_ids(tok))
            for tok in self._refine_tokens
            if tok in vocab
        ]
        # Keep legacy attrs for backward compatibility.
        self._refine_token_id = int(self._refine_token_ids[0]) if self._refine_token_ids else -1
        self._refine_suffix = "".join(self._refine_tokens)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[latent] refine token config: "
                f"base={self._refine_token}, tokens={self._refine_tokens}, "
                f"ids={self._refine_token_ids}, rollout_depth={self._refine_rollout_depth}"
            )
        self._improve_reward_scale = float(args.improve_reward_scale)
        self._margin_reward_scale = float(
            getattr(args, "margin_reward_scale", args.improve_reward_scale)
        )
        self._use_sqr_latent_loss = bool(
            getattr(args, "use_sqr_latent_loss", False)
            and self._latent_enabled
        )
        self._sqr_latent_sigma = float(max(1e-6, getattr(args, "sqr_latent_sigma", 0.05)))
        self._sqr_latent_loss_weight = float(
            max(0.0, getattr(args, "sqr_latent_loss_weight", 1.0))
        )
        self._sqr_latent_clip_epsilon = float(
            max(1e-6, getattr(args, "sqr_latent_clip_epsilon", args.epsilon))
        )
        self._sqr_latent_train_depth = int(
            getattr(args, "sqr_latent_train_depth", -1)
        )
        self._sqr_latent_every_n_steps = int(
            max(1, getattr(args, "sqr_latent_every_n_steps", 1))
        )
        if int(os.environ.get("RANK", "0")) == 0 and self._use_sqr_latent_loss:
            print(
                "[latent][sqr] enabled: "
                f"sigma={self._sqr_latent_sigma}, "
                f"weight={self._sqr_latent_loss_weight}, "
                f"clip_eps={self._sqr_latent_clip_epsilon}, "
                f"train_depth={self._sqr_latent_train_depth}, "
                f"every_n_steps={self._sqr_latent_every_n_steps}"
            )
        self._use_infonce_latent_aux_loss = bool(
            getattr(args, "use_infonce_latent_aux_loss", False)
            and self._latent_enabled
        )
        self._infonce_latent_loss_weight = float(
            max(0.0, getattr(args, "infonce_latent_loss_weight", 1.0))
        )
        self._infonce_latent_temperature = float(
            max(1e-6, getattr(args, "infonce_latent_temperature", 0.1))
        )
        self._infonce_latent_mode = str(
            getattr(args, "infonce_latent_mode", "abs")
        ).strip().lower()
        if self._infonce_latent_mode not in {"abs", "delta"}:
            self._infonce_latent_mode = "abs"
        self._infonce_latent_train_depth = int(
            getattr(args, "infonce_latent_train_depth", -1)
        )
        self._infonce_latent_every_n_steps = int(
            max(1, getattr(args, "infonce_latent_every_n_steps", 1))
        )
        if int(os.environ.get("RANK", "0")) == 0 and self._use_infonce_latent_aux_loss:
            print(
                "[latent][infonce_aux] enabled: "
                f"weight={self._infonce_latent_loss_weight}, "
                f"temp={self._infonce_latent_temperature}, "
                f"mode={self._infonce_latent_mode}, "
                f"train_depth={self._infonce_latent_train_depth}, "
                f"every_n_steps={self._infonce_latent_every_n_steps}"
            )
        self._query_refine_temperature = float(
            os.environ.get("GRPO_QUERY_REFINE_TEMPERATURE", "0.1") or 0.1
        )
        if self._query_refine_temperature <= 0:
            self._query_refine_temperature = 0.1
        self._use_refine_gate = bool(args.use_refine_gate)
        self._use_query_embedder_path = bool(args.use_query_embedder_path)
        self._qfinal_pooling = str(getattr(args, "qfinal_pooling", "latent_last")).strip().lower()
        if self._qfinal_pooling not in {"latent_last", "mean"}:
            self._qfinal_pooling = "latent_last"
        self._qfinal_normalize = bool(getattr(args, "qfinal_normalize", True))
        self._query_embedder_max_length = int(max(8, getattr(args, "query_embedder_max_length", 128)))
        self._query_embedder_input_prefix = str(
            os.environ.get("QUERY_EMBEDDER_INPUT_PREFIX", "")
        ).strip()
        self._query_embedder_model = None
        self._query_embedder_tokenizer = None
        self._query_embedder_needs_param_gather = False
        self._query_meta_by_query: dict[str, tuple[int, str, str]] = {}
        self._video_id_to_row: dict[str, int] = {}
        self._query_embeddings = None
        self._video_embeddings = None
        self._video_meta_index_cache: dict[str, dict] = {}
        # Lightweight parameter probe to verify updates
        self._param_probe_param: Optional[torch.nn.Parameter] = None
        self._param_probe_name: Optional[str] = None
        self._param_probe_prev: Optional[torch.Tensor] = None
        self._param_probe_size: int = 1024
        self._last_probe_grad_abs_mean: Optional[float] = None
        self._last_probe_grad_norm: Optional[float] = None
        self._last_probe_grad_nonzero_frac: Optional[float] = None
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self._stage_debug_enabled = bool(self._stage_debug_enabled or args.search_debug)
        self._setup_stage_watchdog()
        if self._logps_no_chunk:
            self._stage_debug_log("logps:no_chunk_enabled")

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        else:
            # For deepspeed or non-distributed models, the reference model has been created
            self.ref_model = ref_model

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        # Cache latest reward-related metrics so we can optionally log them every step.
        self._last_reward_metrics: dict[str, dict[str, float]] = {"train": {}, "eval": {}}
        self._last_reward_metrics_step: dict[str, Optional[int]] = {
            "train": None,
            "eval": None,
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            assert self.vllm_mode == "colocate"
            # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                raise ValueError(
                    f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                    f"({self.accelerator.num_processes}) evenly."
                )

            if self.vllm_tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(
                            range(
                                i * self.vllm_tensor_parallel_size,
                                (i + 1) * self.vllm_tensor_parallel_size,
                            )
                        )
                        for i in range(
                            self.accelerator.num_processes
                            // self.vllm_tensor_parallel_size
                        )
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(self.accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

            if (
                self.max_prompt_length is not None
                and self.max_completion_length is not None
            ):
                max_model_len = self.max_prompt_length + self.max_completion_length
            else:
                max_model_len = None
            disable_custom_all_reduce = _env_flag(
                "VLLM_DISABLE_CUSTOM_ALL_REDUCE", False
            )
            enforce_eager = _env_flag("VLLM_ENFORCE_EAGER", False)
            if disable_custom_all_reduce or enforce_eager:
                print(
                    "[vllm] runtime flags: "
                    f"disable_custom_all_reduce={disable_custom_all_reduce}, "
                    f"enforce_eager={enforce_eager}"
                )
            _patch_vllm_qwen3_loader_for_soft_refine()
            self.llm = LLM(
                model=model.name_or_path,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_num_seqs=self.args.per_device_train_batch_size
                * self.vllm_tensor_parallel_size
                * self.args.steps_per_generation,
                max_model_len=max_model_len,
                distributed_executor_backend="external_launcher",
                # Feed identical seed for tp groups to ensure sampling results are the same across workers
                seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                max_num_batched_tokens=4096,
                disable_custom_all_reduce=disable_custom_all_reduce,
                enforce_eager=enforce_eager,
            )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = (
                -1
            )  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
            }
            cache_implementation = getattr(args, "cache_implementation", None)
            if cache_implementation is not None:
                generation_kwargs["cache_implementation"] = cache_implementation
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if self._latent_enabled:
            self._init_latent_resources()

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size
            * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        if version.parse(transformers.__version__) >= version.parse("4.52.0"):
            # from transformers 4.52.0, the `seed_worker` requires the `num_workers` and `rank` arguments
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
        else:
            dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=True,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: GRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw

        if video_grid_thw is not None and pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_grid_thw"] = video_grid_thw

        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        model_inputs["logits_to_keep"] = logits_to_keep + 1

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[
            :, -logits_to_keep:, :
        ]  # (B, logits_to_keep, H)
        return last_hidden_state

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        # used in Qwen2.5VL
        messages=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        # Chunk inputs into smaller batches to reduce memory peak
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        fn_t0 = time.perf_counter()
        self._stage_debug_log(
            "logps_fn:start",
            f"input_shape={tuple(input_ids.shape)} attn_shape={tuple(attention_mask.shape)} logits_to_keep={int(logits_to_keep)} batch_size={int(batch_size)}",
        )

        batch = {
            "messages": messages,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
            batch["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos
            batch["video_grid_thw"] = video_grid_thw

        if self._logps_no_chunk:
            model_inputs = {
                k: v for k, v in batch.items() if k != "messages"
            }
            model_inputs["logits_to_keep"] = logits_to_keep + 1
            has_image = (
                "pixel_values" in model_inputs
                and model_inputs["pixel_values"] is not None
            )
            has_video = (
                "pixel_values_videos" in model_inputs
                and model_inputs["pixel_values_videos"] is not None
            )
            self._stage_debug_log(
                "logps_fn:no_chunk_before_forward",
                f"input_shape={tuple(model_inputs['input_ids'].shape)} has_image={has_image} has_video={has_video}",
            )
            logits = model(**model_inputs).logits
            self._stage_debug_log(
                "logps_fn:no_chunk_after_forward",
                f"logits_shape={tuple(logits.shape)}",
            )
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature
            completion_ids = model_inputs["input_ids"][:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            entropies = None
            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
            self._stage_debug_log(
                "logps_fn:done",
                f"logps_shape={tuple(logps.shape)} entropy={'yes' if entropies is not None else 'no'} elapsed={time.perf_counter()-fn_t0:.3f}s",
            )
            return logps, entropies

        splitted_batch = split_visual_data(batch)
        num_chunks = input_ids.size(0) // batch_size
        self._stage_debug_log(
            "logps_fn:split_done",
            f"num_chunks={int(num_chunks)}",
        )
        chunked_batch = split_to_chunk(
            splitted_batch,
            num_chunks=num_chunks,
        )
        for chunk_idx, chunk in enumerate(chunked_batch):
            model_inputs = unsplit_visual_data(chunk)
            model_inputs.pop("messages")

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1
            has_image = (
                "pixel_values" in model_inputs
                and model_inputs["pixel_values"] is not None
            )
            has_video = (
                "pixel_values_videos" in model_inputs
                and model_inputs["pixel_values_videos"] is not None
            )
            local_in = model_inputs.get("input_ids")
            local_shape = tuple(local_in.shape) if torch.is_tensor(local_in) else ()
            self._stage_debug_log(
                "logps_fn:chunk_before_forward",
                f"chunk={chunk_idx} input_shape={local_shape} has_image={has_image} has_video={has_video}",
            )

            logits = model(**model_inputs).logits
            self._stage_debug_log(
                "logps_fn:chunk_after_forward",
                f"chunk={chunk_idx} logits_shape={tuple(logits.shape)}",
            )
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = model_inputs["input_ids"][:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        self._stage_debug_log(
            "logps_fn:done",
            f"logps_shape={tuple(logps.shape)} entropy={'yes' if entropies is not None else 'no'} elapsed={time.perf_counter()-fn_t0:.3f}s",
        )
        return logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    @staticmethod
    def _get_llm_vocab_size(llm_model) -> Optional[int]:
        cfg = getattr(llm_model, "config", None)
        if cfg is None:
            return None
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            return int(text_cfg.vocab_size)
        if hasattr(cfg, "vocab_size"):
            return int(cfg.vocab_size)
        return None

    @staticmethod
    def _build_vocab_adjust_candidates(tensor: torch.Tensor, vocab_size: int) -> list[torch.Tensor]:
        """
        Build fallback candidates for vocab-shaped tensors to match vLLM expected vocab size.
        Used only when direct load triggers vocab size assertions.
        """
        if tensor.ndim not in (1, 2) or vocab_size <= 0:
            return []
        out: list[torch.Tensor] = []
        base = tensor.detach()
        if base.is_cuda:
            base = base.cpu()
        original_shape = tuple(base.shape)
        # Guardrail: avoid catastrophic allocations (e.g. [151k, 151k]) on wrong axis.
        max_safe_elems = max(int(base.numel()) * 4, 200_000_000)

        def _adjust_dim(x: torch.Tensor, dim: int) -> Optional[torch.Tensor]:
            cur = x.size(dim)
            if cur == vocab_size:
                return x
            new_shape = list(x.shape)
            new_shape[dim] = vocab_size
            new_numel = int(np.prod(new_shape))
            if new_numel > max_safe_elems:
                return None
            if cur > vocab_size:
                return x.narrow(dim, 0, vocab_size).contiguous()
            pad_shape = list(x.shape)
            pad_shape[dim] = vocab_size - cur
            pad = x.new_zeros(pad_shape)
            return torch.cat([x, pad], dim=dim).contiguous()

        def _append_candidate(cand: Optional[torch.Tensor]) -> None:
            if cand is None:
                return
            shape = tuple(cand.shape)
            if shape == original_shape:
                return
            if any(tuple(c.shape) == shape for c in out):
                return
            out.append(cand)

        if base.ndim == 1:
            _append_candidate(_adjust_dim(base, 0))
            return out

        # Prefer the dimension whose size is closer to vocab_size.
        dims = [0, 1]
        dims.sort(key=lambda d: abs(int(base.size(d)) - int(vocab_size)))
        for dim in dims:
            _append_candidate(_adjust_dim(base, dim))

        # Also try transposed layouts in case vLLM loader expects vocab on the other axis.
        for cand in list(out):
            _append_candidate(cand.t().contiguous())
        return out

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        # For non-PEFT models, simply gather (if needed) and update each parameter individually.
        for name, param in self.model.named_parameters():
            name = self._fix_param_name_to_vllm(name)
            with gather_if_zero3([param]):
                assert self.vllm_mode == "colocate"
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                param_data = param.data
                try:
                    llm_model.load_weights([(name, param_data)])
                except AssertionError as exc:
                    # Handle vocab mismatch when training model was resized (e.g. added <REFINE>)
                    # but vLLM engine was initialized from base checkpoint vocab.
                    vocab_size = self._get_llm_vocab_size(llm_model)
                    is_vocab_param = ("embed_tokens" in name) or ("lm_head" in name)
                    if not (is_vocab_param and vocab_size is not None):
                        raise
                    if self.accelerator.is_main_process:
                        print(
                            "[vllm][warn] vocab mismatch during sync: "
                            f"{name} param_shape={tuple(param_data.shape)} "
                            f"vllm_vocab_size={vocab_size}"
                        )

                    candidates = self._build_vocab_adjust_candidates(param_data, vocab_size)
                    loaded = False
                    for cand in candidates:
                        try:
                            llm_model.load_weights([(name, cand)])
                            loaded = True
                            if self.accelerator.is_main_process:
                                print(
                                    "[vllm][warn] adjusted vocab-shaped weight for sync: "
                                    f"{name} {tuple(param_data.shape)} -> {tuple(cand.shape)} "
                                    f"(vocab_size={vocab_size})"
                                )
                            break
                        except AssertionError:
                            continue
                    if not loaded:
                        raise exc

        # Reset cache on vLLM
        assert self.vllm_mode == "colocate"
        self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                generation_batch = split_visual_data(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_to_chunk(
                    generation_batch,
                    num_chunks=self.args.steps_per_generation,
                )
                self._buffered_inputs = [
                    unsplit_visual_data(batch) for batch in generation_batches
                ]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, completions, solutions, problem_types, reward_meta=None):
        t0 = time.perf_counter()
        self._stage_debug_log(
            "rewards:start",
            f"local_batch={len(completions)} funcs={len(self.reward_funcs)}",
        )
        device = self.accelerator.device
        rewards_per_func = self._calculate_rewards_local(
            completions, solutions, problem_types, reward_meta=reward_meta
        )
        self._stage_debug_log(
            "rewards:local_done",
            f"shape={tuple(rewards_per_func.shape)} elapsed={time.perf_counter()-t0:.3f}s",
        )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        t_gather = time.perf_counter()
        self._stage_debug_log(
            "rewards:gather_start",
            f"shape={tuple(rewards_per_func.shape)}",
        )
        rewards_per_func = gather(rewards_per_func)
        self._stage_debug_log(
            "rewards:gather_done",
            f"shape={tuple(rewards_per_func.shape)} gather_elapsed={time.perf_counter()-t_gather:.3f}s total_elapsed={time.perf_counter()-t0:.3f}s",
        )
        return rewards_per_func

    def _calculate_rewards_local(self, completions, solutions, problem_types, reward_meta=None):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(
            len(completions),
            len(self.reward_funcs),
            device=device,
        )
        for i, (reward_func, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                output_reward_func = reward_func(
                    completions=completions,
                    solutions=solutions,
                    problem_types=problem_types,
                    reward_meta=reward_meta,
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )
        return rewards_per_func

    def _prepare_vllm_inputs(
        self,
        messages,
        prompts_text,
        image_inputs,
        video_inputs,
        video_kwargs,
    ):
        if not hasattr(self, "_vllm_final_prompt_dumped"):
            self._vllm_final_prompt_dumped = False

        def _maybe_dump_vllm_final_prompt(
            prompt_text: str,
            mm_data: dict[str, Any],
            mm_kwargs: Optional[dict[str, Any]],
            prompt_idx: int,
        ) -> None:
            if not _env_flag("VLLM_DUMP_FINAL_PROMPT", False):
                return
            if self._vllm_final_prompt_dumped:
                return
            if not isinstance(mm_data, dict) or "video" not in mm_data:
                return
            try:
                videos = []
                video_metas = []
                for item in mm_data.get("video", []):
                    if isinstance(item, tuple) and len(item) == 2:
                        videos.append(item[0])
                        video_metas.append(item[1] if isinstance(item[1], dict) else {})
                    else:
                        videos.append(item)
                        video_metas.append({})
                if not videos:
                    return

                proc_kwargs: dict[str, Any] = dict(mm_kwargs or {})
                proc_kwargs.setdefault("do_resize", False)
                proc_inputs: dict[str, Any] = {
                    "text": [prompt_text],
                    "videos": videos,
                    "return_tensors": "pt",
                    "padding": True,
                }
                if any(bool(m) for m in video_metas):
                    proc_inputs["video_metadata"] = video_metas
                proc_inputs.update(proc_kwargs)

                dbg_inputs = self.processing_class(**proc_inputs)
                input_ids = dbg_inputs.get("input_ids", None)
                if input_ids is None:
                    print("[search][dbg][warn] vllm final prompt dump failed: no input_ids")
                    return
                tokenizer = getattr(self.processing_class, "tokenizer", None)
                if tokenizer is None:
                    print("[search][dbg][warn] vllm final prompt dump failed: tokenizer missing")
                    return

                decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                second_tokens = re.findall(r"<\d+(?:\.\d+)? seconds>", decoded)
                print(
                    "[search][dbg] vllm final prompt dump "
                    f"prompt_idx={prompt_idx} decoded_chars={len(decoded)} "
                    f"second_tokens={len(second_tokens)}"
                )
                self._stage_debug_log(
                    "search:vllm_prompt_dump",
                    f"prompt_idx={prompt_idx} decoded_chars={len(decoded)} second_tokens={len(second_tokens)}",
                )
                if second_tokens:
                    head = second_tokens[:8]
                    tail = second_tokens[-8:] if len(second_tokens) > 8 else []
                    print(f"[search][dbg] seconds_head={head}")
                    self._stage_debug_log("search:vllm_seconds_head", str(head))
                    if tail:
                        print(f"[search][dbg] seconds_tail={tail}")
                        self._stage_debug_log("search:vllm_seconds_tail", str(tail))
                    first_pos = decoded.find(second_tokens[0])
                    if first_pos >= 0:
                        st = max(0, first_pos - 120)
                        ed = min(len(decoded), first_pos + 720)
                        snippet = decoded[st:ed].replace("\n", "\\n")
                        print(f"[search][dbg] final_prompt_snippet={snippet}")
                        self._stage_debug_log(
                            "search:vllm_prompt_snippet", snippet[:800]
                        )
                else:
                    print("[search][dbg][warn] no <... seconds> tokens found in decoded prompt")
                    self._stage_debug_log(
                        "search:vllm_prompt_dump_warn", "no <... seconds> tokens found"
                    )
            except Exception as exc:
                print(f"[search][dbg][warn] vllm final prompt dump failed: {exc}")
                self._stage_debug_log("search:vllm_prompt_dump_warn", str(exc))
            finally:
                self._vllm_final_prompt_dumped = True

        # Restore the image_inputs, video_inputs, and video_kwargs that were assembled into a batch through process_vision_info
        # back into a list, where each item corresponds to a vLLM input containing prompt, multi_modal_data, and mm_processor_kwargs.
        vllm_inputs = []
        image_idx = 0
        video_idx = 0

        for message, prompt in zip(messages, prompts_text):
            tmp_image_inputs = []
            tmp_video_inputs = []
            for msg in message:
                if isinstance(msg["content"], list):
                    for ele in msg["content"]:
                        if "image" in ele or "image_url" in ele:
                            tmp_image_inputs.append(image_inputs[image_idx])
                            image_idx += 1
                        elif "video" in ele:
                            tmp_video_inputs.append(video_inputs[video_idx])
                            video_idx += 1

            tmp_llm_inputs = {"prompt": prompt}

            # if contains multi-modal data
            tmp_mm_data = {}
            if len(tmp_image_inputs) > 0:
                tmp_mm_data["image"] = tmp_image_inputs
            if len(tmp_video_inputs) > 0:
                tmp_mm_data["video"] = tmp_video_inputs
            if tmp_mm_data:
                tmp_llm_inputs["multi_modal_data"] = tmp_mm_data
                if self.args.search_debug:
                    video_tags = (
                        "<video>",
                        "<|video|>",
                        "<|video_1|>",
                        "<video_1>",
                        "<|video_pad|>",
                    )
                    image_tags = (
                        "<image>",
                        "<|image|>",
                        "<|image_1|>",
                        "<image_1>",
                        "<|image_pad|>",
                    )
                    missing = []
                    if "video" in tmp_mm_data and not any(
                        tag in prompt for tag in video_tags
                    ):
                        missing.append("video")
                    if "image" in tmp_mm_data and not any(
                        tag in prompt for tag in image_tags
                    ):
                        missing.append("image")
                    if missing:
                        print(
                            f"[search][warn] missing placeholders for {missing} in prompt idx {len(vllm_inputs)}"
                        )
                        print(f"[search][warn] prompt:\n{prompt}")
                    if (
                        len(vllm_inputs) < 2
                        and "video" in tmp_mm_data
                        and len(tmp_mm_data["video"]) > 0
                    ):
                        first_video = tmp_mm_data["video"][0]
                        if isinstance(first_video, tuple) and len(first_video) == 2:
                            video_tensor, video_meta = first_video
                            shape = (
                                tuple(video_tensor.shape)
                                if hasattr(video_tensor, "shape")
                                else None
                            )
                            fps = None
                            total = None
                            idx_len = None
                            idx_last = None
                            dur = None
                            if isinstance(video_meta, dict):
                                fps = video_meta.get("fps", video_meta.get("raw_fps", None))
                                total = video_meta.get("total_num_frames", None)
                                indices = video_meta.get("frames_indices", None)
                                if isinstance(indices, list):
                                    idx_len = len(indices)
                                    idx_last = indices[-1] if indices else None
                                if fps and total:
                                    try:
                                        dur = float(total) / float(fps)
                                    except Exception:
                                        dur = None
                            print(
                                "[search][dbg] vllm video payload "
                                f"prompt_idx={len(vllm_inputs)} shape={shape} "
                                f"fps={fps} total={total} idx_len={idx_len} idx_last={idx_last} "
                                f"duration_sec={dur} video_kwargs={video_kwargs}"
                            )
                            _maybe_dump_vllm_final_prompt(
                                prompt_text=prompt,
                                mm_data=tmp_mm_data,
                                mm_kwargs=video_kwargs,
                                prompt_idx=len(vllm_inputs),
                            )
                        else:
                            shape = (
                                tuple(first_video.shape)
                                if hasattr(first_video, "shape")
                                else None
                            )
                            print(
                                "[search][dbg] vllm video payload "
                                f"prompt_idx={len(vllm_inputs)} non_tuple_video shape={shape} "
                                f"video_kwargs={video_kwargs}"
                            )

            # if contains mm_processor_kwargs
            if len(tmp_video_inputs) > 0:
                tmp_llm_inputs["mm_processor_kwargs"] = video_kwargs

            vllm_inputs.append(tmp_llm_inputs)

        if image_inputs:
            assert image_idx == len(image_inputs), (
                f"Not all images were processed: "
                f"image_idx {image_idx} != len(image_inputs) {len(image_inputs)}"
            )

        if video_inputs:
            assert video_idx == len(video_inputs), (
                f"Not all videos were processed: "
                f"video_idx {video_idx} != len(video_inputs) {len(video_inputs)}"
            )
        return vllm_inputs

    def _build_assistant_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a mask for assistant tokens based on chat template prefixes."""
        seq_len = input_ids.size(1)

        def _match_positions(pattern_ids):
            if seq_len < len(pattern_ids):
                return [
                    torch.tensor([], device=input_ids.device, dtype=torch.long)
                    for _ in range(input_ids.size(0))
                ]
            pattern = torch.tensor(pattern_ids, device=input_ids.device)
            wins = input_ids.unfold(dimension=-1, size=pattern.numel(), step=1)
            match_mask = (wins == pattern).all(dim=-1)
            return [
                torch.nonzero(match_mask[b], as_tuple=False).squeeze(1)
                for b in range(match_mask.size(0))
            ]

        asst_pos = _match_positions(self._assistant_prefix_ids)
        user_pos = _match_positions(self._user_prefix_ids)
        sys_pos = _match_positions(self._system_prefix_ids)

        assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            asst_list = asst_pos[b_idx].tolist()
            if not asst_list:
                continue
            boundary = sorted(
                set(
                    asst_list
                    + user_pos[b_idx].tolist()
                    + sys_pos[b_idx].tolist()
                )
            )
            for pos in asst_list:
                start_idx = pos + len(self._assistant_prefix_ids)
                next_candidates = [p for p in boundary if p > pos]
                end_idx = min(next_candidates) if next_candidates else seq_len
                if start_idx < end_idx:
                    assistant_mask[b_idx, start_idx:end_idx] = True
        return assistant_mask

    def _generate_text_batch(
        self,
        messages: list[list[dict[str, Any]]],
        max_new_tokens: Optional[int] = None,
    ) -> tuple[list[str], list[torch.Tensor]]:
        device = self.accelerator.device
        norm_messages = [self._normalize_messages(m) for m in messages]
        prompts_text = [
            self.processing_class.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in norm_messages
        ]
        if self.args.search_debug:
            video_tags = (
                "<video>",
                "<|video|>",
                "<|video_1|>",
                "<video_1>",
                "<|video_pad|>",
            )
            image_tags = (
                "<image>",
                "<|image|>",
                "<|image_1|>",
                "<image_1>",
                "<|image_pad|>",
            )
            for idx, prompt in enumerate(prompts_text[:2]):
                v_count = sum(prompt.count(tag) for tag in video_tags)
                i_count = sum(prompt.count(tag) for tag in image_tags)
                print(
                    f"[search][dbg] prompt[{idx}] len={len(prompt)} "
                    f"video_tags={v_count} image_tags={i_count}"
                )
        image_inputs, packed_video_inputs, video_kwargs = cached_process_vision_info(
            norm_messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if packed_video_inputs is not None:
            video_inputs, video_metadatas = zip(*packed_video_inputs)
            video_inputs, video_metadatas = (
                list(video_inputs),
                list(video_metadatas),
            )
        else:
            video_inputs = None
            video_metadatas = None

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
            else:
                guided_decoding = None

            generation_kwargs = {
                "n": 1,
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": self.max_completion_length
                if max_new_tokens is None
                else max_new_tokens,
                "guided_decoding": guided_decoding,
            }
            if self.args.generation_kwargs is not None:
                generation_kwargs.update(self.args.generation_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)

            vllm_inputs = self._prepare_vllm_inputs(
                norm_messages,
                prompts_text,
                image_inputs,
                packed_video_inputs,
                video_kwargs,
            )

            if self.vllm_tensor_parallel_size > 1:
                orig_size = len(vllm_inputs)
                gathered_vllm_inputs = [
                    None for _ in range(self.vllm_tensor_parallel_size)
                ]
                torch.distributed.all_gather_object(
                    gathered_vllm_inputs, vllm_inputs, group=self.tp_group
                )
                all_vllm_inputs = [
                    p for sublist in gathered_vllm_inputs for p in sublist
                ]
            else:
                all_vllm_inputs = vllm_inputs

            all_outputs = self.llm.generate(
                all_vllm_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if len(all_outputs) != len(all_vllm_inputs) and self.args.search_debug:
                print(
                    f"[search][warn] vLLM outputs length {len(all_outputs)} "
                    f"!= inputs length {len(all_vllm_inputs)}"
                )

            completion_ids = []
            for req_output in all_outputs:
                if req_output.outputs:
                    completion_ids.append(req_output.outputs[0].token_ids)
                else:
                    completion_ids.append([])
                    if self.args.search_debug:
                        print("[search][warn] vLLM returned empty output for a request")

            if self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(
                    local_rank_in_group * orig_size,
                    (local_rank_in_group + 1) * orig_size,
                )
                completion_ids = completion_ids[tp_slice]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        else:
            prompt_inputs = self.processing_class(
                text=prompts_text,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors="pt",
                padding=True,
            )
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            generation_config = self.generation_config
            if max_new_tokens is not None:
                generation_config = copy.deepcopy(self.generation_config)
                generation_config.max_new_tokens = max_new_tokens
            prompt_completion_ids = self.model.generate(
                **prompt_inputs, generation_config=generation_config
            )
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = [
                ids[prompt_length:].detach() for ids in prompt_completion_ids
            ]

        # Guard against mismatched lengths (e.g., missing vLLM outputs)
        target_len = len(messages)
        if len(completion_ids) != target_len:
            if self.args.search_debug:
                print(
                    f"[search][warn] completion_ids length {len(completion_ids)} "
                    f"!= messages length {target_len}"
                )
                img_count = 0 if image_inputs is None else len(image_inputs)
                vid_count = 0 if video_inputs is None else len(video_inputs)
                print(
                    f"[search][warn] mm counts: images={img_count} videos={vid_count}"
                )
            if len(completion_ids) < target_len:
                missing = target_len - len(completion_ids)
                completion_ids.extend(
                    [
                        torch.tensor([], device=device, dtype=torch.long)
                        for _ in range(missing)
                    ]
                )
            else:
                completion_ids = completion_ids[:target_len]

        completion_ids_padded = pad(completion_ids, padding_value=self.pad_token_id)
        completions_text = self.processing_class.batch_decode(
            completion_ids_padded, skip_special_tokens=False
        )
        completions_text = [_sanitize_generated_text(text) for text in completions_text]
        return completions_text, completion_ids

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        if self.args.use_search:
            return self._generate_and_score_completions_search(inputs)
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        solutions = [example["response"] for example in inputs]  # gt solutions
        problem_types = [example["problem_type"] for example in inputs]
        messages = [example["messages"] for example in inputs]

        prompts_text = [
            self.processing_class.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in messages
        ]
        image_inputs, packed_video_inputs, video_kwargs = cached_process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if packed_video_inputs is not None:
            video_inputs, video_metadatas = zip(*packed_video_inputs)
            video_inputs, video_metadatas = (
                list(video_inputs),
                list(video_metadatas),
            )
        else:
            video_inputs = None
            video_metadatas = None

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            **video_kwargs,
        )

        # move to device
        if "pixel_values" in prompt_inputs:
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(device)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].to(device)
        if "pixel_values_videos" in prompt_inputs:
            prompt_inputs["pixel_values_videos"] = prompt_inputs[
                "pixel_values_videos"
            ].to(device)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].to(device)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"].to(device),
            prompt_inputs["attention_mask"].to(device),
        )

        assert prompt_ids.shape[-1] <= self.max_prompt_length, print(
            f"Prompt length {prompt_ids.shape[-1]} exceeds max_prompt_length {self.max_prompt_length}"
        )

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            assert self.vllm_mode == "colocate"
            if self.guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
            else:
                guided_decoding = None

            generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": self.max_completion_length,
                "guided_decoding": guided_decoding,
            }
            if self.args.generation_kwargs is not None:
                generation_kwargs.update(self.args.generation_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)

            # convert input data to vLLM inputs
            vllm_inputs = self._prepare_vllm_inputs(
                messages,
                prompts_text,
                image_inputs,
                packed_video_inputs,
                video_kwargs,
            )

            if self.vllm_tensor_parallel_size > 1:
                # Gather vLLM inputs from all ranks in the TP group and flatten.
                # Each rank starts with its own vLLM inputs; after gathering, all ranks see the full group set.
                orig_size = len(vllm_inputs)
                gathered_vllm_inputs = [
                    None for _ in range(self.vllm_tensor_parallel_size)
                ]
                torch.distributed.all_gather_object(
                    gathered_vllm_inputs, vllm_inputs, group=self.tp_group
                )
                all_vllm_inputs = [
                    p for sublist in gathered_vllm_inputs for p in sublist
                ]
            else:
                all_vllm_inputs = vllm_inputs

            with profiling_context(self, "vLLM.generate"):
                all_outputs = self.llm.generate(
                    all_vllm_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

            completion_ids = [
                output.token_ids
                for outputs in all_outputs
                for output in outputs.outputs
            ]

            if self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(
                    local_rank_in_group * orig_size,
                    (local_rank_in_group + 1) * orig_size,
                )
                completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = (
                    prompt_ids,
                    prompt_mask,
                )
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            generate_every = (
                self.args.steps_per_generation * self.num_iterations
            )  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    # used in Qwen2.5VL
                    messages=messages,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_values_videos=prompt_inputs.get("pixel_values_videos"),
                    video_grid_thw=prompt_inputs.get("video_grid_thw"),
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        # used in Qwen2.5VL
                        messages=messages,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_values_videos=prompt_inputs.get("pixel_values_videos"),
                        video_grid_thw=prompt_inputs.get("video_grid_thw"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                # used in Qwen2.5VL
                                messages=messages,
                                pixel_values=prompt_inputs.get("pixel_values"),
                                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                                pixel_values_videos=prompt_inputs.get(
                                    "pixel_values_videos"
                                ),
                                video_grid_thw=prompt_inputs.get("video_grid_thw"),
                            )
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            completions_text,
            solutions,
            problem_types,
        )

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        advantages = advantages / (std_grouped_rewards + 1e-4)
        advantages_full = advantages

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        advantages = advantages[process_slice]
        rewards_local_for_adv = rewards[process_slice]
        mean_grouped_rewards_local = mean_grouped_rewards[process_slice]
        std_grouped_rewards_local = std_grouped_rewards[process_slice]
        rewards_per_func_local = rewards_per_func[process_slice]

        reward_debug_data = None
        if self._should_reward_debug():
            max_samples = int(self.args.reward_debug_max_samples or 0)
            if max_samples <= 0:
                max_samples = len(completions_text)
            reward_debug_data = []
            upper = min(len(completions_text), max_samples, rewards_per_func_local.size(0))
            for idx in range(upper):
                func_values = {}
                func_values_weighted = {}
                for i, name in enumerate(self.reward_func_names):
                    value = float(rewards_per_func_local[idx, i].item())
                    weight = float(self.reward_weights[i].detach().cpu().item())
                    func_values[name] = value
                    func_values_weighted[name] = value * weight

                completion_text = str(completions_text[idx] or "")
                answer_pred = _extract_answer_tag(completion_text)
                pred_start, pred_end = _extract_start_end(completion_text)
                solution_text = str(solutions[idx] or "")
                answer_expected = _extract_answer_tag(solution_text)
                if not answer_expected:
                    boxed_hits = re.findall(
                        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution_text
                    )
                    if boxed_hits:
                        answer_expected = (
                            boxed_hits[-1].strip().lower().replace(" ", "_")
                        )
                answer_binary_correct = bool(
                    answer_pred and answer_expected and answer_pred == answer_expected
                )

                query_text = str(inputs[idx].get("query", "") or "")
                if not query_text:
                    query_text = prompts_text[idx]
                gt_start, gt_end = _normalize_gt_span(inputs[idx].get("gt_time", None))
                time_iou = _compute_temporal_iou(pred_start, pred_end, gt_start, gt_end)

                reward_debug_data.append(
                    {
                        "idx": idx,
                        "qid": str(inputs[idx].get("qid", "") or f"idx_{idx}"),
                        "query": query_text,
                        "gt_video": str(inputs[idx].get("gt_video", "") or ""),
                        "completion_text": completion_text,
                        "solution": solution_text,
                        "problem_type": str(problem_types[idx] or ""),
                        "reward_total": float(rewards_local_for_adv[idx].item()),
                        "reward_funcs": func_values,
                        "reward_funcs_weighted": func_values_weighted,
                        "answer_pred": answer_pred,
                        "answer_expected": answer_expected,
                        "answer_binary_correct": answer_binary_correct,
                        "pred_start": pred_start,
                        "pred_end": pred_end,
                        "gt_start": gt_start,
                        "gt_end": gt_end,
                        "iou": time_iou,
                        "reward_total_global": float(
                            rewards_local_for_adv[idx].detach().cpu().item()
                        ),
                        "reward_group_mean": float(
                            mean_grouped_rewards_local[idx].detach().cpu().item()
                        ),
                        "reward_group_std": float(
                            std_grouped_rewards_local[idx].detach().cpu().item()
                        ),
                        "advantage": float(advantages[idx].detach().cpu().item()),
                    }
                )
            # Keep length aligned with local batch for downstream permutation.
            if len(reward_debug_data) < len(completions_text):
                reward_debug_data.extend(
                    [{"idx": None} for _ in range(len(completions_text) - len(reward_debug_data))]
                )

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Debug reward/advantage stats (global, before slicing)
        self._metrics[mode]["debug/reward_mean"].append(rewards.mean().item())
        self._metrics[mode]["debug/reward_std"].append(rewards.std().item())
        self._metrics[mode]["debug/reward_min"].append(rewards.min().item())
        self._metrics[mode]["debug/reward_max"].append(rewards.max().item())
        self._metrics[mode]["debug/group_reward_std_mean"].append(
            std_grouped_rewards.mean().item()
        )
        self._metrics[mode]["debug/group_reward_std_min"].append(
            std_grouped_rewards.min().item()
        )
        self._metrics[mode]["debug/group_reward_std_max"].append(
            std_grouped_rewards.max().item()
        )
        self._metrics[mode]["debug/adv_mean"].append(advantages_full.mean().item())
        self._metrics[mode]["debug/adv_abs_mean"].append(
            advantages_full.abs().mean().item()
        )
        self._metrics[mode]["debug/adv_std"].append(advantages_full.std().item())
        self._metrics[mode]["debug/adv_min"].append(advantages_full.min().item())
        self._metrics[mode]["debug/adv_max"].append(advantages_full.max().item())
        self._metrics[mode]["debug/adv_zero_frac"].append(
            (advantages_full.abs() < 1e-9).float().mean().item()
        )

        # Answer diversity stats (per batch)
        # reward_meta exists in search-specific path; for generic path, parse from completion text.
        answers = []
        if "reward_meta" in locals() and reward_meta is not None:
            for meta in reward_meta:
                turns = meta.get("turns", [])
                ans = ""
                if turns:
                    ans = turns[-1].get("answer") or ""
                answers.append(ans)
        else:
            for text in completions_text:
                ans = _extract_answer_tag(str(text or ""))
                answers.append(ans if ans in {"matched", "not_matched"} else "")
        if answers:
            matched_rate = sum(1 for a in answers if a == "matched") / len(answers)
            not_matched_rate = sum(1 for a in answers if a == "not_matched") / len(answers)
            empty_rate = sum(1 for a in answers if a == "") / len(answers)
            group_count = max(int(self.num_generations), 1)
            diverse = 0
            total_groups = 0
            for i in range(0, len(answers), group_count):
                grp = answers[i : i + group_count]
                if not grp:
                    continue
                total_groups += 1
                if len(set(grp)) > 1:
                    diverse += 1
            diverse_rate = diverse / total_groups if total_groups else 0.0
            self._metrics[mode]["debug/answer_matched_rate"].append(matched_rate)
            self._metrics[mode]["debug/answer_not_matched_rate"].append(not_matched_rate)
            self._metrics[mode]["debug/answer_empty_rate"].append(empty_rate)
            self._metrics[mode]["debug/answer_diverse_group_rate"].append(diverse_rate)

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor(
            [ids[-1] not in eos_and_pad for ids in completion_ids], device=device
        )
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(
            agg_is_truncated.float().mean().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        output = {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_values_videos" in prompt_inputs:
            output["pixel_values_videos"] = prompt_inputs["pixel_values_videos"]
        if "video_grid_thw" in prompt_inputs:
            output["video_grid_thw"] = prompt_inputs["video_grid_thw"]
        if reward_debug_data:
            output["reward_debug"] = reward_debug_data
        return output

    def _retrieve_topk(
        self,
        queries: list[str],
        rank_for_ids: Optional[list[str]] = None,
    ) -> tuple[list[list[str]], list[list[float]], Optional[list[int]]]:
        if not self.args.retriever_url:
            raise ValueError("retriever_url must be set when use_search is enabled.")
        payload = {"queries": queries, "topk": self.args.search_topk}
        if self.args.search_rank_k and rank_for_ids is not None:
            payload["rank_for_ids"] = rank_for_ids
            payload["rank_k"] = int(self.args.search_rank_k)
        try:
            resp = requests.post(self.args.retriever_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("ids", [])
            scores = data.get("scores", [])
            ranks = data.get("ranks")
            rank_for_ids_echo = data.get("rank_for_ids_echo")
            req_id = data.get("rid")
            if ranks is not None and len(ranks) != len(queries):
                raise RuntimeError(
                    "Retriever ranks length mismatch: "
                    f"len(ranks)={len(ranks)} len(queries)={len(queries)} "
                    f"rid={req_id}"
                )
            if (
                rank_for_ids is not None
                and rank_for_ids_echo is not None
                and len(rank_for_ids_echo) != len(queries)
            ):
                raise RuntimeError(
                    "Retriever rank_for_ids_echo length mismatch: "
                    f"len(rank_for_ids_echo)={len(rank_for_ids_echo)} "
                    f"len(queries)={len(queries)} rid={req_id}"
                )
            if rank_for_ids is not None and rank_for_ids_echo is not None:
                for i, (sent_id, echo_id) in enumerate(
                    zip(rank_for_ids, rank_for_ids_echo)
                ):
                    if sent_id != echo_id:
                        raise RuntimeError(
                            "Retriever rank_for_ids echo mismatch: "
                            f"idx={i} sent={sent_id} echo={echo_id} rid={req_id}"
                        )
            if self.args.search_debug and req_id is not None:
                print(f"[search][dbg] retriever rid={req_id}")
            return ids, scores, ranks
        except Exception as exc:
            if self.args.search_debug:
                print(f"[search][error] retriever request failed: {exc}")
            return ([[] for _ in queries], [[] for _ in queries], None)

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "environment":
                role = "user"
            normalized.append({"role": role, "content": msg.get("content")})
        return normalized

    def _get_search_log_dir(self) -> str:
        if self._search_log_dir:
            return self._search_log_dir
        base_dir = os.environ.get(
            "SEARCH_LOG_BASE_DIR",
            os.path.join(os.environ.get("VIDEOSEARCH_OUTPUT_ROOT", "outputs"), "search_logs"),
        )
        run_id = (
            os.environ.get("SEARCH_LOG_RUN_ID")
            or os.environ.get("WANDB_RUN_ID")
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        log_dir = os.path.join(base_dir, f"run_{run_id}")
        os.makedirs(log_dir, exist_ok=True)
        self._search_log_dir = log_dir
        return log_dir

    def _stage_debug_active(self) -> bool:
        if not self._stage_debug_enabled:
            return False
        step = int(getattr(self.state, "global_step", -1))
        if step < 0:
            return True
        return (step % self._stage_debug_every) == 0

    def _stage_debug_log(self, stage: str, extra: Optional[str] = None) -> None:
        if not self._stage_debug_active():
            return
        rank = int(getattr(self.accelerator, "process_index", os.environ.get("RANK", 0)))
        step = int(getattr(self.state, "global_step", -1))
        gen_step = int(getattr(self, "_step", -1))
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = f"[stage][{ts}] rank={rank} step={step} gen_step={gen_step} {stage}"
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)
        try:
            log_dir = self._get_search_log_dir()
            rank_dir = os.path.join(log_dir, f"rank{rank}")
            os.makedirs(rank_dir, exist_ok=True)
            with open(os.path.join(rank_dir, "stage_trace.log"), "a", encoding="utf-8") as fp:
                fp.write(msg + "\n")
        except Exception:
            pass

    def _setup_stage_watchdog(self) -> None:
        if self._stage_watchdog_enabled or self._stage_watchdog_sec <= 0:
            return
        rank = int(getattr(self.accelerator, "process_index", os.environ.get("RANK", 0)))
        try:
            debug_dir = os.path.join(self.args.output_dir, "deadlock_debug")
            os.makedirs(debug_dir, exist_ok=True)
            trace_path = os.path.join(debug_dir, f"rank{rank}.stacktrace.log")
            self._stage_watchdog_file = open(trace_path, "a", encoding="utf-8", buffering=1)
            faulthandler.enable(file=self._stage_watchdog_file, all_threads=True)
            faulthandler.dump_traceback_later(
                timeout=self._stage_watchdog_sec,
                repeat=True,
                file=self._stage_watchdog_file,
            )
            self._stage_watchdog_enabled = True
            self._stage_debug_log(
                "watchdog_enabled",
                f"timeout={self._stage_watchdog_sec}s trace_file={trace_path}",
            )
        except Exception as exc:
            print(f"[stage][warn] failed to enable watchdog: {exc}", flush=True)

    def _parse_reward_debug_steps(self) -> set[int]:
        if self._reward_debug_steps_set is not None:
            return self._reward_debug_steps_set
        steps_raw = self.args.reward_debug_steps
        steps_set: set[int] = set()
        if steps_raw:
            for part in str(steps_raw).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    steps_set.add(int(part))
                except ValueError:
                    continue
        self._reward_debug_steps_set = steps_set
        return steps_set

    def _should_reward_debug(self) -> bool:
        if not self.args.reward_debug:
            return False
        step = getattr(self.state, "global_step", -1)
        if step < 0:
            return False
        steps_set = self._parse_reward_debug_steps()
        if steps_set:
            return step in steps_set
        every = int(self.args.reward_debug_every or 1)
        if every <= 0:
            every = 1
        return (step % every) == 0

    def _get_debug_group_context(self, idx: int) -> tuple[int, int, int, int, str]:
        rank = getattr(self.accelerator, "process_index", 0)
        group_id = idx // max(int(self.num_generations), 1)
        step = getattr(self.state, "global_step", -1)
        gen_step = getattr(self, "_step", -1)
        log_dir = self._get_search_log_dir()
        group_dir = os.path.join(
            log_dir, f"rank{rank}", f"step{step}", f"group{group_id}"
        )
        os.makedirs(group_dir, exist_ok=True)
        return rank, step, gen_step, group_id, group_dir

    def _append_rollout_event(self, idx: int, event: str, payload: dict[str, Any]) -> None:
        rank, step, gen_step, group_id, group_dir = self._get_debug_group_context(idx)
        log_path = os.path.join(group_dir, f"idx{idx}.rollout.jsonl")
        record = {
            "event": event,
            "rank": rank,
            "step": step,
            "gen_step": gen_step,
            "idx": idx,
            "group": group_id,
        }
        record.update(payload)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_merged_group_reward_debug(
        self, reward_debug: list[dict[str, Any]]
    ) -> None:
        if not self.args.reward_debug or not reward_debug:
            return
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return

        rank = int(getattr(self.accelerator, "process_index", 0))
        world_size = int(torch.distributed.get_world_size())
        step = int(getattr(self.state, "global_step", -1))
        gen_step = int(getattr(self, "_step", -1))

        local_payload = []
        for item in reward_debug:
            idx = item.get("idx")
            if idx is None:
                continue
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            row = dict(item)
            row["src_rank"] = rank
            row["src_idx"] = idx
            local_payload.append(row)

        gathered_payloads: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_payloads, local_payload)

        if rank != 0:
            return

        merged_rows: list[dict[str, Any]] = []
        for rows in gathered_payloads:
            if rows:
                merged_rows.extend(rows)
        if not merged_rows:
            return

        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in merged_rows:
            key = (
                str(row.get("qid", "") or ""),
                str(row.get("query", "") or ""),
                str(row.get("gt_video", "") or ""),
            )
            grouped[key].append(row)

        log_dir = self._get_search_log_dir()
        merged_dir = os.path.join(log_dir, "merged", f"step{step}")
        os.makedirs(merged_dir, exist_ok=True)
        fallback_idx = 0
        for (qid, query, gt_video), rows in grouped.items():
            rows = sorted(
                rows,
                key=lambda x: (
                    int(x.get("src_rank", 0)),
                    int(x.get("batch_row_idx", -1)),
                    int(x.get("src_idx", -1)),
                ),
            )
            safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid).strip("._")
            if not safe_qid:
                safe_qid = f"nogid_{fallback_idx}"
                fallback_idx += 1
            out_path = os.path.join(merged_dir, f"{safe_qid}.group_rollout.jsonl")
            payload = {
                "event": "group_reward_merged",
                "step": step,
                "gen_step": gen_step,
                "qid": qid,
                "query": query,
                "gt_video": gt_video,
                "num_rollouts": len(rows),
                "rollouts": rows,
            }
            with open(out_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_reward_debug_log(self, idx: int, payload: dict[str, Any]) -> None:
        if not self.args.reward_debug:
            return
        rank, step, gen_step, group_id, group_dir = self._get_debug_group_context(idx)
        log_path = os.path.join(group_dir, f"idx{idx}.reward.jsonl")
        record = {
            "rank": rank,
            "step": step,
            "gen_step": gen_step,
            "idx": idx,
            "group": group_id,
        }
        record.update(payload)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._append_rollout_event(idx, "reward", payload)

    def _write_search_debug_log(
        self,
        idx: int,
        turn: int,
        query: str,
        gt_video: str,
        model_q: str,
        retrieval_q: str,
        gt_rank: int,
        topk_ids: list[str],
        retrieved_id: str,
        answer: str,
        correct_accept: bool,
        correct_reject: bool,
        has_refine_token: bool = False,
        raw_output: Optional[str] = None,
        raw_search_output: Optional[str] = None,
        gt_rank_full: Optional[int] = None,
    ) -> None:
        if not self.args.search_debug:
            return
        rank, step, gen_step, group_id, group_dir = self._get_debug_group_context(idx)
        log_path = os.path.join(group_dir, f"idx{idx}.log")
        topk_preview = topk_ids[:10]
        gt_rank_full_val = gt_rank_full if gt_rank_full is not None else -1
        expected_answer = (
            "matched"
            if retrieved_id and gt_video and str(retrieved_id) == str(gt_video)
            else "not_matched"
        )
        answer_binary_correct = answer == expected_answer
        line = (
            f"[rank {rank}][step {step}][gen_step {gen_step}]"
            f"[idx {idx}][group {group_id}][turn {turn}] "
            f"q={query!r} gt={gt_video!r} model_q={model_q!r} "
            f"retrieval_q={retrieval_q!r} gt_rank={gt_rank} gt_rank_full={gt_rank_full_val} "
            f"topk_len={len(topk_ids)} topk_head={topk_preview} "
            f"retrieved={retrieved_id!r} answer={answer!r} has_refine_token={has_refine_token} "
            f"expected_answer={expected_answer!r} answer_binary_correct={answer_binary_correct} "
            f"correct_accept={correct_accept} correct_reject={correct_reject}"
        )
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")
            if raw_search_output is not None:
                fp.write("RAW_SEARCH_START\n")
                fp.write(raw_search_output + "\n")
                fp.write("RAW_SEARCH_END\n")
            if raw_output:
                fp.write("RAW_OUTPUT_START\n")
                fp.write(raw_output + "\n")
                fp.write("RAW_OUTPUT_END\n")
        self._append_rollout_event(
            idx,
            "turn",
            {
                "turn": int(turn),
                "query": query,
                "gt_video": gt_video,
                "model_search_query": model_q,
                "retrieval_query": retrieval_q,
                "gt_rank": int(gt_rank),
                "gt_rank_full": int(gt_rank_full_val),
                "topk_len": int(len(topk_ids)),
                "topk_head": topk_preview,
                "retrieved_id": retrieved_id,
                "answer": answer,
                "expected_answer": expected_answer,
                "answer_binary_correct": bool(answer_binary_correct),
                "has_refine_token": bool(has_refine_token),
                "correct_accept": bool(correct_accept),
                "correct_reject": bool(correct_reject),
                "raw_search_output": raw_search_output,
                "raw_output": raw_output,
            },
        )

    def _init_latent_resources(self) -> None:
        if not self._refine_token_ids:
            print(
                "[latent][warn] refine tokens are not in tokenizer vocab; "
                f"tokens={self._refine_tokens} "
                "latent improve reward disabled."
            )
            self._latent_enabled = False
            return
        try:
            self._query_embeddings = np.load(
                self.args.query_embeddings_path, mmap_mode="r"
            )
            self._video_embeddings = np.load(
                self.args.video_embeddings_path, mmap_mode="r"
            )
            self._query_meta_by_query = _load_query_meta_by_query(
                self.args.query_meta_path
            )
            with open(self.args.video_docid2row_path, "r", encoding="utf-8") as f:
                self._video_id_to_row = {str(k): int(v) for k, v in json.load(f).items()}
            print(
                "[latent] loaded resources: "
                f"q_embed={self._query_embeddings.shape}, "
                f"video_embed={self._video_embeddings.shape}, "
                f"query_meta={len(self._query_meta_by_query)}, "
                f"docid2row={len(self._video_id_to_row)}"
            )
        except Exception as exc:
            print(f"[latent][error] failed to load retrieval resources: {exc}")
            self._latent_enabled = False
            return

        if self._use_query_embedder_path:
            try:
                q_proc = AutoProcessor.from_pretrained(
                    self.args.query_embedder_model_path, padding_side="left"
                )
                self._query_embedder_tokenizer = q_proc.tokenizer
                q_kwargs = {}
                if self.args.bf16:
                    q_kwargs["torch_dtype"] = torch.bfloat16
                with _zero_init_disabled_ctx():
                    try:
                        q_kwargs["attn_implementation"] = "flash_attention_2"
                        self._query_embedder_model = (
                            Qwen3VLForConditionalGeneration.from_pretrained(
                                self.args.query_embedder_model_path, **q_kwargs
                            )
                        )
                    except Exception:
                        q_kwargs.pop("attn_implementation", None)
                        self._query_embedder_model = (
                            Qwen3VLForConditionalGeneration.from_pretrained(
                                self.args.query_embedder_model_path, **q_kwargs
                            )
                        )
                if hasattr(self._query_embedder_model, "config"):
                    self._query_embedder_model.config.use_cache = False
                for p in self._query_embedder_model.parameters():
                    p.requires_grad = False
                self._query_embedder_model.eval()
                self._query_embedder_model.to(self.accelerator.device)
                emb_layer = self._query_embedder_model.get_input_embeddings()
                emb_weight = getattr(emb_layer, "weight", None)
                q_total_params = 0
                q_zero_numel_params = 0
                for p in self._query_embedder_model.parameters():
                    q_total_params += 1
                    if p.numel() == 0:
                        q_zero_numel_params += 1
                self._query_embedder_needs_param_gather = _module_has_zero3_partitioned_params(
                    self._query_embedder_model
                )
                if emb_weight is not None:
                    print(
                        "[latent] query embedder embedding weight shape="
                        f"{tuple(emb_weight.shape)}, dtype={emb_weight.dtype}"
                    )
                print(
                    "[latent] query embedder param status: "
                    f"total={q_total_params}, zero_numel={q_zero_numel_params}, "
                    f"needs_gather={self._query_embedder_needs_param_gather}"
                )
                print(
                    "[latent] query embedder loaded (frozen) from "
                    f"{self.args.query_embedder_model_path}"
                )
                if self._query_embedder_input_prefix:
                    print(
                        "[latent] query embedder input prefix enabled: "
                        f"{self._query_embedder_input_prefix}"
                    )
            except Exception as exc:
                print(f"[latent][warn] failed to load query embedder: {exc}")
                self._use_query_embedder_path = False

    def _resolve_video_meta_path_for_example(
        self, meta_hint: str, video_root: str
    ) -> str:
        hint = str(meta_hint or "").strip()
        if hint and os.path.exists(hint):
            return hint

        root = str(video_root or "").strip()
        if not root:
            return ""

        cand1 = os.path.join(root, "meta.jsonl")
        if os.path.exists(cand1):
            return cand1

        parent = os.path.dirname(root.rstrip("/"))
        cand2 = os.path.join(parent, "meta.jsonl")
        if os.path.exists(cand2):
            return cand2
        return ""

    def _lookup_video_meta_for_path(
        self, video_path: str, meta_hint: str, video_root: str
    ) -> Optional[dict]:
        meta_path = self._resolve_video_meta_path_for_example(meta_hint, video_root)
        if not meta_path:
            if self.args.search_debug:
                print(f"[search][meta] no meta path for video={video_path}")
            return None
        meta_index = self._video_meta_index_cache.get(meta_path)
        if meta_index is None:
            meta_index = load_video_meta_index(meta_path)
            self._video_meta_index_cache[meta_path] = meta_index
        payload = resolve_video_meta_for_video_path(video_path, meta_index)
        if self.args.search_debug:
            if not hasattr(self, "_meta_debug_seen"):
                self._meta_debug_seen = set()
            key = (str(video_path), str(meta_path))
            if key not in self._meta_debug_seen and len(self._meta_debug_seen) < 32:
                self._meta_debug_seen.add(key)
                if payload:
                    fps = payload.get("raw_fps", payload.get("fps", None))
                    total = payload.get("total_num_frames", None)
                    idx = payload.get("frames_indices", None)
                    idx_len = len(idx) if isinstance(idx, list) else None
                    idx_last = idx[-1] if isinstance(idx, list) and idx else None
                    dur = None
                    if fps and total:
                        try:
                            dur = float(total) / float(fps)
                        except Exception:
                            dur = None
                    print(
                        "[search][meta] hit "
                        f"path={video_path} meta={meta_path} "
                        f"fps={fps} total={total} idx_len={idx_len} idx_last={idx_last} "
                        f"duration_sec={dur}"
                    )
                else:
                    print(
                        f"[search][meta] miss path={video_path} meta={meta_path}"
                    )
        return payload

    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        unwrapped = model
        visited = set()
        while hasattr(unwrapped, "module") and id(unwrapped) not in visited:
            visited.add(id(unwrapped))
            unwrapped = unwrapped.module
        return unwrapped

    def _get_refine_modules(self, model: torch.nn.Module):
        base_model = self._unwrap_model(model)
        projector = getattr(base_model, "refine_projector", None)
        gate = getattr(base_model, "refine_gate", None)
        if projector is None:
            raise AttributeError(
                f"Refine projector is missing on model type {type(base_model)}."
            )
        return projector, gate

    def _make_refine_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        if len(self._refine_token_ids) == 1:
            return input_ids.eq(int(self._refine_token_ids[0]))
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tok_id in self._refine_token_ids:
            mask |= input_ids.eq(int(tok_id))
        return mask

    def _resolve_sqr_train_depth(self, rollout_depth: Optional[int] = None) -> int:
        depth = int(max(1, rollout_depth if rollout_depth is not None else self._refine_rollout_depth))
        if self._sqr_latent_train_depth > 0:
            depth = min(depth, int(self._sqr_latent_train_depth))
        return int(max(1, depth))

    def _resolve_infonce_train_depth(self, rollout_depth: Optional[int] = None) -> int:
        depth = int(max(1, rollout_depth if rollout_depth is not None else self._refine_rollout_depth))
        if self._infonce_latent_train_depth > 0:
            depth = min(depth, int(self._infonce_latent_train_depth))
        return int(max(1, depth))

    def _should_apply_sqr_latent_loss(self) -> bool:
        if not self.model.training:
            return False
        if not self._use_sqr_latent_loss:
            return False
        if self._sqr_latent_loss_weight <= 0:
            return False
        micro_step = int(max(0, getattr(self, "_step", 0) - 1))
        return (micro_step % self._sqr_latent_every_n_steps) == 0

    def _should_apply_infonce_latent_aux_loss(self) -> bool:
        if not self.model.training:
            return False
        if not self._use_infonce_latent_aux_loss:
            return False
        if self._infonce_latent_loss_weight <= 0:
            return False
        micro_step = int(max(0, getattr(self, "_step", 0) - 1))
        return (micro_step % self._infonce_latent_every_n_steps) == 0

    @staticmethod
    def _inject_latent_tokens(
        query_token_embeds: torch.Tensor,
        latent_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        insert_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_tokens.dim() != 3:
            raise ValueError(
                f"latent_tokens must be 3D [B,L,D], got shape={tuple(latent_tokens.shape)}"
            )
        if query_token_embeds.dim() != 3:
            raise ValueError(
                f"query_token_embeds must be 3D [B,T,D], got shape={tuple(query_token_embeds.shape)}"
            )
        if query_token_embeds.size(0) != latent_tokens.size(0):
            raise ValueError(
                "batch size mismatch between query_token_embeds and latent_tokens: "
                f"{query_token_embeds.size(0)} vs {latent_tokens.size(0)}"
            )
        seq_len = int(query_token_embeds.size(1))
        insert_idx = max(0, min(int(insert_idx), seq_len))

        head_embeds = query_token_embeds[:, :insert_idx, :]
        tail_embeds = query_token_embeds[:, insert_idx:, :]
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

    def _build_row_rollout_inputs(
        self, model_inputs: dict[str, Any], row_index: int
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        input_ids = model_inputs.get("input_ids", None)
        if input_ids is None or not torch.is_tensor(input_ids):
            return out

        bsz = int(input_ids.size(0))
        row = int(row_index)
        if row < 0 or row >= bsz:
            return out

        image_slice = None
        image_grid = model_inputs.get("image_grid_thw", None)
        pixel_values = model_inputs.get("pixel_values", None)
        if (
            torch.is_tensor(image_grid)
            and torch.is_tensor(pixel_values)
            and image_grid.dim() == 2
            and int(image_grid.size(0)) == bsz
        ):
            image_lengths = image_grid.prod(dim=1).to(dtype=torch.long).tolist()
            image_start = int(sum(image_lengths[:row]))
            image_end = int(image_start + image_lengths[row])
            image_slice = (image_start, image_end)

        video_slice = None
        video_grid = model_inputs.get("video_grid_thw", None)
        pixel_values_videos = model_inputs.get("pixel_values_videos", None)
        if (
            torch.is_tensor(video_grid)
            and torch.is_tensor(pixel_values_videos)
            and video_grid.dim() == 2
            and int(video_grid.size(0)) == bsz
        ):
            video_lengths = video_grid.prod(dim=1).to(dtype=torch.long).tolist()
            video_start = int(sum(video_lengths[:row]))
            video_end = int(video_start + video_lengths[row])
            video_slice = (video_start, video_end)

        for key, value in model_inputs.items():
            if not torch.is_tensor(value):
                continue
            if key in {"labels", "output_hidden_states"}:
                continue
            if key == "pixel_values" and image_slice is not None:
                s, e = image_slice
                out[key] = value[s:e]
                continue
            if key == "image_grid_thw" and image_slice is not None:
                out[key] = value[row : row + 1]
                continue
            if key == "pixel_values_videos" and video_slice is not None:
                s, e = video_slice
                out[key] = value[s:e]
                continue
            if key == "video_grid_thw" and video_slice is not None:
                out[key] = value[row : row + 1]
                continue
            if value.dim() > 0 and int(value.size(0)) == bsz:
                out[key] = value[row : row + 1]
            else:
                out[key] = value

        if "input_ids" in out and "attention_mask" not in out:
            out["attention_mask"] = torch.ones_like(out["input_ids"], dtype=torch.long)
        return out

    def _rollout_refine_latents(
        self,
        model: PreTrainedModel,
        first_update: torch.Tensor,
        rollout_depth: int,
        row_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if first_update.dim() == 1:
            first_update = first_update.unsqueeze(0)
        if first_update.dim() != 2:
            raise ValueError(
                f"first_update must be 1D/2D tensor, got shape={tuple(first_update.shape)}"
            )
        depth = int(max(1, rollout_depth))
        z_seed = first_update.mean(dim=0, keepdim=True).to(dtype=torch.float32)
        if depth <= 1:
            return z_seed
        if row_model_inputs is None or "input_ids" not in row_model_inputs:
            return z_seed.repeat(depth, 1)

        base_model = self._unwrap_model(model)
        vlm_core = getattr(base_model, "model", None)
        if vlm_core is None or not hasattr(base_model, "get_input_embeddings"):
            return z_seed.repeat(depth, 1)

        refine_projector, refine_gate = self._get_refine_modules(model)
        roll_in_projector = getattr(base_model, "refine_latent_input_projector", None)

        input_ids = row_model_inputs["input_ids"]
        attention_mask = row_model_inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        query_token_embeds = _embed_tokens_safely(base_model.get_input_embeddings(), input_ids)
        refine_mask = self._make_refine_mask(input_ids)
        positions = torch.nonzero(refine_mask[0], as_tuple=False).squeeze(-1)
        if positions.numel() > 0:
            vlm_insert_idx = int(positions[-1].item()) + 1
        else:
            vlm_insert_idx = int(query_token_embeds.size(1))

        extra_forward_inputs: dict[str, torch.Tensor] = {}
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

        base_position_ids = None
        if hasattr(vlm_core, "get_rope_index"):
            try:
                base_position_ids, _ = vlm_core.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=extra_forward_inputs.get("image_grid_thw"),
                    video_grid_thw=extra_forward_inputs.get("video_grid_thw"),
                    attention_mask=attention_mask,
                )
            except Exception:
                base_position_ids = None

        z_list: list[torch.Tensor] = [z_seed.squeeze(0)]
        for _ in range(1, depth):
            z_context = torch.stack(z_list, dim=0)
            z_context_for_llm = z_context
            if roll_in_projector is not None:
                try:
                    latent_proj_dtype = next(roll_in_projector.parameters()).dtype
                except StopIteration:
                    latent_proj_dtype = z_context.dtype
                z_context_for_llm = roll_in_projector(
                    z_context.to(dtype=latent_proj_dtype)
                ).to(dtype=torch.float32)

            latent_tokens = z_context_for_llm.to(dtype=query_token_embeds.dtype).unsqueeze(0)
            inputs_embeds, full_attention = self._inject_latent_tokens(
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
                    [head_pos, latent_pos, tail_pos + latent_len], dim=-1
                )

            vlm_outputs = vlm_core(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention,
                position_ids=full_position_ids,
                use_cache=False,
                **extra_forward_inputs,
            )
            hidden = vlm_outputs.last_hidden_state.to(dtype=torch.float32)
            if self._qfinal_pooling == "mean":
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

            if self._use_refine_gate and refine_gate is not None:
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

    def _build_q_final_from_update(
        self,
        model: PreTrainedModel,
        q_orig: torch.Tensor,
        update: torch.Tensor,
        query_text: str,
        query_embedder_params_gathered: bool = False,
        enable_grad_through_query_embedder: bool = False,
    ) -> tuple[torch.Tensor, str]:
        if update.dim() == 1:
            update = update.unsqueeze(0)
        if update.dim() != 2:
            raise ValueError(
                f"update must be 1D/2D tensor, got shape={tuple(update.shape)}"
            )
        base_model = self._unwrap_model(model)
        if self._use_query_embedder_path:
            if self._query_embedder_model is None or self._query_embedder_tokenizer is None:
                raise RuntimeError("query embedder path enabled but model/tokenizer is missing.")
            embedder_query_text = _build_query_embedder_text(
                self._query_embedder_tokenizer,
                query_text,
                self._query_embedder_input_prefix,
            )
            tokenized = self._query_embedder_tokenizer(
                [embedder_query_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._query_embedder_max_length,
            )
            device = update.device
            input_ids = tokenized["input_ids"].to(device=device)
            attention_mask = tokenized["attention_mask"].to(device=device)

            q_embedder = self._query_embedder_model
            query_embed_layer = q_embedder.get_input_embeddings()
            query_token_embeds = _embed_tokens_safely(query_embed_layer, input_ids)
            tail_len = int(min(6, max(int(query_token_embeds.size(1)), 0)))
            embedder_insert_idx = int(query_token_embeds.size(1) - tail_len)
            append_in_projector = getattr(base_model, "refine_append_input_projector", None)
            if append_in_projector is None:
                append_in_projector = getattr(base_model, "refine_latent_input_projector", None)
            update_for_llm = update
            if append_in_projector is not None:
                try:
                    latent_proj_dtype = next(append_in_projector.parameters()).dtype
                except StopIteration:
                    latent_proj_dtype = update.dtype
                update_for_llm = append_in_projector(
                    update.to(dtype=latent_proj_dtype)
                ).to(dtype=torch.float32)
            latent_tokens = update_for_llm.to(dtype=query_token_embeds.dtype).unsqueeze(0)
            inputs_embeds, full_attention = self._inject_latent_tokens(
                query_token_embeds=query_token_embeds,
                latent_tokens=latent_tokens,
                attention_mask=attention_mask,
                insert_idx=embedder_insert_idx,
            )

            gather_ctx = nullcontext()
            if (
                self._query_embedder_needs_param_gather
                and not query_embedder_params_gathered
            ):
                gather_ctx = _gather_module_params_ctx(q_embedder)
            with gather_ctx:
                if enable_grad_through_query_embedder:
                    outputs = q_embedder.language_model(
                        input_ids=None,
                        attention_mask=full_attention,
                        inputs_embeds=inputs_embeds,
                        use_cache=False,
                    )
                    hidden = outputs.last_hidden_state.to(dtype=torch.float32)
                else:
                    with torch.no_grad():
                        outputs = q_embedder.language_model(
                            input_ids=None,
                            attention_mask=full_attention,
                            inputs_embeds=inputs_embeds,
                            use_cache=False,
                        )
                        hidden = outputs.last_hidden_state.to(dtype=torch.float32)

            if self._qfinal_pooling == "mean":
                mask = full_attention.to(dtype=hidden.dtype).unsqueeze(-1)
                denom = torch.clamp(mask.sum(dim=1), min=1.0)
                pooled = (hidden * mask).sum(dim=1) / denom
            else:
                pooled = hidden[:, -1, :]

            q_head = getattr(base_model, "query_embedder_head", None)
            if q_head is not None:
                try:
                    q_head_dtype = next(q_head.parameters()).dtype
                except StopIteration:
                    q_head_dtype = pooled.dtype
                pooled = q_head(pooled.to(dtype=q_head_dtype))
            q_final = pooled.squeeze(0).to(dtype=torch.float32)
            if q_final.size(-1) != q_orig.size(-1):
                raise ValueError(
                    "q_final dim mismatch in query_embedder path: "
                    f"{q_final.size(-1)} vs expected {q_orig.size(-1)}"
                )
            qmode = "query_embedder"
        else:
            q_final = q_orig + update.mean(dim=0)
            qmode = "residual_add"

        if self._qfinal_normalize:
            q_final = _l2_norm(q_final.unsqueeze(0)).squeeze(0)
        return q_final, qmode

    def _populate_latent_improve_meta(
        self,
        reward_meta: list[dict[str, Any]],
        queries: list[str],
        gt_videos: list[str],
        hard_negative_ids_list: list[list[str]],
        full_inputs: dict[str, Any],
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        prompt_lengths: list[int],
    ) -> dict[str, torch.Tensor]:
        for meta in reward_meta:
            meta.setdefault("improve_delta", 0.0)
            meta.setdefault("improve_delta_raw", 0.0)
            meta.setdefault("improve_has_refine", False)
            meta.setdefault("improve_qfinal_mode", "")
            meta.setdefault("improve_sim_before", 0.0)
            meta.setdefault("improve_sim_after", 0.0)
            meta.setdefault("margin_has_neg", False)
            meta.setdefault("margin_neg_count", 0)
            meta.setdefault("margin_sim_neg_before", 0.0)
            meta.setdefault("margin_sim_neg_after", 0.0)
            meta.setdefault("margin_before", 0.0)
            meta.setdefault("margin_after", 0.0)
            meta.setdefault("margin_delta_raw", 0.0)
            meta.setdefault("margin_delta", 0.0)
            meta.setdefault("query_refine_quality_has_neg", False)
            meta.setdefault("query_refine_quality_neg_count", 0)
            meta.setdefault("query_refine_quality_sim_pos", 0.0)
            meta.setdefault("query_refine_quality_neg_lse", 0.0)
            meta.setdefault("query_refine_quality_raw", 0.0)
            meta.setdefault("query_refine_quality", 0.0)
            meta.setdefault("query_refine_quality_infonce", 0.0)
            meta.setdefault("query_refine_quality_infonce_with_qorig", 0.0)
            meta.setdefault("query_refine_prob_before", 0.0)
            meta.setdefault("query_refine_prob_after", 0.0)
            meta.setdefault("latent_rollout_depth", 0)
            meta.setdefault("sqr_old_norm_mean", 0.0)
            meta.setdefault("sqr_action_norm_mean", 0.0)
            meta.setdefault("sqr_action_dist2_mean", 0.0)
            meta.setdefault("sqr_mask_sum", 0.0)

        if not self._latent_enabled:
            return {}
        if self._query_embeddings is None or self._video_embeddings is None:
            return {}

        model = self.model
        base_model = self._unwrap_model(model)
        projector = getattr(base_model, "refine_projector", None)
        gate = getattr(base_model, "refine_gate", None)
        if projector is None:
            return {}

        fwd_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "output_hidden_states": True,
        }
        use_latent_forward_vision = _env_flag("GRPO_LATENT_FORWARD_USE_VISION", True)
        if use_latent_forward_vision:
            for key in (
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
            ):
                if key in full_inputs:
                    fwd_inputs[key] = full_inputs[key]
        if self._stage_debug_active():
            self._stage_debug_log(
                "latent:forward_start",
                f"use_vision={use_latent_forward_vision} input_shape={tuple(full_input_ids.shape)}",
            )
        with torch.no_grad():
            outputs = model(**fwd_inputs)
            hidden = outputs.hidden_states[-1].to(dtype=torch.float32)
        if self._stage_debug_active():
            self._stage_debug_log(
                "latent:forward_done",
                f"hidden_shape={tuple(hidden.shape)}",
            )
        try:
            projector_dtype = next(projector.parameters()).dtype
        except StopIteration:
            projector_dtype = hidden.dtype
        gate_dtype = hidden.dtype
        if gate is not None:
            try:
                gate_dtype = next(gate.parameters()).dtype
            except StopIteration:
                gate_dtype = hidden.dtype

        device = hidden.device
        rollout_depth = int(max(1, self._refine_rollout_depth))
        sqr_enabled = bool(
            self._use_sqr_latent_loss and self._sqr_latent_loss_weight > 0.0
        )
        sqr_train_depth = self._resolve_sqr_train_depth(rollout_depth)
        sqr_sigma = float(self._sqr_latent_sigma)
        latent_dim = int(self._query_embeddings.shape[1])
        sqr_old_mean_rows: list[torch.Tensor] = []
        sqr_action_rows: list[torch.Tensor] = []
        sqr_logp_old_rows: list[torch.Tensor] = []
        sqr_mask_rows: list[torch.Tensor] = []

        seq_len = full_input_ids.size(1)
        token_pos = torch.arange(seq_len, device=device)
        batch_gt_rows = []
        for vid in gt_videos:
            row = self._video_id_to_row.get(str(vid).strip())
            batch_gt_rows.append(int(row) if row is not None else -1)
        video_emb_cache: dict[int, torch.Tensor] = {}

        def _get_video_emb(row: int) -> torch.Tensor:
            emb = video_emb_cache.get(int(row))
            if emb is not None:
                return emb
            emb = torch.from_numpy(
                np.array(self._video_embeddings[int(row)], dtype=np.float32, copy=True)
            ).to(device=device, dtype=torch.float32)
            emb = _l2_norm(emb.unsqueeze(0)).squeeze(0)
            video_emb_cache[int(row)] = emb
            return emb

        def _append_sqr_default(meta_obj: dict[str, Any]) -> None:
            if not sqr_enabled:
                return
            old_mean = torch.zeros(
                (sqr_train_depth, latent_dim),
                device=device,
                dtype=torch.float32,
            )
            action = old_mean.clone()
            logp_old = torch.zeros(
                (sqr_train_depth,),
                device=device,
                dtype=torch.float32,
            )
            mask = torch.zeros(
                (sqr_train_depth,),
                device=device,
                dtype=torch.float32,
            )
            meta_obj["sqr_old_norm_mean"] = 0.0
            meta_obj["sqr_action_norm_mean"] = 0.0
            meta_obj["sqr_action_dist2_mean"] = 0.0
            meta_obj["sqr_mask_sum"] = 0.0
            sqr_old_mean_rows.append(old_mean.detach())
            sqr_action_rows.append(action.detach())
            sqr_logp_old_rows.append(logp_old.detach())
            sqr_mask_rows.append(mask.detach())

        query_embedder_outer_ctx = nullcontext()
        query_embedder_params_gathered = False
        if (
            self._use_query_embedder_path
            and self._query_embedder_needs_param_gather
            and self._query_embedder_model is not None
        ):
            # Gather once per batch to avoid per-sample collective mismatches.
            query_embedder_outer_ctx = _gather_module_params_ctx(
                self._query_embedder_model
            )
            query_embedder_params_gathered = True

        with query_embedder_outer_ctx:
            for idx, meta in enumerate(reward_meta):
                query_text = str(queries[idx]).strip()
                gt_video = str(gt_videos[idx]).strip()
                if not query_text or not gt_video:
                    _append_sqr_default(meta)
                    continue

                qmeta = self._query_meta_by_query.get(query_text)
                gt_row = self._video_id_to_row.get(gt_video)
                if qmeta is None or gt_row is None:
                    _append_sqr_default(meta)
                    continue

                q_row = int(qmeta[0])
                prompt_len = int(prompt_lengths[idx])
                ids_row = full_input_ids[idx]
                pos_mask = torch.zeros_like(ids_row, dtype=torch.bool)
                for tok_id in self._refine_token_ids:
                    pos_mask |= ids_row.eq(int(tok_id))
                pos_mask &= token_pos >= prompt_len
                refine_positions = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                q_orig = torch.from_numpy(
                    np.array(self._query_embeddings[q_row], dtype=np.float32, copy=True)
                ).to(device=device, dtype=torch.float32)
                q_orig = _l2_norm(q_orig.unsqueeze(0)).squeeze(0)
                gt_emb = _get_video_emb(int(gt_row))

                has_refine = refine_positions.numel() > 0
                if has_refine:
                    h_ref = hidden[idx, refine_positions].to(dtype=torch.float32)
                    h_ref_for_projector = h_ref.to(dtype=projector_dtype)
                    delta = projector(h_ref_for_projector).to(dtype=torch.float32)

                    if self._use_refine_gate and gate is not None:
                        h_ref_for_gate = h_ref.to(dtype=gate_dtype)
                        alpha = (
                            torch.sigmoid(gate(h_ref_for_gate)).to(dtype=torch.float32)
                        )
                        update = delta * alpha
                    else:
                        update = delta
                    row_rollout_inputs = self._build_row_rollout_inputs(fwd_inputs, idx)
                    update = self._rollout_refine_latents(
                        model=model,
                        first_update=update,
                        rollout_depth=rollout_depth,
                        row_model_inputs=row_rollout_inputs,
                    )
                else:
                    # Keep per-rank execution symmetric in ZeRO-3 gather path.
                    update = torch.zeros(
                        (1, int(q_orig.size(-1))),
                        device=device,
                        dtype=torch.float32,
                    )
                meta["latent_rollout_depth"] = int(update.size(0))

                if sqr_enabled:
                    if has_refine:
                        old_mean = update[:sqr_train_depth]
                        if old_mean.size(0) < sqr_train_depth:
                            pad_count = int(sqr_train_depth - old_mean.size(0))
                            old_mean = torch.cat(
                                [
                                    old_mean,
                                    old_mean[-1:].repeat(pad_count, 1),
                                ],
                                dim=0,
                            )
                        noise = torch.randn_like(old_mean) * sqr_sigma
                        action = old_mean + noise
                        dist2_old = ((action - old_mean) ** 2).mean(dim=-1)
                        logp_old = -dist2_old / (2.0 * (sqr_sigma ** 2))
                        mask = torch.ones(
                            (sqr_train_depth,), device=device, dtype=torch.float32
                        )
                    else:
                        old_mean = torch.zeros(
                            (sqr_train_depth, latent_dim),
                            device=device,
                            dtype=torch.float32,
                        )
                        action = old_mean.clone()
                        logp_old = torch.zeros(
                            (sqr_train_depth,), device=device, dtype=torch.float32
                        )
                        dist2_old = torch.zeros(
                            (sqr_train_depth,), device=device, dtype=torch.float32
                        )
                        mask = torch.zeros(
                            (sqr_train_depth,), device=device, dtype=torch.float32
                        )

                    meta["sqr_old_norm_mean"] = float(
                        old_mean.norm(dim=-1).mean().item()
                    )
                    meta["sqr_action_norm_mean"] = float(
                        action.norm(dim=-1).mean().item()
                    )
                    meta["sqr_action_dist2_mean"] = float(dist2_old.mean().item())
                    meta["sqr_mask_sum"] = float(mask.sum().item())

                    sqr_old_mean_rows.append(old_mean.detach())
                    sqr_action_rows.append(action.detach())
                    sqr_logp_old_rows.append(logp_old.detach())
                    sqr_mask_rows.append(mask.detach())

                if self._stage_debug_active():
                    self._stage_debug_log(
                        "latent:qfinal_start",
                        f"idx={idx} has_refine={bool(has_refine)} q_row={q_row} gt_row={int(gt_row)}",
                    )
                q_final, qmode = self._build_q_final_from_update(
                    model=model,
                    q_orig=q_orig,
                    update=update,
                    query_text=query_text,
                    query_embedder_params_gathered=query_embedder_params_gathered,
                )
                if self._stage_debug_active():
                    self._stage_debug_log(
                        "latent:qfinal_done",
                        f"idx={idx} mode={qmode}",
                    )

                if not has_refine:
                    meta["improve_qfinal_mode"] = f"{qmode}:no_refine"
                    continue

                sim_before = float(torch.dot(q_orig, gt_emb).item())
                sim_after = float(torch.dot(q_final, gt_emb).item())
                delta_raw = sim_after - sim_before
                delta_scaled = delta_raw * self._improve_reward_scale

                neg_rows: set[int] = set()
                if idx < len(hard_negative_ids_list):
                    for hard_vid in hard_negative_ids_list[idx]:
                        if not hard_vid or hard_vid == gt_video:
                            continue
                        hard_row = self._video_id_to_row.get(str(hard_vid))
                        if hard_row is not None and int(hard_row) != int(gt_row):
                            neg_rows.add(int(hard_row))
                for turn in meta.get("turns", []):
                    rid = str(turn.get("retrieved_id", "") or "").strip()
                    if rid and rid != gt_video:
                        rid_row = self._video_id_to_row.get(rid)
                        if rid_row is not None and int(rid_row) != int(gt_row):
                            neg_rows.add(int(rid_row))
                    rid_list = turn.get("retrieved_ids", [])
                    if isinstance(rid_list, list):
                        for rid_item in rid_list:
                            rid2 = str(rid_item or "").strip()
                            if not rid2 or rid2 == gt_video:
                                continue
                            rid2_row = self._video_id_to_row.get(rid2)
                            if rid2_row is not None and int(rid2_row) != int(gt_row):
                                neg_rows.add(int(rid2_row))

                # In-batch negatives: other samples' GT videos.
                for j, other_row in enumerate(batch_gt_rows):
                    if j == idx or other_row < 0:
                        continue
                    if int(other_row) == int(gt_row):
                        continue
                    other_gt = str(gt_videos[j]).strip()
                    if other_gt and other_gt == gt_video:
                        continue
                    neg_rows.add(int(other_row))

                if neg_rows:
                    neg_embs = torch.stack(
                        [_get_video_emb(r) for r in sorted(neg_rows)], dim=0
                    )
                    sim_neg_all_after = torch.matmul(neg_embs, q_final)
                    sim_neg_all_before = torch.matmul(neg_embs, q_orig)
                    temp = float(self._query_refine_temperature)
                    neg_lse_after = float(
                        (temp * torch.logsumexp(sim_neg_all_after / temp, dim=0)).item()
                    )
                    quality_raw = sim_after - neg_lse_after
                    quality_scaled = quality_raw * self._margin_reward_scale

                    # InfoNCE probability variants (remove -log from CE form):
                    # - quality_scaled_infonce: p_after
                    # - quality_scaled_infonce_with_qorig: p_after - p_before
                    logits_after = torch.cat(
                        [torch.dot(q_final, gt_emb).unsqueeze(0), sim_neg_all_after], dim=0
                    ) / temp
                    logits_before = torch.cat(
                        [torch.dot(q_orig, gt_emb).unsqueeze(0), sim_neg_all_before], dim=0
                    ) / temp
                    p_after = torch.softmax(logits_after, dim=0)[0]
                    p_before = torch.softmax(logits_before, dim=0)[0]
                    quality_scaled_infonce = float(p_after.item())
                    quality_scaled_infonce_with_qorig = float((p_after - p_before).item())

                    sim_neg_before = float(sim_neg_all_before.max().item())
                    sim_neg_after = float(torch.matmul(neg_embs, q_final).max().item())
                    margin_before = sim_before - sim_neg_before
                    margin_after = sim_after - sim_neg_after
                    margin_delta_raw = margin_after - margin_before
                    margin_delta = margin_delta_raw * self._margin_reward_scale
                    
                    meta["query_refine_quality_has_neg"] = True
                    meta["query_refine_quality_neg_count"] = int(len(neg_rows))
                    meta["query_refine_quality_sim_pos"] = float(sim_after)
                    meta["query_refine_quality_neg_lse"] = float(neg_lse_after)
                    meta["query_refine_quality_raw"] = float(quality_raw)
                    meta["query_refine_quality_infonce"] = float(quality_scaled_infonce)
                    meta["query_refine_quality_infonce_with_qorig"] = float(
                        quality_scaled_infonce_with_qorig
                    )
                    meta["query_refine_prob_before"] = float(p_before.item())
                    meta["query_refine_prob_after"] = float(p_after.item())
                    

                    ####
                    #meta["query_refine_quality"] = float(quality_scaled)
                    #meta["query_refine_quality"] = float(quality_scaled_infonce)
                    meta["query_refine_quality"] = float(quality_scaled_infonce_with_qorig)
                    ###


                    meta["margin_has_neg"] = True
                    meta["margin_neg_count"] = int(len(neg_rows))
                    meta["margin_sim_neg_before"] = float(sim_neg_before)
                    meta["margin_sim_neg_after"] = float(sim_neg_after)
                    meta["margin_before"] = float(margin_before)
                    meta["margin_after"] = float(margin_after)
                    meta["margin_delta_raw"] = float(margin_delta_raw)
                    meta["margin_delta"] = float(margin_delta)

                meta["improve_has_refine"] = True
                meta["improve_qfinal_mode"] = qmode
                meta["improve_sim_before"] = sim_before
                meta["improve_sim_after"] = sim_after
                meta["improve_delta_raw"] = float(delta_raw)
                meta["improve_delta"] = float(delta_scaled)

        if not sqr_enabled:
            return {}
        if not sqr_old_mean_rows:
            batch_size = int(full_input_ids.size(0))
            zeros_mean = torch.zeros(
                (batch_size, sqr_train_depth, latent_dim),
                device=device,
                dtype=torch.float32,
            )
            zeros_logp = torch.zeros(
                (batch_size, sqr_train_depth),
                device=device,
                dtype=torch.float32,
            )
            return {
                "sqr_latent_old_mean": zeros_mean,
                "sqr_latent_action": zeros_mean.clone(),
                "sqr_latent_logp_old": zeros_logp,
                "sqr_latent_mask": zeros_logp.clone(),
            }
        return {
            "sqr_latent_old_mean": torch.stack(sqr_old_mean_rows, dim=0),
            "sqr_latent_action": torch.stack(sqr_action_rows, dim=0),
            "sqr_latent_logp_old": torch.stack(sqr_logp_old_rows, dim=0),
            "sqr_latent_mask": torch.stack(sqr_mask_rows, dim=0),
        }

    def _generate_and_score_completions_search(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        solutions = [example["response"] for example in inputs]  # gt solutions
        problem_types = [example["problem_type"] for example in inputs]
        base_messages = [copy.deepcopy(example["messages"]) for example in inputs]
        messages = [copy.deepcopy(example["messages"]) for example in inputs]

        queries = [
            str(example.get("query", "")).strip() for example in inputs
        ]
        gt_videos = [
            str(example.get("gt_video", example.get("response", ""))).strip()
            for example in inputs
        ]
        def _normalize_gt_span(raw: Any) -> Optional[list[float]]:
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                return None
            try:
                start = float(raw[0])
                end = float(raw[1])
            except (TypeError, ValueError):
                return None
            return [start, end]

        gt_times = [_normalize_gt_span(example.get("gt_time", None)) for example in inputs]
        hard_negative_ids_list: list[list[str]] = []
        for example in inputs:
            hnegs = example.get("hard_negative_ids", [])
            if not isinstance(hnegs, list):
                hnegs = []
            hard_negative_ids_list.append(
                [str(x).strip() for x in hnegs if str(x).strip()]
            )
        bootstrap_videos = [
            str(example.get("bootstrap_video", "")).strip()
            for example in inputs
        ]
        video_roots = [
            str(example.get("video_root", "")).strip() for example in inputs
        ]
        video_meta_hints = [
            str(example.get("video_meta_path", "")).strip() for example in inputs
        ]
        video_cfgs = [
            example.get("video_cfg", {})
            for example in inputs
        ]

        active = [True] * len(inputs)
        retrieval_query = queries[:]
        last_search_query_used = queries[:]
        shadow_queries = [None] * len(inputs)
        reward_meta = []
        for i in range(len(inputs)):
            reward_meta.append(
                {
                    "turns": [],
                    "gt_video": gt_videos[i],
                    "gt_time": gt_times[i],
                }
            )
        dummy_messages = [[{"role": "user", "content": "ping"}]]
        batch_size = len(inputs)
        self._stage_debug_log(
            "search:start",
            f"batch={batch_size} active={sum(1 for x in active if x)} latent={self._latent_enabled}",
        )

        for turn in range(self.args.search_max_turns):
            active_idx = [i for i, flag in enumerate(active) if flag]
            use_bootstrap_turn0 = (
                turn == 0
                and bool(self.args.search_use_bootstrap_video)
                and bool(active_idx)
                and all(bool(bootstrap_videos[i]) for i in active_idx)
            )
            # Search step (only on turn 0, following eval protocol)
            search_queries_full = [""] * batch_size
            search_ok_flags_full = [False] * batch_size
            model_search_queries_full = [""] * batch_size
            search_raw_full = [None] * batch_size
            if turn == 0:
                if use_bootstrap_turn0:
                    for idx in active_idx:
                        search_query_used = retrieval_query[idx]
                        search_queries_full[idx] = search_query_used
                        search_ok_flags_full[idx] = True
                        model_search_queries_full[idx] = search_query_used
                        search_raw_full[idx] = None
                elif self.args.search_use_original_query:
                    for idx in active_idx:
                        search_query_used = retrieval_query[idx]
                        search_queries_full[idx] = search_query_used
                        search_ok_flags_full[idx] = True
                        model_search_queries_full[idx] = ""
                        search_raw_full[idx] = None
                else:
                    search_batch_full = [
                        messages[i] if active[i] else dummy_messages[0]
                        for i in range(batch_size)
                    ]
                    search_texts, _ = self._generate_text_batch(search_batch_full)
                    for idx in active_idx:
                        raw = search_texts[idx]
                        search_raw_full[idx] = raw
                        search_query = _extract_search_query(raw)
                        search_ok = bool(search_query)
                        if search_ok and _is_bad_search_query(
                            search_query,
                            self.args.search_min_query_chars,
                            self.args.search_min_query_alnum,
                        ):
                            search_ok = False
                        search_query_used = search_query if search_ok else retrieval_query[idx]
                        search_used = raw if search_ok else _default_search_msg(search_query_used, "")
                        messages[idx].append({"role": "assistant", "content": search_used})
                        search_queries_full[idx] = search_query_used
                        search_ok_flags_full[idx] = search_ok
                        model_search_queries_full[idx] = search_query
            else:
                for idx in active_idx:
                    search_queries_full[idx] = retrieval_query[idx]
                    search_ok_flags_full[idx] = True
                    model_search_queries_full[idx] = ""
                    search_raw_full[idx] = None

            # Retrieval
            if active_idx:
                if use_bootstrap_turn0:
                    retrieved_ids = [[bootstrap_videos[idx]] for idx in active_idx]
                    retrieved_scores = [[0.0] for _ in active_idx]
                    retrieved_ranks = None
                else:
                    search_queries = [search_queries_full[idx] for idx in active_idx]
                    rank_for_ids = None
                    if self.args.search_rank_k:
                        rank_for_ids = [gt_videos[idx] for idx in active_idx]
                    retrieved_ids, retrieved_scores, retrieved_ranks = self._retrieve_topk(
                        search_queries, rank_for_ids=rank_for_ids
                    )
            else:
                retrieved_ids, retrieved_scores, retrieved_ranks = [], [], None

            retrieved_id_list = [""] * batch_size
            retrieved_ids_list = [[] for _ in range(batch_size)]
            retrieved_scores_list = [[] for _ in range(batch_size)]
            gt_rank_list = [-1] * batch_size
            gt_rank_full_list = [-1] * batch_size
            missing_video_count = 0

            for local_i, idx in enumerate(active_idx):
                id_list = retrieved_ids[local_i] if local_i < len(retrieved_ids) else []
                score_list = retrieved_scores[local_i] if local_i < len(retrieved_scores) else []
                rid = id_list[0] if id_list else ""
                retrieved_id_list[idx] = rid
                retrieved_ids_list[idx] = id_list
                retrieved_scores_list[idx] = score_list
                video_root = video_roots[idx]
                video_path = _resolve_retrieved_video_path(video_root, rid)
                if not video_path:
                    missing_video_count += 1
                    env_content = [
                        {"type": "text", "text": "<information>\nRetrieved video: </information>"}
                    ]
                else:
                    video_meta = None
                    if str(video_path).lower().endswith((".npy", ".npz")):
                        video_meta = self._lookup_video_meta_for_path(
                            video_path=video_path,
                            meta_hint=video_meta_hints[idx],
                            video_root=video_root,
                        )
                    env_content = _build_env_content(
                        video_path,
                        video_cfgs[idx],
                        video_meta=video_meta,
                    )
                # Use user role so chat template inserts MM placeholders for vLLM.
                messages[idx].append({"role": "user", "content": env_content})
            if active_idx:
                self._stage_debug_log(
                    "search:retrieval_batch",
                    f"turn={turn} active={len(active_idx)} missing_video={missing_video_count}",
                )

            # Answer step
            answer_batch_full = [
                messages[i] if active[i] else dummy_messages[0]
                for i in range(batch_size)
            ]
            answer_texts, answer_ids = self._generate_text_batch(answer_batch_full)
            for local_i, idx in enumerate(active_idx):
                output = answer_texts[idx]
                has_refine_token = False
                if (
                    bool(self._refine_token_ids)
                    and idx < len(answer_ids)
                    and torch.is_tensor(answer_ids[idx])
                    and answer_ids[idx].numel() > 0
                ):
                    refine_mask = torch.zeros_like(answer_ids[idx], dtype=torch.bool)
                    for tok_id in self._refine_token_ids:
                        refine_mask |= answer_ids[idx].eq(int(tok_id))
                    has_refine_token = bool(refine_mask.any().item())
                elif self._refine_suffix:
                    # Base checkpoints may not contain explicit refine vocab.
                    has_refine_token = self._refine_suffix in output
                if self.args.search_force_refine_token and self._refine_suffix:
                    if self._refine_suffix not in output:
                        output = output.rstrip() + f" {self._refine_suffix}"
                        # Keep reward-meta consistent with forced append behavior.
                        has_refine_token = True
                answer = _extract_answer_tag(output)
                instruction = _extract_search_instruction(output)
                answer_ok = answer in {"matched", "not_matched"}
                instruction_ok = bool(instruction) if answer == "not_matched" else False

                retrieved_id = retrieved_id_list[idx]
                gt_video = gt_videos[idx]
                correct_accept = retrieved_id == gt_video and answer == "matched"
                correct_reject = retrieved_id != gt_video and answer == "not_matched"
                gt_rank = -1
                id_list = retrieved_ids_list[idx]
                score_list = retrieved_scores_list[idx]
                if gt_video and id_list:
                    try:
                        gt_rank = id_list.index(gt_video) + 1
                    except ValueError:
                        gt_rank = -1
                gt_rank_list[idx] = gt_rank
                if retrieved_ranks is not None and local_i < len(retrieved_ranks):
                    gt_rank_full_list[idx] = int(retrieved_ranks[local_i] or -1)
                    # Strict sanity:
                    # 1) If top1 == GT, full-rank must be 1.
                    if id_list and gt_video and id_list[0] == gt_video:
                        if gt_rank_full_list[idx] != 1:
                            raise RuntimeError(
                                "Inconsistent rank_full: top1==GT but gt_rank_full "
                                f"={gt_rank_full_list[idx]} (query={queries[idx]!r}, "
                                f"gt={gt_video}, rank_k={self.args.search_rank_k})"
                            )
                    # 2) If full-rank says GT is 1, top1 must be GT.
                    if gt_video and gt_rank_full_list[idx] == 1:
                        if not id_list or id_list[0] != gt_video:
                            raise RuntimeError(
                                "Inconsistent rank_full: gt_rank_full==1 but top1!=GT "
                                f"(query={queries[idx]!r}, gt={gt_video}, "
                                f"top1={id_list[0] if id_list else ''}, "
                                f"rank_k={self.args.search_rank_k})"
                            )

                reward_meta[idx]["turns"].append(
                    {
                        "model_search_query": model_search_queries_full[idx],
                        "retrieval_query": search_queries_full[idx],
                        "retrieved_id": retrieved_id,
                        "retrieved_ids": id_list,
                        "retrieved_scores": score_list,
                        "gt_rank": gt_rank,
                        "gt_rank_full": gt_rank_full_list[idx],
                        "answer": answer,
                        "search_ok": bool(search_ok_flags_full[idx]),
                        "answer_ok": answer_ok,
                        "has_refine_token": has_refine_token,
                        "instruction_ok": instruction_ok,
                        "correct_accept": correct_accept,
                        "correct_reject": correct_reject,
                    }
                )

                messages[idx].append({"role": "assistant", "content": output})

                raw_output = None
                if self.args.search_debug:
                    raw_output = output

                self._write_search_debug_log(
                    idx=idx,
                    turn=turn,
                    query=queries[idx],
                    gt_video=gt_video,
                    model_q=model_search_queries_full[idx],
                    retrieval_q=search_queries_full[idx],
                    gt_rank=gt_rank,
                    gt_rank_full=gt_rank_full_list[idx],
                    topk_ids=id_list,
                    retrieved_id=retrieved_id,
                    answer=answer,
                    has_refine_token=has_refine_token,
                    correct_accept=correct_accept,
                    correct_reject=correct_reject,
                    raw_output=raw_output,
                    raw_search_output=search_raw_full[idx],
                )

                if answer == "matched":
                    active[idx] = False
                else:
                    if self.args.search_use_instruction and instruction:
                        base_query = last_search_query_used[idx] or queries[idx]
                        retrieval_query[idx] = f"{base_query}\n{instruction}"
                        last_search_query_used[idx] = base_query
                        if self.args.search_shadow_retrieve and self.args.search_max_turns == 1:
                            shadow_queries[idx] = retrieval_query[idx]
                    else:
                        retrieval_query[idx] = last_search_query_used[idx] or queries[idx]

            if self.args.search_debug and active_idx:
                dbg_idx = active_idx[0]
                dbg_group = dbg_idx // max(int(self.num_generations), 1)
                last_turn = reward_meta[dbg_idx]["turns"][-1]
                topk_ids = last_turn.get("retrieved_ids", [])
                if not topk_ids:
                    # Fallback to single id if list wasn't stored
                    topk_ids = [last_turn.get("retrieved_id", "")]
                topk_preview = topk_ids[:5]
                print(
                    f"[search][turn {turn}] idx={dbg_idx} group={dbg_group} "
                    f"q='{queries[dbg_idx]}' "
                    f"gt='{gt_videos[dbg_idx]}' "
                    f"model_q='{last_turn.get('model_search_query', '')}' "
                    f"retrieval_q={last_turn.get('retrieval_query', '')!r} "
                    f"gt_rank={last_turn.get('gt_rank', -1)} "
                    f"gt_rank_full={last_turn.get('gt_rank_full', -1)} "
                    f"topk_len={len(topk_ids)} topk_head={topk_preview} "
                    f"retrieved='{last_turn['retrieved_id']}' "
                    f"answer='{last_turn['answer']}' "
                    f"correct_accept={last_turn['correct_accept']} "
                    f"correct_reject={last_turn['correct_reject']}"
                )
            self._stage_debug_log(
                "search:turn_done",
                f"turn={turn} active={sum(1 for x in active if x)}",
            )

        if self.args.search_shadow_retrieve and self.args.search_max_turns == 1:
            shadow_idx = [i for i, q in enumerate(shadow_queries) if q]
            if shadow_idx:
                shadow_qs = [shadow_queries[i] for i in shadow_idx]
                rank_for_ids = None
                if self.args.search_rank_k:
                    rank_for_ids = [gt_videos[i] for i in shadow_idx]
                shadow_ids, shadow_scores, shadow_ranks = self._retrieve_topk(
                    shadow_qs, rank_for_ids=rank_for_ids
                )
                for local_i, idx in enumerate(shadow_idx):
                    id_list = shadow_ids[local_i] if local_i < len(shadow_ids) else []
                    score_list = shadow_scores[local_i] if local_i < len(shadow_scores) else []
                    rid = id_list[0] if id_list else ""
                    gt_video = gt_videos[idx]
                    gt_rank = -1
                    gt_rank_full = -1
                    if gt_video and id_list:
                        try:
                            gt_rank = id_list.index(gt_video) + 1
                        except ValueError:
                            gt_rank = -1
                    if shadow_ranks is not None and local_i < len(shadow_ranks):
                        gt_rank_full = int(shadow_ranks[local_i] or -1)
                        # Strict sanity for shadow retrieve as well.
                        if id_list and gt_video and id_list[0] == gt_video:
                            if gt_rank_full != 1:
                                raise RuntimeError(
                                    "Inconsistent shadow rank_full: top1==GT but gt_rank_full "
                                    f"={gt_rank_full} (query={queries[idx]!r}, gt={gt_video}, "
                                    f"rank_k={self.args.search_rank_k})"
                                )
                        if gt_video and gt_rank_full == 1:
                            if not id_list or id_list[0] != gt_video:
                                raise RuntimeError(
                                    "Inconsistent shadow rank_full: gt_rank_full==1 but top1!=GT "
                                    f"(query={queries[idx]!r}, gt={gt_video}, "
                                    f"top1={id_list[0] if id_list else ''}, "
                                    f"rank_k={self.args.search_rank_k})"
                                )
                    reward_meta[idx]["turns"].append(
                        {
                            "model_search_query": "",
                            "retrieval_query": shadow_queries[idx],
                            "retrieved_id": rid,
                            "retrieved_ids": id_list,
                            "retrieved_scores": score_list,
                            "gt_rank": gt_rank,
                            "gt_rank_full": gt_rank_full,
                            "answer": "shadow",
                            "search_ok": True,
                            "answer_ok": False,
                            "instruction_ok": False,
                            "correct_accept": False,
                            "correct_reject": False,
                            "shadow": True,
                        }
                    )
                    if self.args.search_debug:
                        self._write_search_debug_log(
                            idx=idx,
                            turn=1,
                            query=queries[idx],
                            gt_video=gt_video,
                            model_q="",
                            retrieval_q=shadow_queries[idx],
                            gt_rank=gt_rank,
                            gt_rank_full=gt_rank_full,
                            topk_ids=id_list,
                            retrieved_id=rid,
                            answer="shadow",
                            has_refine_token=False,
                            correct_accept=False,
                            correct_reject=False,
                            raw_output=None,
                            raw_search_output=None,
                        )
        self._stage_debug_log(
            "search:turns_completed",
            f"num_samples={len(reward_meta)}",
        )

        # Summarize reward metadata
        for meta in reward_meta:
            turns_all = meta.get("turns", [])
            turns = [t for t in turns_all if not t.get("shadow")]
            if not turns:
                meta.update(
                    {
                        "final_correct": False,
                        "middle_correct_rate": 0.0,
                        "turn_penalty": 0.0,
                        "format_search": 0.0,
                        "format_answer": 0.0,
                        "format_instruction": 0.0,
                    }
                )
                continue
            final = turns[-1]
            middle = turns[:-1]
            middle_hits = [t for t in middle if t["correct_accept"] or t["correct_reject"]]
            instr_turns = [t for t in turns if t["answer"] == "not_matched"]
            meta.update(
                {
                    "final_correct": bool(final["correct_accept"]),
                    "middle_correct_rate": (len(middle_hits) / max(len(middle), 1))
                    if middle
                    else 0.0,
                    "turn_penalty": -float(max(len(turns) - 1, 0)),
                    "format_search": 1.0 if turns and turns[0]["search_ok"] else 0.0,
                    "format_answer": sum(t["answer_ok"] for t in turns) / len(turns),
                    "format_instruction": (
                        sum(t["instruction_ok"] for t in instr_turns) / len(instr_turns)
                        if instr_turns
                        else 1.0
                    ),
                }
            )

        # Build completion texts for reward functions
        completions_text = []
        for msg_list in messages:
            parts = []
            for msg in msg_list:
                if msg.get("role") == "assistant":
                    content = str(msg.get("content", ""))
                    # Keep completion_text focused on final answer content; omit pure search-call traces.
                    if _extract_search_query(content) and not _extract_answer_tag(content):
                        continue
                    parts.append(content)
            completions_text.append("\n".join(parts))

        # Tokenize full conversation
        t_tokenize = time.perf_counter()
        self._stage_debug_log("search:tokenize_start", f"batch={len(messages)}")
        normalized_full = [self._normalize_messages(m) for m in messages]
        normalized_prompt = [self._normalize_messages(m) for m in base_messages]

        full_texts = [
            self.processing_class.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
            )
            for msg in normalized_full
        ]
        prompt_texts = [
            self.processing_class.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in normalized_prompt
        ]

        full_image_inputs, full_packed_video_inputs, full_video_kwargs = cached_process_vision_info(
            normalized_full,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if full_packed_video_inputs is not None:
            full_video_inputs, full_video_metadatas = zip(*full_packed_video_inputs)
            full_video_inputs = list(full_video_inputs)
            full_video_metadatas = list(full_video_metadatas)
        else:
            full_video_inputs = None
            full_video_metadatas = None

        full_inputs = self.processing_class(
            text=full_texts,
            images=full_image_inputs,
            videos=full_video_inputs,
            video_metadata=full_video_metadatas,
            do_resize=False,
            return_tensors="pt",
            padding=True,
            **full_video_kwargs,
        )
        full_inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in full_inputs.items()
        }
        self._stage_debug_log(
            "search:tokenize_full_done",
            f"elapsed={time.perf_counter()-t_tokenize:.3f}s",
        )

        prompt_image_inputs, prompt_packed_video_inputs, prompt_video_kwargs = cached_process_vision_info(
            normalized_prompt,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if prompt_packed_video_inputs is not None:
            prompt_video_inputs, prompt_video_metadatas = zip(*prompt_packed_video_inputs)
            prompt_video_inputs = list(prompt_video_inputs)
            prompt_video_metadatas = list(prompt_video_metadatas)
        else:
            prompt_video_inputs = None
            prompt_video_metadatas = None

        prompt_inputs = self.processing_class(
            text=prompt_texts,
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            video_metadata=prompt_video_metadatas,
            do_resize=False,
            return_tensors="pt",
            padding=True,
            **prompt_video_kwargs,
        )
        prompt_inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in prompt_inputs.items()
        }
        self._stage_debug_log(
            "search:tokenize_prompt_done",
            f"elapsed={time.perf_counter()-t_tokenize:.3f}s",
        )

        full_input_ids = full_inputs["input_ids"]
        full_attention_mask = full_inputs["attention_mask"]
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        prompt_lengths = prompt_mask.sum(dim=1).tolist()
        t_latent = time.perf_counter()
        self._stage_debug_log("search:latent_meta_start")
        sqr_latent_batch = self._populate_latent_improve_meta(
            reward_meta=reward_meta,
            queries=queries,
            gt_videos=gt_videos,
            hard_negative_ids_list=hard_negative_ids_list,
            full_inputs=full_inputs,
            full_input_ids=full_input_ids,
            full_attention_mask=full_attention_mask,
            prompt_lengths=prompt_lengths,
        )
        self._stage_debug_log(
            "search:latent_meta_done",
            f"elapsed={time.perf_counter()-t_latent:.3f}s",
        )
        assistant_mask = self._build_assistant_mask(full_input_ids)

        completion_ids_list = []
        completion_mask_list = []
        completion_attn_list = []
        for i in range(full_input_ids.size(0)):
            plen = int(prompt_lengths[i])
            comp_ids = full_input_ids[i, plen:]
            comp_attn = full_attention_mask[i, plen:]
            comp_mask = assistant_mask[i, plen:].int()
            completion_ids_list.append(comp_ids)
            completion_mask_list.append(comp_mask)
            completion_attn_list.append(comp_attn)

        completion_ids = pad(completion_ids_list, padding_value=self.pad_token_id)
        completion_mask = pad(completion_mask_list, padding_value=0)
        completion_attn = pad(completion_attn_list, padding_value=0)
        attention_mask = torch.cat([prompt_mask, completion_attn], dim=1)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        t_logps = time.perf_counter()
        self._stage_debug_log("search:logps_start")
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                self._stage_debug_log("search:old_logps_start")
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    messages=messages,
                    pixel_values=full_inputs.get("pixel_values"),
                    image_grid_thw=full_inputs.get("image_grid_thw"),
                    pixel_values_videos=full_inputs.get("pixel_values_videos"),
                    video_grid_thw=full_inputs.get("video_grid_thw"),
                )
                self._stage_debug_log("search:old_logps_done")
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    self._stage_debug_log("search:ref_logps_start")
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        messages=messages,
                        pixel_values=full_inputs.get("pixel_values"),
                        image_grid_thw=full_inputs.get("image_grid_thw"),
                        pixel_values_videos=full_inputs.get("pixel_values_videos"),
                        video_grid_thw=full_inputs.get("video_grid_thw"),
                    )
                    self._stage_debug_log("search:ref_logps_done")
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        self._stage_debug_log("search:ref_logps_adapter_start")
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            messages=messages,
                            pixel_values=full_inputs.get("pixel_values"),
                            image_grid_thw=full_inputs.get("image_grid_thw"),
                            pixel_values_videos=full_inputs.get("pixel_values_videos"),
                            video_grid_thw=full_inputs.get("video_grid_thw"),
                        )
                        self._stage_debug_log("search:ref_logps_adapter_done")
            else:
                ref_per_token_logps = None
        self._stage_debug_log(
            "search:logps_done",
            f"elapsed={time.perf_counter()-t_logps:.3f}s",
        )

        # Calculate rewards
        t_reward = time.perf_counter()
        self._stage_debug_log("search:rewards_start")
        rewards_per_func = self._calculate_rewards(
            completions_text,
            solutions,
            problem_types,
            reward_meta=reward_meta,
        )
        self._stage_debug_log(
            "search:rewards_done",
            f"elapsed={time.perf_counter()-t_reward:.3f}s",
        )

        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        reward_debug_data = None
        if self._should_reward_debug():
            t_reward_local = time.perf_counter()
            self._stage_debug_log("search:reward_debug_local_start")
            rewards_per_func_local = self._calculate_rewards_local(
                completions_text,
                solutions,
                problem_types,
                reward_meta=reward_meta,
            )
            self._stage_debug_log(
                "search:reward_debug_local_done",
                f"elapsed={time.perf_counter()-t_reward_local:.3f}s",
            )
            rewards_local = (
                rewards_per_func_local * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
            max_samples = int(self.args.reward_debug_max_samples or 0)
            if max_samples <= 0:
                max_samples = len(reward_meta)
            reward_debug_data = []
            for idx in range(min(len(reward_meta), max_samples)):
                meta = reward_meta[idx]
                completion_text = str(completions_text[idx] or "")
                pred_start, pred_end = _extract_start_end(completion_text)
                gt_start, gt_end = _normalize_gt_span(meta.get("gt_time", None))
                time_iou = _compute_temporal_iou(pred_start, pred_end, gt_start, gt_end)
                func_values = {}
                func_values_weighted = {}
                for i, name in enumerate(self.reward_func_names):
                    value = float(rewards_per_func_local[idx, i].item())
                    weight = float(self.reward_weights[i].detach().cpu().item())
                    func_values[name] = value
                    func_values_weighted[name] = value * weight
                turns_info = []
                for t_idx, t in enumerate(meta.get("turns", [])):
                    topk_ids = t.get("retrieved_ids", [])
                    topk_head = topk_ids[:5] if isinstance(topk_ids, list) else []
                    retrieved_id = str(t.get("retrieved_id", "") or "")
                    gt_video = str(meta.get("gt_video", "") or "")
                    expected_answer = (
                        "matched"
                        if retrieved_id and gt_video and retrieved_id == gt_video
                        else "not_matched"
                    )
                    answer = str(t.get("answer", "") or "")
                    turns_info.append(
                        {
                            "turn": t_idx,
                            "shadow": bool(t.get("shadow")),
                            "retrieval_query": t.get("retrieval_query", ""),
                            "gt_rank": int(t.get("gt_rank", -1) or -1),
                            "gt_rank_full": int(t.get("gt_rank_full", -1) or -1),
                            "retrieved_id": retrieved_id,
                            "answer": answer,
                            "expected_answer": expected_answer,
                            "answer_binary_correct": bool(answer == expected_answer),
                            "correct_accept": bool(t.get("correct_accept")),
                            "correct_reject": bool(t.get("correct_reject")),
                            "topk_head": topk_head,
                        }
                    )
                final_turn = None
                for t in reversed(turns_info):
                    if not t.get("shadow"):
                        final_turn = t
                        break
                if final_turn is None and turns_info:
                    final_turn = turns_info[-1]
                reward_debug_data.append(
                    {
                        "idx": idx,
                        "qid": str(inputs[idx].get("qid", "") or ""),
                        "query": queries[idx],
                        "gt_video": gt_videos[idx],
                        "completion_text": completion_text,
                        "reward_total": float(rewards_local[idx].item()),
                        "reward_funcs": func_values,
                        "reward_funcs_weighted": func_values_weighted,
                        "turns": turns_info,
                        "answer_pred": "" if final_turn is None else final_turn.get("answer", ""),
                        "answer_expected": ""
                        if final_turn is None
                        else final_turn.get("expected_answer", ""),
                        "answer_binary_correct": bool(
                            final_turn is not None and final_turn.get("answer_binary_correct")
                        ),
                        "pred_start": pred_start,
                        "pred_end": pred_end,
                        "gt_start": gt_start,
                        "gt_end": gt_end,
                        "iou": time_iou,
                        "improve_delta": float(meta.get("improve_delta", 0.0)),
                        "improve_delta_raw": float(meta.get("improve_delta_raw", 0.0)),
                        "improve_has_refine": bool(meta.get("improve_has_refine", False)),
                        "margin_delta": float(meta.get("margin_delta", 0.0)),
                        "margin_delta_raw": float(meta.get("margin_delta_raw", 0.0)),
                        "margin_has_neg": bool(meta.get("margin_has_neg", False)),
                        "margin_neg_count": int(meta.get("margin_neg_count", 0) or 0),
                        "query_refine_quality": float(
                            meta.get("query_refine_quality", 0.0)
                        ),
                        "query_refine_quality_raw": float(
                            meta.get("query_refine_quality_raw", 0.0)
                        ),
                        "query_refine_quality_has_neg": bool(
                            meta.get("query_refine_quality_has_neg", False)
                        ),
                        "query_refine_quality_neg_count": int(
                            meta.get("query_refine_quality_neg_count", 0) or 0
                        ),
                        "latent_rollout_depth": int(meta.get("latent_rollout_depth", 0) or 0),
                        "sqr_old_norm_mean": float(meta.get("sqr_old_norm_mean", 0.0)),
                        "sqr_action_norm_mean": float(meta.get("sqr_action_norm_mean", 0.0)),
                        "sqr_action_dist2_mean": float(meta.get("sqr_action_dist2_mean", 0.0)),
                        "sqr_mask_sum": float(meta.get("sqr_mask_sum", 0.0)),
                    }
                )
            # Pad to full batch length so shuffle_sequence_dict won't index out of range.
            # Use idx=None so downstream logging skips these entries safely.
            if len(reward_debug_data) < len(reward_meta):
                reward_debug_data.extend(
                    [{"idx": None} for _ in range(len(reward_meta) - len(reward_debug_data))]
                )

        self._stage_debug_log("search:advantage_start")
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompt_texts),
            (self.accelerator.process_index + 1) * len(prompt_texts),
        )
        advantages = advantages[process_slice]
        rewards_local_for_adv = rewards[process_slice]
        mean_grouped_rewards_local = mean_grouped_rewards[process_slice]
        std_grouped_rewards_local = std_grouped_rewards[process_slice]

        if reward_debug_data:
            for row_idx, item in enumerate(reward_debug_data):
                if item.get("idx") is None:
                    continue
                if row_idx >= advantages.size(0):
                    continue
                item["batch_row_idx"] = int(row_idx)
                item["reward_total_global"] = float(
                    rewards_local_for_adv[row_idx].detach().cpu().item()
                )
                item["reward_group_mean"] = float(
                    mean_grouped_rewards_local[row_idx].detach().cpu().item()
                )
                item["reward_group_std"] = float(
                    std_grouped_rewards_local[row_idx].detach().cpu().item()
                )
                item["advantage"] = float(advantages[row_idx].detach().cpu().item())
        self._stage_debug_log("search:advantage_done")

        # Log metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )
        self._metrics[mode]["completions/clipped_ratio"].append(0.0)

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Search-specific metrics
        turns = [
            len([t for t in m.get("turns", []) if not t.get("shadow")])
            for m in reward_meta
        ]
        if turns:
            self._metrics[mode]["search/turns_mean"].append(
                sum(turns) / len(turns)
            )
            self._metrics[mode]["search/final_match_rate"].append(
                sum(1.0 for m in reward_meta if m.get("final_correct")) / len(turns)
            )
            self._metrics[mode]["search/middle_correct_rate"].append(
                sum(m.get("middle_correct_rate", 0.0) for m in reward_meta) / len(turns)
            )
            improve_vals = [float(m.get("improve_delta", 0.0)) for m in reward_meta]
            improve_raw = [float(m.get("improve_delta_raw", 0.0)) for m in reward_meta]
            refine_hits = [1.0 if bool(m.get("improve_has_refine", False)) else 0.0 for m in reward_meta]
            margin_vals = [float(m.get("margin_delta", 0.0)) for m in reward_meta]
            margin_raw = [float(m.get("margin_delta_raw", 0.0)) for m in reward_meta]
            margin_has_neg = [1.0 if bool(m.get("margin_has_neg", False)) else 0.0 for m in reward_meta]
            qref_vals = [float(m.get("query_refine_quality", 0.0)) for m in reward_meta]
            qref_raw = [float(m.get("query_refine_quality_raw", 0.0)) for m in reward_meta]
            qref_has_neg = [
                1.0 if bool(m.get("query_refine_quality_has_neg", False)) else 0.0
                for m in reward_meta
            ]
            rollout_depth_vals = [
                float(m.get("latent_rollout_depth", 0) or 0) for m in reward_meta
            ]
            sqr_old_norm_vals = [
                float(m.get("sqr_old_norm_mean", 0.0)) for m in reward_meta
            ]
            sqr_action_norm_vals = [
                float(m.get("sqr_action_norm_mean", 0.0)) for m in reward_meta
            ]
            sqr_action_dist2_vals = [
                float(m.get("sqr_action_dist2_mean", 0.0)) for m in reward_meta
            ]
            if improve_vals:
                self._metrics[mode]["debug/improve_delta_mean"].append(
                    sum(improve_vals) / len(improve_vals)
                )
                self._metrics[mode]["debug/improve_delta_raw_mean"].append(
                    sum(improve_raw) / len(improve_raw)
                )
                self._metrics[mode]["debug/improve_positive_rate"].append(
                    sum(1.0 for x in improve_vals if x > 0.0) / len(improve_vals)
                )
                self._metrics[mode]["debug/refine_presence_rate"].append(
                    sum(refine_hits) / len(refine_hits)
                )
            if margin_vals:
                self._metrics[mode]["debug/margin_delta_mean"].append(
                    sum(margin_vals) / len(margin_vals)
                )
                self._metrics[mode]["debug/margin_delta_raw_mean"].append(
                    sum(margin_raw) / len(margin_raw)
                )
                self._metrics[mode]["debug/margin_positive_rate"].append(
                    sum(1.0 for x in margin_vals if x > 0.0) / len(margin_vals)
                )
                self._metrics[mode]["debug/margin_has_neg_rate"].append(
                    sum(margin_has_neg) / len(margin_has_neg)
                )
            if qref_vals:
                self._metrics[mode]["debug/query_refine_quality_mean"].append(
                    sum(qref_vals) / len(qref_vals)
                )
                self._metrics[mode]["debug/query_refine_quality_raw_mean"].append(
                    sum(qref_raw) / len(qref_raw)
                )
                self._metrics[mode]["debug/query_refine_quality_positive_rate"].append(
                    sum(1.0 for x in qref_vals if x > 0.0) / len(qref_vals)
                )
                self._metrics[mode]["debug/query_refine_quality_has_neg_rate"].append(
                    sum(qref_has_neg) / len(qref_has_neg)
                )
            if rollout_depth_vals:
                self._metrics[mode]["debug/latent_rollout_depth_mean"].append(
                    sum(rollout_depth_vals) / len(rollout_depth_vals)
                )
                self._metrics[mode]["debug/latent_rollout_depth_max"].append(
                    max(rollout_depth_vals)
                )
            if sqr_old_norm_vals:
                self._metrics[mode]["debug/sqr_old_norm_mean"].append(
                    sum(sqr_old_norm_vals) / len(sqr_old_norm_vals)
                )
            if sqr_action_norm_vals:
                self._metrics[mode]["debug/sqr_action_norm_mean"].append(
                    sum(sqr_action_norm_vals) / len(sqr_action_norm_vals)
                )
            if sqr_action_dist2_vals:
                self._metrics[mode]["debug/sqr_action_dist2_mean"].append(
                    sum(sqr_action_dist2_vals) / len(sqr_action_dist2_vals)
                )

        output = {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "attention_mask": attention_mask,
            "action_mask": completion_mask,
            "advantages": advantages,
            "full_input_ids": full_input_ids,
            "full_attention_mask": full_attention_mask,
            "prompt_lengths": torch.tensor(
                prompt_lengths, device=device, dtype=torch.long
            ),
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in full_inputs:
            output["pixel_values"] = full_inputs["pixel_values"].to(device)
        if "image_grid_thw" in full_inputs:
            output["image_grid_thw"] = full_inputs["image_grid_thw"].to(device)
        if "pixel_values_videos" in full_inputs:
            output["pixel_values_videos"] = full_inputs["pixel_values_videos"].to(
                device
            )
        if "video_grid_thw" in full_inputs:
            output["video_grid_thw"] = full_inputs["video_grid_thw"].to(device)
        if reward_debug_data:
            output["reward_debug"] = reward_debug_data
        if sqr_latent_batch:
            output.update(sqr_latent_batch)
        if self._use_infonce_latent_aux_loss:
            output["latent_query_texts"] = list(queries)
            output["latent_gt_videos"] = list(gt_videos)
            output["latent_hard_negative_ids"] = [list(x) for x in hard_negative_ids_list]

        self._stage_debug_log("search:return_output")
        return output

    def _compute_sqr_latent_aux_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        advantages: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, dict[str, Any]]]:
        if not self._should_apply_sqr_latent_loss():
            return None

        old_mean = inputs.get("sqr_latent_old_mean")
        actions = inputs.get("sqr_latent_action")
        logp_old = inputs.get("sqr_latent_logp_old")
        latent_mask = inputs.get("sqr_latent_mask")
        if (
            old_mean is None
            or actions is None
            or logp_old is None
            or latent_mask is None
            or not torch.is_tensor(old_mean)
            or not torch.is_tensor(actions)
            or not torch.is_tensor(logp_old)
            or not torch.is_tensor(latent_mask)
        ):
            return None

        full_input_ids = inputs.get("full_input_ids")
        if full_input_ids is None:
            full_input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        full_attention_mask = inputs.get("full_attention_mask")
        if full_attention_mask is None:
            full_attention_mask = inputs.get("attention_mask")
        if full_attention_mask is None:
            full_attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        prompt_lengths = inputs.get("prompt_lengths")
        if prompt_lengths is None:
            prompt_lengths = inputs["prompt_mask"].sum(dim=1)
        if prompt_lengths.dim() == 0:
            prompt_lengths = prompt_lengths.unsqueeze(0)

        fwd_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "output_hidden_states": True,
        }
        use_latent_forward_vision = _env_flag("GRPO_LATENT_FORWARD_USE_VISION", True)
        if use_latent_forward_vision:
            for key in (
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
            ):
                if key in inputs and torch.is_tensor(inputs[key]):
                    fwd_inputs[key] = inputs[key]

        outputs = model(**fwd_inputs)
        hidden = outputs.hidden_states[-1].to(dtype=torch.float32)
        projector, gate = self._get_refine_modules(model)
        try:
            projector_dtype = next(projector.parameters()).dtype
        except StopIteration:
            projector_dtype = hidden.dtype
        gate_dtype = hidden.dtype
        if gate is not None:
            try:
                gate_dtype = next(gate.parameters()).dtype
            except StopIteration:
                gate_dtype = hidden.dtype

        batch_size = int(full_input_ids.size(0))
        seq_len = int(full_input_ids.size(1))
        token_pos = torch.arange(seq_len, device=full_input_ids.device)
        rollout_depth = int(max(1, self._refine_rollout_depth))
        train_depth = self._resolve_sqr_train_depth(rollout_depth)
        theta_rows: list[torch.Tensor] = []

        for idx in range(batch_size):
            ids_row = full_input_ids[idx]
            prompt_len = int(prompt_lengths[idx].item())
            pos_mask = self._make_refine_mask(ids_row)
            pos_mask &= token_pos >= prompt_len
            refine_positions = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
            if refine_positions.numel() > 0:
                h_ref = hidden[idx, refine_positions].to(dtype=torch.float32)
                delta = projector(h_ref.to(dtype=projector_dtype)).to(dtype=torch.float32)
                if self._use_refine_gate and gate is not None:
                    alpha = torch.sigmoid(gate(h_ref.to(dtype=gate_dtype))).to(dtype=torch.float32)
                    update = delta * alpha
                else:
                    update = delta
                row_rollout_inputs = self._build_row_rollout_inputs(fwd_inputs, idx)
                theta = self._rollout_refine_latents(
                    model=model,
                    first_update=update,
                    rollout_depth=rollout_depth,
                    row_model_inputs=row_rollout_inputs,
                )
            else:
                theta = torch.zeros(
                    (1, int(old_mean.size(-1))),
                    device=hidden.device,
                    dtype=torch.float32,
                )

            theta = theta[:train_depth]
            if theta.size(0) < train_depth:
                pad_count = int(train_depth - theta.size(0))
                theta = torch.cat([theta, theta[-1:].repeat(pad_count, 1)], dim=0)
            theta_rows.append(theta)

        theta_mean = torch.stack(theta_rows, dim=0)
        target_depth = min(
            int(theta_mean.size(1)),
            int(old_mean.size(1)),
            int(actions.size(1)),
            int(logp_old.size(1)),
            int(latent_mask.size(1)),
        )
        target_dim = min(int(theta_mean.size(2)), int(old_mean.size(2)), int(actions.size(2)))
        theta_mean = theta_mean[:, :target_depth, :target_dim]
        old_mean = old_mean[:, :target_depth, :target_dim]
        actions = actions[:, :target_depth, :target_dim]
        logp_old = logp_old[:, :target_depth]
        latent_mask = latent_mask[:, :target_depth].to(dtype=torch.float32)

        valid_count = latent_mask.sum()
        if valid_count.item() <= 0:
            return None

        sigma2 = float(self._sqr_latent_sigma ** 2)
        dist2_theta = ((actions - theta_mean) ** 2).mean(dim=-1)
        dist2_old = ((actions - old_mean) ** 2).mean(dim=-1)
        logp_theta = -dist2_theta / (2.0 * sigma2)
        ratio = torch.exp(logp_theta - logp_old)
        ratio_clip = torch.clamp(
            ratio,
            1.0 - self._sqr_latent_clip_epsilon,
            1.0 + self._sqr_latent_clip_epsilon,
        )

        adv = advantages.unsqueeze(1).to(dtype=ratio.dtype)
        surrogate_1 = ratio * adv
        surrogate_2 = ratio_clip * adv
        per_step_loss = -torch.min(surrogate_1, surrogate_2)
        latent_loss = (per_step_loss * latent_mask).sum() / valid_count.clamp(min=1.0)

        clip_hits = ((ratio < (1.0 - self._sqr_latent_clip_epsilon)) | (ratio > (1.0 + self._sqr_latent_clip_epsilon))).to(dtype=latent_mask.dtype)
        ratio_mean = (ratio * latent_mask).sum() / valid_count
        clip_rate = (clip_hits * latent_mask).sum() / valid_count
        dist2_theta_mean = (dist2_theta * latent_mask).sum() / valid_count
        dist2_old_mean = (dist2_old * latent_mask).sum() / valid_count
        cosine = F.cosine_similarity(theta_mean, old_mean, dim=-1)
        cosine_mean = (cosine * latent_mask).sum() / valid_count

        sample_denom = latent_mask.sum(dim=1).clamp(min=1.0)
        sample_ratio_mean = (ratio * latent_mask).sum(dim=1) / sample_denom
        sample_dist2_theta = (dist2_theta * latent_mask).sum(dim=1) / sample_denom
        sample_dist2_old = (dist2_old * latent_mask).sum(dim=1) / sample_denom
        sample_clip_rate = (clip_hits * latent_mask).sum(dim=1) / sample_denom
        sample_cosine = (cosine * latent_mask).sum(dim=1) / sample_denom

        stats = {
            "ratio_mean": ratio_mean.detach(),
            "clip_rate": clip_rate.detach(),
            "dist2_theta_mean": dist2_theta_mean.detach(),
            "dist2_old_mean": dist2_old_mean.detach(),
            "cosine_mean": cosine_mean.detach(),
            "sample_ratio_mean": sample_ratio_mean.detach(),
            "sample_dist2_theta": sample_dist2_theta.detach(),
            "sample_dist2_old": sample_dist2_old.detach(),
            "sample_clip_rate": sample_clip_rate.detach(),
            "sample_cosine": sample_cosine.detach(),
            "mask_sum": valid_count.detach(),
        }
        return latent_loss, stats

    def _compute_latent_infonce_aux_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
    ) -> Optional[tuple[torch.Tensor, dict[str, Any]]]:
        if not self._should_apply_infonce_latent_aux_loss():
            return None

        queries = inputs.get("latent_query_texts")
        gt_videos = inputs.get("latent_gt_videos")
        hard_negative_ids_list = inputs.get("latent_hard_negative_ids")
        if (
            queries is None
            or gt_videos is None
            or hard_negative_ids_list is None
            or not isinstance(queries, list)
            or not isinstance(gt_videos, list)
            or not isinstance(hard_negative_ids_list, list)
        ):
            return None

        full_input_ids = inputs.get("full_input_ids")
        if full_input_ids is None:
            full_input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        full_attention_mask = inputs.get("full_attention_mask")
        if full_attention_mask is None:
            full_attention_mask = inputs.get("attention_mask")
        if full_attention_mask is None:
            full_attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        prompt_lengths = inputs.get("prompt_lengths")
        if prompt_lengths is None:
            prompt_lengths = inputs["prompt_mask"].sum(dim=1)
        if prompt_lengths.dim() == 0:
            prompt_lengths = prompt_lengths.unsqueeze(0)

        batch_size = int(full_input_ids.size(0))
        if len(queries) != batch_size or len(gt_videos) != batch_size:
            return None
        if len(hard_negative_ids_list) != batch_size:
            return None

        fwd_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "output_hidden_states": True,
        }
        use_latent_forward_vision = _env_flag("GRPO_LATENT_FORWARD_USE_VISION", True)
        if use_latent_forward_vision:
            for key in (
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
            ):
                if key in inputs and torch.is_tensor(inputs[key]):
                    fwd_inputs[key] = inputs[key]

        outputs = model(**fwd_inputs)
        hidden = outputs.hidden_states[-1].to(dtype=torch.float32)
        projector, gate = self._get_refine_modules(model)
        try:
            projector_dtype = next(projector.parameters()).dtype
        except StopIteration:
            projector_dtype = hidden.dtype
        gate_dtype = hidden.dtype
        if gate is not None:
            try:
                gate_dtype = next(gate.parameters()).dtype
            except StopIteration:
                gate_dtype = hidden.dtype

        device = hidden.device
        seq_len = int(full_input_ids.size(1))
        token_pos = torch.arange(seq_len, device=device)
        rollout_depth = int(max(1, self._refine_rollout_depth))
        train_depth = self._resolve_infonce_train_depth(rollout_depth)
        temp = float(self._infonce_latent_temperature)
        infonce_mode = str(self._infonce_latent_mode)

        batch_gt_rows = []
        for vid in gt_videos:
            row = self._video_id_to_row.get(str(vid).strip())
            batch_gt_rows.append(int(row) if row is not None else -1)

        video_emb_cache: dict[int, torch.Tensor] = {}

        def _get_video_emb(row: int) -> torch.Tensor:
            emb = video_emb_cache.get(int(row))
            if emb is not None:
                return emb
            emb = torch.from_numpy(
                np.array(self._video_embeddings[int(row)], dtype=np.float32, copy=True)
            ).to(device=device, dtype=torch.float32)
            emb = _l2_norm(emb.unsqueeze(0)).squeeze(0)
            video_emb_cache[int(row)] = emb
            return emb

        per_sample_losses: list[torch.Tensor] = []
        per_sample_valid: list[torch.Tensor] = []
        per_sample_p_after: list[torch.Tensor] = []
        per_sample_p_before: list[torch.Tensor] = []
        per_sample_neg_count: list[torch.Tensor] = []

        query_embedder_outer_ctx = nullcontext()
        query_embedder_params_gathered = False
        if (
            self._use_query_embedder_path
            and self._query_embedder_needs_param_gather
            and self._query_embedder_model is not None
        ):
            query_embedder_outer_ctx = _gather_module_params_ctx(
                self._query_embedder_model
            )
            query_embedder_params_gathered = True

        with query_embedder_outer_ctx:
            for idx in range(batch_size):
                query_text = str(queries[idx]).strip()
                gt_video = str(gt_videos[idx]).strip()
                valid = torch.zeros([], device=device, dtype=torch.float32)
                sample_loss = torch.zeros([], device=device, dtype=torch.float32)
                sample_p_after = torch.zeros([], device=device, dtype=torch.float32)
                sample_p_before = torch.zeros([], device=device, dtype=torch.float32)
                sample_neg_count = torch.zeros([], device=device, dtype=torch.float32)

                qmeta = self._query_meta_by_query.get(query_text)
                gt_row = self._video_id_to_row.get(gt_video) if gt_video else None
                if not query_text or not gt_video or qmeta is None or gt_row is None:
                    per_sample_losses.append(sample_loss)
                    per_sample_valid.append(valid)
                    per_sample_p_after.append(sample_p_after)
                    per_sample_p_before.append(sample_p_before)
                    per_sample_neg_count.append(sample_neg_count)
                    continue

                q_row = int(qmeta[0])
                q_orig = torch.from_numpy(
                    np.array(self._query_embeddings[q_row], dtype=np.float32, copy=True)
                ).to(device=device, dtype=torch.float32)
                q_orig = _l2_norm(q_orig.unsqueeze(0)).squeeze(0)
                gt_emb = _get_video_emb(int(gt_row))

                ids_row = full_input_ids[idx]
                prompt_len = int(prompt_lengths[idx].item())
                pos_mask = self._make_refine_mask(ids_row)
                pos_mask &= token_pos >= prompt_len
                refine_positions = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                if refine_positions.numel() == 0:
                    per_sample_losses.append(sample_loss)
                    per_sample_valid.append(valid)
                    per_sample_p_after.append(sample_p_after)
                    per_sample_p_before.append(sample_p_before)
                    per_sample_neg_count.append(sample_neg_count)
                    continue

                h_ref = hidden[idx, refine_positions].to(dtype=torch.float32)
                delta = projector(h_ref.to(dtype=projector_dtype)).to(dtype=torch.float32)
                if self._use_refine_gate and gate is not None:
                    alpha = torch.sigmoid(gate(h_ref.to(dtype=gate_dtype))).to(dtype=torch.float32)
                    update = delta * alpha
                else:
                    update = delta
                row_rollout_inputs = self._build_row_rollout_inputs(fwd_inputs, idx)
                update = self._rollout_refine_latents(
                    model=model,
                    first_update=update,
                    rollout_depth=rollout_depth,
                    row_model_inputs=row_rollout_inputs,
                )
                update = update[:train_depth]
                if update.size(0) < train_depth:
                    pad_count = int(train_depth - update.size(0))
                    update = torch.cat([update, update[-1:].repeat(pad_count, 1)], dim=0)

                q_final, _ = self._build_q_final_from_update(
                    model=model,
                    q_orig=q_orig,
                    update=update,
                    query_text=query_text,
                    query_embedder_params_gathered=query_embedder_params_gathered,
                    enable_grad_through_query_embedder=True,
                )

                neg_rows: set[int] = set()
                hard_negs = hard_negative_ids_list[idx]
                if isinstance(hard_negs, list):
                    for hard_vid in hard_negs:
                        hv = str(hard_vid).strip()
                        if not hv or hv == gt_video:
                            continue
                        hard_row = self._video_id_to_row.get(hv)
                        if hard_row is not None and int(hard_row) != int(gt_row):
                            neg_rows.add(int(hard_row))
                for j, other_row in enumerate(batch_gt_rows):
                    if j == idx or other_row < 0:
                        continue
                    if int(other_row) == int(gt_row):
                        continue
                    other_gt = str(gt_videos[j]).strip()
                    if other_gt and other_gt == gt_video:
                        continue
                    neg_rows.add(int(other_row))

                if not neg_rows:
                    per_sample_losses.append(sample_loss)
                    per_sample_valid.append(valid)
                    per_sample_p_after.append(sample_p_after)
                    per_sample_p_before.append(sample_p_before)
                    per_sample_neg_count.append(sample_neg_count)
                    continue

                neg_embs = torch.stack(
                    [_get_video_emb(r) for r in sorted(neg_rows)], dim=0
                )
                sim_pos_after = torch.dot(q_final, gt_emb)
                sim_neg_after = torch.matmul(neg_embs, q_final)
                logits_after = torch.cat([sim_pos_after.unsqueeze(0), sim_neg_after], dim=0) / temp
                target = torch.zeros((1,), device=device, dtype=torch.long)
                loss_after = F.cross_entropy(logits_after.unsqueeze(0), target)
                p_after = torch.softmax(logits_after, dim=0)[0]

                sim_pos_before = torch.dot(q_orig, gt_emb)
                sim_neg_before = torch.matmul(neg_embs, q_orig)
                logits_before = torch.cat([sim_pos_before.unsqueeze(0), sim_neg_before], dim=0) / temp
                loss_before = F.cross_entropy(logits_before.unsqueeze(0), target)
                p_before = torch.softmax(logits_before, dim=0)[0]

                if infonce_mode == "delta":
                    sample_loss = loss_after - loss_before
                else:
                    sample_loss = loss_after
                sample_p_after = p_after
                sample_p_before = p_before
                sample_neg_count = torch.tensor(float(len(neg_rows)), device=device, dtype=torch.float32)
                valid = torch.ones([], device=device, dtype=torch.float32)

                per_sample_losses.append(sample_loss)
                per_sample_valid.append(valid)
                per_sample_p_after.append(sample_p_after)
                per_sample_p_before.append(sample_p_before)
                per_sample_neg_count.append(sample_neg_count)

        losses = torch.stack(per_sample_losses, dim=0)
        valid_mask = torch.stack(per_sample_valid, dim=0)
        p_after_vec = torch.stack(per_sample_p_after, dim=0)
        p_before_vec = torch.stack(per_sample_p_before, dim=0)
        neg_count_vec = torch.stack(per_sample_neg_count, dim=0)

        valid_count = valid_mask.sum()
        if valid_count.item() <= 0:
            return None

        aux_loss = (losses * valid_mask).sum() / valid_count
        p_after_mean = (p_after_vec * valid_mask).sum() / valid_count
        p_before_mean = (p_before_vec * valid_mask).sum() / valid_count
        p_delta_mean = ((p_after_vec - p_before_vec) * valid_mask).sum() / valid_count
        neg_count_mean = (neg_count_vec * valid_mask).sum() / valid_count

        stats = {
            "valid_count": valid_count.detach(),
            "p_after_mean": p_after_mean.detach(),
            "p_before_mean": p_before_mean.detach(),
            "p_delta_mean": p_delta_mean.detach(),
            "neg_count_mean": neg_count_mean.detach(),
            "sample_loss": losses.detach(),
            "sample_valid": valid_mask.detach(),
            "sample_p_after": p_after_vec.detach(),
            "sample_p_before": p_before_vec.detach(),
            "sample_p_delta": (p_after_vec - p_before_vec).detach(),
        }
        return aux_loss, stats

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        action_mask = inputs.get("action_mask", completion_mask)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=self.top_entropy_quantile < 1.0,
            messages=inputs.get("messages"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = get_high_entropy_mask(
                entropies, action_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * action_mask).sum(
                -1
            ) / action_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * action_mask).sum(-1)
                / action_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * action_mask
            ).sum() / action_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * action_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        sqr_stats = None
        sqr_loss_metric_t = torch.zeros([], device=loss.device, dtype=torch.float32)
        sqr_loss_weighted_metric_t = torch.zeros([], device=loss.device, dtype=torch.float32)
        sqr_applied_metric_t = torch.zeros([], device=loss.device, dtype=torch.float32)
        sqr_aux = self._compute_sqr_latent_aux_loss(
            model=model,
            inputs=inputs,
            advantages=advantages,
        )
        if sqr_aux is not None:
            sqr_loss, sqr_stats = sqr_aux
            loss = loss + self._sqr_latent_loss_weight * sqr_loss
            sqr_loss_metric_t = sqr_loss.detach()
            sqr_loss_weighted_metric_t = (
                self._sqr_latent_loss_weight * sqr_loss
            ).detach()
            sqr_applied_metric_t = torch.ones([], device=loss.device, dtype=torch.float32)

        infonce_stats = None
        infonce_loss_metric_t = torch.zeros([], device=loss.device, dtype=torch.float32)
        infonce_loss_weighted_metric_t = torch.zeros([], device=loss.device, dtype=torch.float32)
        infonce_applied_metric_t = torch.zeros([], device=loss.device, dtype=torch.float32)
        infonce_aux = self._compute_latent_infonce_aux_loss(
            model=model,
            inputs=inputs,
        )
        if infonce_aux is not None:
            infonce_loss, infonce_stats = infonce_aux
            loss = loss + self._infonce_latent_loss_weight * infonce_loss
            infonce_loss_metric_t = infonce_loss.detach()
            infonce_loss_weighted_metric_t = (
                self._infonce_latent_loss_weight * infonce_loss
            ).detach()
            infonce_applied_metric_t = torch.ones([], device=loss.device, dtype=torch.float32)

        if self._should_reward_debug():
            reward_debug = inputs.get("reward_debug")
            if reward_debug:
                token_counts = action_mask.sum(-1).clamp(min=1.0)
                mean_logprob = (per_token_logps * action_mask).sum(-1) / token_counts
                sum_logprob = (per_token_logps * action_mask).sum(-1)
                mean_kl = None
                if self.beta != 0.0:
                    mean_kl = (per_token_kl * action_mask).sum(-1) / token_counts
                for row_idx, item in enumerate(reward_debug):
                    sample_idx = item.get("idx")
                    if sample_idx is None:
                        continue
                    try:
                        sample_idx = int(sample_idx)
                    except (TypeError, ValueError):
                        continue
                    if row_idx < 0 or row_idx >= mean_logprob.size(0):
                        continue
                    item["batch_row_idx"] = int(row_idx)
                    item["sample_idx"] = int(sample_idx)
                    item["mean_logprob"] = float(
                        mean_logprob[row_idx].detach().cpu().item()
                    )
                    item["sum_logprob"] = float(
                        sum_logprob[row_idx].detach().cpu().item()
                    )
                    item["advantage"] = float(advantages[row_idx].detach().cpu().item())
                    if mean_kl is not None:
                        item["mean_kl"] = float(mean_kl[row_idx].detach().cpu().item())
                    if sqr_stats is not None:
                        item["sqr_ratio_mean"] = float(
                            sqr_stats["sample_ratio_mean"][row_idx].detach().cpu().item()
                        )
                        item["sqr_dist2_theta_mean"] = float(
                            sqr_stats["sample_dist2_theta"][row_idx].detach().cpu().item()
                        )
                        item["sqr_dist2_old_mean"] = float(
                            sqr_stats["sample_dist2_old"][row_idx].detach().cpu().item()
                        )
                        item["sqr_clip_rate"] = float(
                            sqr_stats["sample_clip_rate"][row_idx].detach().cpu().item()
                        )
                        item["sqr_cosine_mean"] = float(
                            sqr_stats["sample_cosine"][row_idx].detach().cpu().item()
                        )
                    if infonce_stats is not None:
                        item["infonce_aux_loss"] = float(
                            infonce_stats["sample_loss"][row_idx].detach().cpu().item()
                        )
                        item["infonce_aux_valid"] = float(
                            infonce_stats["sample_valid"][row_idx].detach().cpu().item()
                        )
                        item["infonce_aux_p_after"] = float(
                            infonce_stats["sample_p_after"][row_idx].detach().cpu().item()
                        )
                        item["infonce_aux_p_before"] = float(
                            infonce_stats["sample_p_before"][row_idx].detach().cpu().item()
                        )
                        item["infonce_aux_p_delta"] = float(
                            infonce_stats["sample_p_delta"][row_idx].detach().cpu().item()
                        )
                    self._write_reward_debug_log(sample_idx, item)
                self._stage_debug_log(
                    "compute_loss:merge_reward_debug_start",
                    f"items={len(reward_debug)}",
                )
                self._write_merged_group_reward_debug(reward_debug)
                self._stage_debug_log("compute_loss:merge_reward_debug_done")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = action_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        if self.top_entropy_quantile < 1.0:
            mean_entropy = masked_batch_mean(entropies)
            self._metrics[mode]["entropy"].append(
                self.accelerator.gather(mean_entropy).nanmean().item()
            )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )
        if self._use_sqr_latent_loss:
            zero = torch.zeros([], device=loss.device, dtype=torch.float32)
            loss_t = sqr_loss_metric_t
            loss_weighted_t = sqr_loss_weighted_metric_t
            applied_t = sqr_applied_metric_t
            ratio_mean_t = zero
            clip_rate_t = zero
            dist2_theta_t = zero
            dist2_old_t = zero
            cosine_mean_t = zero
            mask_sum_t = zero
            if sqr_stats is not None:
                ratio_mean_t = sqr_stats["ratio_mean"]
                clip_rate_t = sqr_stats["clip_rate"]
                dist2_theta_t = sqr_stats["dist2_theta_mean"]
                dist2_old_t = sqr_stats["dist2_old_mean"]
                cosine_mean_t = sqr_stats["cosine_mean"]
                mask_sum_t = sqr_stats["mask_sum"]
            self._metrics[mode]["sqr/loss"].append(
                self.accelerator.gather(loss_t).nanmean().item()
            )
            self._metrics[mode]["sqr/loss_weighted"].append(
                self.accelerator.gather(loss_weighted_t).nanmean().item()
            )
            self._metrics[mode]["sqr/applied_rate"].append(
                self.accelerator.gather(applied_t).nanmean().item()
            )
            self._metrics[mode]["sqr/ratio_mean"].append(
                self.accelerator.gather(ratio_mean_t).nanmean().item()
            )
            self._metrics[mode]["sqr/clip_rate"].append(
                self.accelerator.gather(clip_rate_t).nanmean().item()
            )
            self._metrics[mode]["sqr/dist2_theta_mean"].append(
                self.accelerator.gather(dist2_theta_t).nanmean().item()
            )
            self._metrics[mode]["sqr/dist2_old_mean"].append(
                self.accelerator.gather(dist2_old_t).nanmean().item()
            )
            self._metrics[mode]["sqr/cosine_mean"].append(
                self.accelerator.gather(cosine_mean_t).nanmean().item()
            )
            self._metrics[mode]["sqr/mask_sum"].append(
                self.accelerator.gather(mask_sum_t).sum().item()
            )
        if self._use_infonce_latent_aux_loss:
            zero = torch.zeros([], device=loss.device, dtype=torch.float32)
            loss_t = infonce_loss_metric_t
            loss_weighted_t = infonce_loss_weighted_metric_t
            applied_t = infonce_applied_metric_t
            p_after_t = zero
            p_before_t = zero
            p_delta_t = zero
            valid_count_t = zero
            neg_count_t = zero
            if infonce_stats is not None:
                p_after_t = infonce_stats["p_after_mean"]
                p_before_t = infonce_stats["p_before_mean"]
                p_delta_t = infonce_stats["p_delta_mean"]
                valid_count_t = infonce_stats["valid_count"]
                neg_count_t = infonce_stats["neg_count_mean"]
            self._metrics[mode]["infonce/loss"].append(
                self.accelerator.gather(loss_t).nanmean().item()
            )
            self._metrics[mode]["infonce/loss_weighted"].append(
                self.accelerator.gather(loss_weighted_t).nanmean().item()
            )
            self._metrics[mode]["infonce/applied_rate"].append(
                self.accelerator.gather(applied_t).nanmean().item()
            )
            self._metrics[mode]["infonce/p_after_mean"].append(
                self.accelerator.gather(p_after_t).nanmean().item()
            )
            self._metrics[mode]["infonce/p_before_mean"].append(
                self.accelerator.gather(p_before_t).nanmean().item()
            )
            self._metrics[mode]["infonce/p_delta_mean"].append(
                self.accelerator.gather(p_delta_t).nanmean().item()
            )
            self._metrics[mode]["infonce/valid_count"].append(
                self.accelerator.gather(valid_count_t).sum().item()
            )
            self._metrics[mode]["infonce/neg_count_mean"].append(
                self.accelerator.gather(neg_count_t).nanmean().item()
            )
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def _get_param_probe_sample(self) -> Optional[torch.Tensor]:
        if self._param_probe_param is None:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self._param_probe_param = param
                    self._param_probe_name = name
                    break
        if self._param_probe_param is None:
            return None
        with torch.no_grad():
            flat = self._param_probe_param.detach().float().view(-1)
            if flat.numel() == 0:
                return None
            stride = max(1, flat.numel() // self._param_probe_size)
            sample = flat[::stride][: self._param_probe_size].cpu()
        return sample

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._stage_debug_active():
            self._stage_debug_log("training_step:start")
        # Run the standard training step (includes backward).
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self._stage_debug_active():
            self._stage_debug_log("training_step:super_done")

        # Capture a cheap gradient probe on a single trainable param (main process only).
        if self.accelerator.is_main_process and self.model.training:
            if self._param_probe_param is None:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self._param_probe_param = param
                        self._param_probe_name = name
                        break
            grad = None
            if self._param_probe_param is not None:
                grad = self._param_probe_param.grad
            if grad is not None:
                g = grad.detach().float()
                self._last_probe_grad_abs_mean = float(g.abs().mean().item())
                self._last_probe_grad_norm = float(g.norm().item())
                nz = torch.count_nonzero(g).item()
                self._last_probe_grad_nonzero_frac = float(nz / g.numel()) if g.numel() else 0.0
            else:
                self._last_probe_grad_abs_mean = None
                self._last_probe_grad_norm = None
                self._last_probe_grad_nonzero_frac = None

        if self._stage_debug_active():
            self._stage_debug_log("training_step:done")
        return loss

    @staticmethod
    def _is_reward_metric_key(key: str) -> bool:
        if key in {"reward", "frac_reward_zero_std"}:
            return True
        if key.startswith("rewards/"):
            return True
        if key.startswith("debug/reward") or key.startswith("debug/group_reward"):
            return True
        if key.startswith("debug/adv") or key.startswith("debug/answer"):
            return True
        return False

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        reward_metrics = {
            key: val for key, val in metrics.items() if self._is_reward_metric_key(key)
        }
        if reward_metrics:
            self._last_reward_metrics[mode] = reward_metrics
            self._last_reward_metrics_step[mode] = self.state.global_step
            metrics["debug/reward_cached"] = 0.0
        else:
            cached = self._last_reward_metrics.get(mode) or {}
            if cached:
                metrics.update(cached)
                metrics["debug/reward_cached"] = 1.0
                cached_step = self._last_reward_metrics_step.get(mode)
                if cached_step is not None:
                    metrics["debug/reward_cached_from_step"] = float(cached_step)

        # Param probe to verify updates (main process only)
        if self.accelerator.is_main_process and mode == "train":
            sample = self._get_param_probe_sample()
            if sample is not None:
                logs["debug/param_probe_abs_mean"] = float(sample.abs().mean().item())
                logs["debug/param_probe_std"] = float(sample.std().item())
                if self._param_probe_prev is not None:
                    delta = (sample - self._param_probe_prev).abs().mean().item()
                    logs["debug/param_probe_delta"] = float(delta)
                self._param_probe_prev = sample
            if self._last_probe_grad_abs_mean is not None:
                logs["debug/probe_grad_abs_mean"] = self._last_probe_grad_abs_mean
            if self._last_probe_grad_norm is not None:
                logs["debug/probe_grad_norm"] = self._last_probe_grad_norm
            if self._last_probe_grad_nonzero_frac is not None:
                logs["debug/probe_grad_nonzero_frac"] = self._last_probe_grad_nonzero_frac

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
