import copy
import json
import logging
import os
import random
import time
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from trl import SFTTrainer

logger = logging.getLogger(__name__)


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


class EvalDetailLoggerCallback(TrainerCallback):
    """Writes per-eval summary metrics as JSONL for quick progress tracking."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.output_path = os.path.join(output_dir, "eval_detail_history.jsonl")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        os.makedirs(self.output_dir, exist_ok=True)
        row = {
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "step": int(getattr(state, "global_step", 0)),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
        }
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    row[k] = float(v)
                else:
                    row[k] = v
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if not logs or not any(str(k).startswith("eval_recall_") for k in logs.keys()):
            return
        os.makedirs(self.output_dir, exist_ok=True)
        row = {
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "step": int(getattr(state, "global_step", 0)),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "source": "on_log",
        }
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
            else:
                row[k] = v
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SoftRefineTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        refine_token_ids: Optional[Sequence[int]] = None,
        refine_token_id: Optional[int] = None,
        query_embeddings: np.ndarray,
        qid_to_query_row: Dict[str, int],
        qid_to_pos_video: Dict[str, str],
        qid_to_query_text: Optional[Dict[str, str]] = None,
        video_embeddings: np.ndarray,
        video_id_to_row: Dict[str, int],
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        hard_negative_refresh_steps: int = 0,
        enable_retrieval_optimization: bool = True,
        retrieval_loss_weight: float = 0.5,
        retrieval_temperature: float = 0.05,
        negative_pool_size: int = 32,
        retrieval_ignore_ambiguous_negatives: bool = False,
        ambiguous_negative_margin: float = 0.02,
        strict_negative_topk: int = 0,
        projector_lr: float = 1e-4,
        use_refine_gate: bool = True,
        retrieval_on_eval: bool = False,
        retrieval_debug: bool = False,
        retrieval_debug_max_logs: int = 500,
        retrieval_debug_jsonl: str = "",
        use_query_embedder_path: bool = False,
        query_embedder_model: Optional[nn.Module] = None,
        query_embedder_tokenizer=None,
        qfinal_pooling: str = "latent_last",
        qfinal_normalize: bool = True,
        tune_query_embedder: bool = False,
        query_embedder_lr: float = 5e-6,
        query_embedder_max_length: int = 128,
        refine_rollout_depth: int = 1,
        random_seed: int = 42,
        eval_recall_on_eval: bool = False,
        eval_recall_ks: str = "1,5,10",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if refine_token_ids is None:
            if refine_token_id is None:
                raise ValueError("Either refine_token_ids or refine_token_id must be provided.")
            refine_token_ids = [int(refine_token_id)]
        self.refine_token_ids = sorted({int(x) for x in refine_token_ids})
        if not self.refine_token_ids:
            raise ValueError("refine_token_ids must contain at least one valid token id.")
        # Keep legacy attribute for backward compatibility with old logs/scripts.
        self.refine_token_id = int(self.refine_token_ids[0])
        self.query_embeddings = query_embeddings
        self.qid_to_query_row = qid_to_query_row
        self.qid_to_pos_video = qid_to_pos_video
        self.qid_to_query_text = qid_to_query_text or {}
        self.video_embeddings = video_embeddings
        self.video_id_to_row = video_id_to_row
        self.video_ids: List[str] = list(video_id_to_row.keys())
        max_video_row = max((int(r) for r in video_id_to_row.values()), default=-1)
        self.video_id_by_row: List[str] = [""] * (max_video_row + 1)
        for vid, row in video_id_to_row.items():
            row_idx = int(row)
            if 0 <= row_idx < len(self.video_id_by_row) and not self.video_id_by_row[row_idx]:
                self.video_id_by_row[row_idx] = str(vid)
        self.hard_negatives = hard_negatives or {}
        self.hard_negative_refresh_steps = int(max(0, hard_negative_refresh_steps))
        self._last_hard_negative_refresh_step = -1
        self.enable_retrieval_optimization = bool(enable_retrieval_optimization)
        self.retrieval_loss_weight = float(retrieval_loss_weight)
        self.retrieval_temperature = float(max(retrieval_temperature, 1e-6))
        self.negative_pool_size = int(max(1, negative_pool_size))
        self.retrieval_ignore_ambiguous_negatives = bool(
            retrieval_ignore_ambiguous_negatives
        )
        self.ambiguous_negative_margin = float(max(0.0, ambiguous_negative_margin))
        self.strict_negative_topk = int(max(0, strict_negative_topk))
        self.projector_lr = float(projector_lr)
        self.use_refine_gate = bool(use_refine_gate)
        self.retrieval_on_eval = bool(retrieval_on_eval)
        self.retrieval_debug = bool(retrieval_debug)
        self.retrieval_debug_max_logs = int(max(0, retrieval_debug_max_logs))
        self.retrieval_debug_jsonl = str(retrieval_debug_jsonl or "").strip()
        self.use_query_embedder_path = bool(use_query_embedder_path)
        self.query_embedder_model = query_embedder_model
        self.query_embedder_tokenizer = query_embedder_tokenizer
        self.qfinal_pooling = str(qfinal_pooling or "latent_last").strip().lower()
        if self.qfinal_pooling not in {"latent_last", "mean"}:
            logger.warning(f"Unknown qfinal_pooling={qfinal_pooling}; fallback to latent_last.")
            self.qfinal_pooling = "latent_last"
        self.qfinal_normalize = bool(qfinal_normalize)
        self.tune_query_embedder = bool(tune_query_embedder)
        self.query_embedder_lr = float(query_embedder_lr)
        self.query_embedder_max_length = int(max(8, query_embedder_max_length))
        self.refine_rollout_depth = int(max(1, refine_rollout_depth))
        self.query_embedder_input_prefix = str(
            os.environ.get("QUERY_EMBEDDER_INPUT_PREFIX", "")
        ).strip()
        # 임시 운영 옵션:
        # - SOFT_REFINE_SKIP_OOM_BATCH=1: CUDA OOM 배치를 건너뛰고 학습 지속
        # - SOFT_REFINE_SKIP_QIDS="qid1,qid2": 지정 qid 배치를 강제 스킵
        self.skip_oom_batch = str(
            os.environ.get("SOFT_REFINE_SKIP_OOM_BATCH", "1")
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self.skip_qids = {
            tok.strip()
            for tok in str(os.environ.get("SOFT_REFINE_SKIP_QIDS", "")).split(",")
            if tok.strip()
        }
        self._rng = random.Random(int(random_seed))
        self._eval_loss_ret_values: List[float] = []
        self._eval_loss_sft_values: List[float] = []
        self._debug_logged = 0
        self._oom_skip_count = 0
        self.eval_recall_on_eval = bool(eval_recall_on_eval)
        self.eval_recall_ks = self._parse_eval_recall_ks(eval_recall_ks)
        self._video_embeddings_norm: Optional[np.ndarray] = None
        if self.use_query_embedder_path:
            if self.query_embedder_model is None or self.query_embedder_tokenizer is None:
                logger.warning(
                    "use_query_embedder_path=True but query embedder model/tokenizer missing; "
                    "fallback to residual q_final."
                )
                self.use_query_embedder_path = False
            else:
                for p in self.query_embedder_model.parameters():
                    p.requires_grad = bool(self.tune_query_embedder)
                if self.tune_query_embedder:
                    self.query_embedder_model.train()
                else:
                    self.query_embedder_model.eval()
                logger.info(
                    "Query embedder path enabled: "
                    f"pooling={self.qfinal_pooling}, normalize={self.qfinal_normalize}, "
                    f"tune_query_embedder={self.tune_query_embedder}"
                )
                if self.query_embedder_input_prefix:
                    logger.info(
                        "Query embedder input prefix enabled: "
                        f"{self.query_embedder_input_prefix}"
                    )
        if not self.use_refine_gate:
            base_model = self._unwrap_model(self.model)
            gate = getattr(base_model, "refine_gate", None)
            if gate is not None:
                for p in gate.parameters():
                    p.requires_grad = False
            logger.info(
                "Refine gate disabled: using q_final = normalize(q_orig + delta)."
            )
        if self.hard_negative_refresh_steps > 0:
            logger.info(
                "Online hard-negative refresh enabled: "
                f"every {self.hard_negative_refresh_steps} optimizer step(s), "
                f"topk={self.negative_pool_size}"
            )
        if self.retrieval_ignore_ambiguous_negatives:
            logger.info(
                "Ambiguous-negative filtering enabled: "
                f"q_orig margin={self.ambiguous_negative_margin:.4f}, "
                f"strict_topk={self.strict_negative_topk or 'all'}"
            )

    def _parse_eval_recall_ks(self, value: str | Sequence[int]) -> List[int]:
        if isinstance(value, str):
            tokens = [tok.strip() for tok in value.split(",")]
        else:
            tokens = [str(x).strip() for x in value]
        out: List[int] = []
        for tok in tokens:
            if not tok:
                continue
            try:
                k = int(tok)
            except ValueError:
                continue
            if k > 0:
                out.append(k)
        if not out:
            out = [1, 5, 10]
        return sorted(set(out))

    @staticmethod
    def _l2_norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom = np.maximum(denom, eps)
        return x / denom

    @staticmethod
    def _compute_ranks(
        q_mat: np.ndarray,
        video_mat: np.ndarray,
        pos_rows: np.ndarray,
    ) -> np.ndarray:
        sims = np.matmul(q_mat, video_mat.T)
        pos_scores = sims[np.arange(sims.shape[0]), pos_rows]
        return (sims > pos_scores[:, None]).sum(axis=1) + 1

    def _get_video_embeddings_norm(self) -> np.ndarray:
        if self._video_embeddings_norm is None:
            self._video_embeddings_norm = self._l2_norm_rows(
                np.asarray(self.video_embeddings, dtype=np.float32)
            )
        return self._video_embeddings_norm

    def _get_query_text(self, qid: str, meta: Optional[dict]) -> str:
        text = str(self.qid_to_query_text.get(str(qid), "")).strip()
        if text:
            return text
        if isinstance(meta, dict):
            for key in ("query", "query_text", "user_query", "query_raw"):
                v = meta.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""

    def _make_refine_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        if len(self.refine_token_ids) == 1:
            return input_ids.eq(int(self.refine_token_ids[0]))
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for tok_id in self.refine_token_ids:
            mask |= input_ids.eq(int(tok_id))
        return mask


    def _resolve_active_rollout_depth(self, for_training_step: bool) -> int:
        # Use full rollout depth from the very first step (no curriculum).
        return int(max(1, self.refine_rollout_depth))

    # def _resolve_active_rollout_depth(self, for_training_step: bool) -> int:
    #     max_depth = int(max(1, self.refine_rollout_depth))
    #     if max_depth <= 1:
    #         return 1
    #     if not for_training_step:
    #         return max_depth

    #     stage_count = max_depth
    #     max_steps = int(getattr(self.args, "max_steps", 0) or 0)
    #     if max_steps > 0:
    #         step = int(max(0, getattr(self.state, "global_step", 0)))
    #         progress = min(max(float(step) / float(max_steps), 0.0), 0.999999)
    #         return int(min(max_depth, int(progress * stage_count) + 1))

    #     num_epochs = float(getattr(self.args, "num_train_epochs", 0.0) or 0.0)
    #     if num_epochs > 0.0:
    #         epoch = float(getattr(self.state, "epoch", 0.0) or 0.0)
    #         progress = min(max(epoch / num_epochs, 0.0), 0.999999)
    #         return int(min(max_depth, int(progress * stage_count) + 1))
    #     return 1

    @staticmethod
    def _inject_latent_tokens(
        query_token_embeds: torch.Tensor,
        latent_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        insert_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_tokens.dim() != 3:
            raise ValueError(f"latent_tokens must be 3D [B,L,D], got shape={tuple(latent_tokens.shape)}")
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
        self,
        model_inputs: Dict[str, torch.Tensor],
        row_index: int,
    ) -> Dict[str, torch.Tensor]:
        """
        rollout용 single-row 입력을 구성한다.
        배치 차원(B)과 같은 첫 축을 가진 텐서는 row 단위로 슬라이스하고,
        그 외 텐서는 그대로 유지한다.
        """
        out: Dict[str, torch.Tensor] = {}
        input_ids = model_inputs.get("input_ids", None)
        if input_ids is None or not torch.is_tensor(input_ids):
            return out
        bsz = int(input_ids.size(0))
        row = int(row_index)
        if row < 0 or row >= bsz:
            return out

        for key, value in model_inputs.items():
            if not torch.is_tensor(value):
                continue
            if key in {"labels", "output_hidden_states"}:
                continue
            if value.dim() > 0 and int(value.size(0)) == bsz:
                out[key] = value[row : row + 1]
            else:
                # 예: batch와 독립적인 메타 텐서(grid_thw 등)
                out[key] = value

        if "input_ids" in out and "attention_mask" not in out:
            out["attention_mask"] = torch.ones_like(out["input_ids"], dtype=torch.long)

        # Multimodal inputs are flattened across the whole batch:
        # - pixel_values / pixel_values_videos: [sum_i (t_i*h_i*w_i), C]
        # - image_grid_thw / video_grid_thw: [num_media_items, 3]
        # For rollout on a single row, keep only the media chunk that belongs to that row
        # (common case in this training setup: one video item per sample).
        def _slice_media_for_row(
            flat_key: str,
            grid_key: str,
            token_id: Optional[int],
        ) -> None:
            flat = model_inputs.get(flat_key, None)
            grid = model_inputs.get(grid_key, None)
            if not (torch.is_tensor(flat) and torch.is_tensor(grid)):
                return
            if flat.dim() == 0 or grid.dim() < 2:
                return

            # If media item count aligns with batch size, slice by row directly.
            if int(grid.size(0)) == bsz:
                lengths = grid.to(dtype=torch.long).prod(dim=1)
                total_len = int(lengths.sum().item())
                if total_len != int(flat.size(0)):
                    # Keep original tensors if shape metadata is inconsistent.
                    return
                start = int(lengths[:row].sum().item())
                end = start + int(lengths[row].item())
                out[grid_key] = grid[row : row + 1]
                out[flat_key] = flat[start:end]
                return

            # General path: map media items to each row using modality token counts.
            if token_id is None:
                return
            per_row_items = input_ids.eq(int(token_id)).sum(dim=1).to(dtype=torch.long)
            total_items = int(per_row_items.sum().item())
            if total_items != int(grid.size(0)):
                return

            item_start = int(per_row_items[:row].sum().item())
            item_count = int(per_row_items[row].item())
            if item_count <= 0:
                out[grid_key] = grid[:0]
                out[flat_key] = flat[:0]
                return
            item_end = item_start + item_count

            lengths = grid.to(dtype=torch.long).prod(dim=1)
            total_len = int(lengths.sum().item())
            if total_len != int(flat.size(0)):
                return
            flat_start = int(lengths[:item_start].sum().item())
            flat_end = flat_start + int(lengths[item_start:item_end].sum().item())
            out[grid_key] = grid[item_start:item_end]
            out[flat_key] = flat[flat_start:flat_end]

        base_model = self._unwrap_model(self.model)
        model_cfg = getattr(base_model, "config", None)
        image_token_id = getattr(model_cfg, "image_token_id", None)
        video_token_id = getattr(model_cfg, "video_token_id", None)
        _slice_media_for_row("pixel_values", "image_grid_thw", image_token_id)
        _slice_media_for_row("pixel_values_videos", "video_grid_thw", video_token_id)
        return out

    def _rollout_refine_latents(
        self,
        model: nn.Module,
        first_update: torch.Tensor,
        query_text: Optional[str],
        rollout_depth: int,
        row_model_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if first_update.dim() == 1:
            first_update = first_update.unsqueeze(0)
        if first_update.dim() != 2:
            raise ValueError(
                f"first_update must be 1D/2D tensor, got shape={tuple(first_update.shape)}"
            )
        depth = int(max(1, rollout_depth))

        z_seed = first_update.mean(dim=0, keepdim=True).to(dtype=torch.float32)
        # 커리큘럼: depth=1이면 seed 1개만 사용
        if depth <= 1:
            return z_seed

        if row_model_inputs is None or "input_ids" not in row_model_inputs:
            # rollout 입력이 없으면 형태만 depth에 맞춘다.
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

        # VLM 기본 입력(query+video+기존 텍스트) 임베딩
        query_token_embeds = base_model.get_input_embeddings()(input_ids)
        # rollout(VLM): <REFINE> 토큰 바로 뒤에 latent를 삽입한다.
        refine_mask = self._make_refine_mask(input_ids)
        positions = torch.nonzero(refine_mask[0], as_tuple=False).squeeze(-1)


        if positions.numel() > 0:
            vlm_insert_idx = int(positions[-1].item()) + 1
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

        z_list: List[torch.Tensor] = [z_seed.squeeze(0)]
        # depth=2 -> [u1,u2], depth=3 -> [u1,u2,u3] 형태로 누적 생성.
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
                    [head_pos, latent_pos, tail_pos + latent_len],
                    dim=-1,
                )

            # [기존 로직 - 주석 보존] position_ids를 전달하지 않고 attention_mask만 전달했다.
            # vlm_outputs = vlm_core(
            #     input_ids=None,
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=full_attention,
            #     use_cache=False,
            #     **extra_forward_inputs,
            # )
            # [변경 로직] 재구성한 position_ids를 명시적으로 전달해
            # latent 삽입 후에도 멀티모달 위치 정렬(mRoPE)이 유지되게 한다.
            vlm_outputs = vlm_core(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention,
                position_ids=full_position_ids,
                use_cache=False,
                **extra_forward_inputs,
            )
            hidden = vlm_outputs.last_hidden_state.to(dtype=torch.float32)
            # if self.qfinal_pooling == "mean":
            #     mask = full_attention.to(dtype=hidden.dtype).unsqueeze(-1)
            #     denom = torch.clamp(mask.sum(dim=1), min=1.0)
            #     pooled = (hidden * mask).sum(dim=1) / denom
            # else:
            #     pooled = hidden[:, -1, :]
            if self.qfinal_pooling == "mean":
                mask = full_attention.to(dtype=hidden.dtype).unsqueeze(-1)
                denom = torch.clamp(mask.sum(dim=1), min=1.0)
                pooled = (hidden * mask).sum(dim=1) / denom
            else:
                # non-mean에서는 "방금 삽입한 마지막 latent 위치"를 정확히 읽는다.
                extracted_idx = vlm_insert_idx + int(latent_tokens.size(1)) - 1
                pooled = hidden[:, extracted_idx, :]


            # VLM hidden(H) -> latent(D) readout
            try:
                projector_dtype = next(refine_projector.parameters()).dtype
            except StopIteration:
                projector_dtype = pooled.dtype
            delta = refine_projector(pooled.to(dtype=projector_dtype)).to(dtype=torch.float32)
            if self.use_refine_gate:
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

    def _build_q_final(
        self,
        model: nn.Module,
        q_orig: torch.Tensor,
        update: torch.Tensor,
        query_text: Optional[str],
    ) -> tuple[torch.Tensor, str]:
        if update.dim() == 1:
            update = update.unsqueeze(0)
        if update.dim() != 2:
            raise ValueError(f"update must be 1D/2D tensor, got shape={tuple(update.shape)}")

        if self.use_query_embedder_path:
            if not query_text:
                raise ValueError("query_text is empty for query-embedder q_final path.")
            q_embedder = self.query_embedder_model
            if self.tune_query_embedder:
                q_embedder.train()
            else:
                q_embedder.eval()

            embedder_query_text = _build_query_embedder_text(
                self.query_embedder_tokenizer,
                query_text,
                self.query_embedder_input_prefix,
            )
            tokenized = self.query_embedder_tokenizer(
                [embedder_query_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.query_embedder_max_length,
            )
            device = update.device
            input_ids = tokenized["input_ids"].to(device=device)
            attention_mask = tokenized["attention_mask"].to(device=device)

            query_token_embeds = q_embedder.get_input_embeddings()(input_ids)
            # q_final(embedder): 기존 chat-template tail(기본 6) 앞에 삽입.
            tail_len = int(min(6, max(int(query_token_embeds.size(1)), 0)))
            embedder_insert_idx = int(query_token_embeds.size(1) - tail_len)
            base_model = self._unwrap_model(model)
            head = getattr(base_model, "query_embedder_head", None)
            # q_final 생성 시 query에 append되는 latent 전용 projector.
            append_in_projector = getattr(base_model, "refine_append_input_projector", None)
            if append_in_projector is None:
                append_in_projector = getattr(base_model, "refine_latent_input_projector", None)

            # q_final 단계: rollout과 분리된 append projector를 거친 latent를 query에 붙임
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

            embed_outputs = q_embedder.language_model(
                input_ids=None,
                attention_mask=full_attention,
                inputs_embeds=inputs_embeds,
                use_cache=False,
            )
            hidden = embed_outputs.last_hidden_state.to(dtype=torch.float32)

            if self.qfinal_pooling == "mean":
                mask = full_attention.to(dtype=hidden.dtype).unsqueeze(-1)
                denom = torch.clamp(mask.sum(dim=1), min=1.0)
                pooled = (hidden * mask).sum(dim=1) / denom
            else:
                pooled = hidden[:, -1, :]

            if head is not None:
                pooled = head(pooled)
            q_final = pooled.squeeze(0)
            mode = "query_embedder"
        else:
            # Multi-refine fallback for residual mode: average updates to avoid scale blow-up.
            q_final = q_orig + update.mean(dim=0)
            mode = "residual_add"

        if self.qfinal_normalize:
            q_final = F.normalize(q_final, p=2, dim=-1)
        return q_final, mode

    def _compute_eval_recall_metrics(self, eval_dataset=None) -> Dict[str, float]:
        if not self.enable_retrieval_optimization:
            return {}
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            return {}
        start_time = time.perf_counter()
        total_eval_rows = int(len(eval_dataset))
        logger.info(
            "Starting eval recall computation: "
            f"rows={total_eval_rows}, ks={self.eval_recall_ks}, "
            f"qfinal_mode={'query_embedder' if self.use_query_embedder_path else 'residual_add'}, "
            f"pooling={self.qfinal_pooling}, normalize={self.qfinal_normalize}"
        )

        accel = getattr(self, "accelerator", None)
        model = self.model
        device = getattr(self.args, "device", None)
        if accel is not None:
            device = accel.device

        projector, gate = self._get_refine_modules(model)
        try:
            projector_dtype = next(projector.parameters()).dtype
        except StopIteration:
            projector_dtype = torch.float32
        gate_dtype = projector_dtype
        try:
            gate_dtype = next(gate.parameters()).dtype
        except StopIteration:
            pass
        was_training = bool(model.training)
        model.eval()

        collator = self.data_collator

        def _safe_collate(examples):
            return collator([copy.deepcopy(ex) for ex in examples])

        loader = DataLoader(
            eval_dataset,
            batch_size=max(1, int(self.args.per_device_eval_batch_size)),
            shuffle=False,
            num_workers=0,
            collate_fn=_safe_collate,
            pin_memory=False,
        )

        orig_vecs: List[np.ndarray] = []
        refine_vecs: List[np.ndarray] = []
        pos_rows: List[int] = []
        skip_qid_missing = 0
        skip_pos_missing = 0
        skip_refine_missing = 0
        skip_query_text_missing = 0
        qfinal_mode_eval = "query_embedder" if self.use_query_embedder_path else "residual_add"
        processed_rows = 0
        progress_interval = max(1, total_eval_rows // 10) if total_eval_rows > 0 else 1
        active_rollout_depth = self._resolve_active_rollout_depth(for_training_step=was_training)

        with torch.no_grad():
            for batch in loader:
                qids = batch.pop("qids", [])
                metas = batch.pop("metas", [])

                model_inputs = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        model_inputs[k] = v.to(device, non_blocking=False)
                    else:
                        model_inputs[k] = v
                model_inputs["output_hidden_states"] = True

                outputs = model(**model_inputs)
                input_ids = model_inputs["input_ids"]
                last_hidden = outputs.hidden_states[-1]
                refine_mask = self._make_refine_mask(input_ids)
                labels = model_inputs.get("labels", None)
                if labels is not None:
                    
                    # Use only trainable assistant tokens for refine position search.
                    refine_mask = refine_mask & labels.ne(-100)

                bs = input_ids.size(0)
                for row in range(bs):
                    processed_rows += 1
                    qid = str(qids[row]).strip() if row < len(qids) else ""
                    q_row = self.qid_to_query_row.get(qid)
                    if q_row is None:
                        skip_qid_missing += 1
                        continue

                    meta = metas[row] if row < len(metas) and isinstance(metas[row], dict) else {}
                    pos_vid = (
                        meta.get("query_pos_video_id")
                        or self.qid_to_pos_video.get(qid)
                        or meta.get("pos_video_id")
                    )
                    pos_row = self.video_id_to_row.get(str(pos_vid)) if pos_vid else None
                    if pos_row is None:
                        skip_pos_missing += 1
                        continue

                    positions = torch.nonzero(refine_mask[row], as_tuple=False).squeeze(1)
                    if positions.numel() == 0:
                        skip_refine_missing += 1
                        continue
                    positions = positions[-1:]

                    h_ref = last_hidden[row, positions]
                    h_ref_for_projector = h_ref.to(dtype=projector_dtype)
                    delta = projector(h_ref_for_projector).to(dtype=torch.float32)

                    if self.use_refine_gate:
                        h_ref_for_gate = h_ref.to(dtype=gate_dtype)
                        alpha = torch.sigmoid(gate(h_ref_for_gate)).to(dtype=torch.float32)
                        update = delta * alpha
                    else:
                        update = delta

                    query_text = None
                    if self.use_query_embedder_path:
                        query_text_val = self._get_query_text(qid=qid, meta=meta)
                        if not query_text_val:
                            skip_query_text_missing += 1
                            continue
                        query_text = query_text_val
                    row_rollout_inputs = self._build_row_rollout_inputs(model_inputs, row)
                    update = self._rollout_refine_latents(
                        model=model,
                        first_update=update,
                        query_text=query_text,
                        rollout_depth=active_rollout_depth,
                        row_model_inputs=row_rollout_inputs,
                    )

                    q_orig = np.asarray(self.query_embeddings[q_row], dtype=np.float32)
                    q_orig = q_orig / max(float(np.linalg.norm(q_orig)), 1e-12)
                    q_orig_t = torch.from_numpy(q_orig).to(device=device, dtype=torch.float32)
                    q_final_t, qfinal_mode_eval = self._build_q_final(
                        model=model,
                        q_orig=q_orig_t,
                        update=update,
                        query_text=query_text,
                    )

                    orig_vecs.append(q_orig.astype(np.float32))
                    refine_vecs.append(q_final_t.detach().cpu().numpy().astype(np.float32))
                    pos_rows.append(int(pos_row))

                if processed_rows % progress_interval == 0 or processed_rows >= total_eval_rows:
                    logger.info(
                        "Eval recall progress: "
                        f"processed={processed_rows}/{total_eval_rows}, valid={len(orig_vecs)}, "
                        f"skip_qid={skip_qid_missing}, skip_pos={skip_pos_missing}, "
                        f"skip_refine={skip_refine_missing}, skip_query_text={skip_query_text_missing}"
                    )

        if was_training:
            model.train()

        elapsed_sec = float(time.perf_counter() - start_time)
        metrics: Dict[str, float] = {
            "eval_recall_eval_rows": float(len(eval_dataset)),
            "eval_recall_valid_rows": float(len(orig_vecs)),
            "eval_recall_skip_qid_missing": float(skip_qid_missing),
            "eval_recall_skip_pos_missing": float(skip_pos_missing),
            "eval_recall_skip_refine_token_missing": float(skip_refine_missing),
            "eval_recall_skip_query_text_missing": float(skip_query_text_missing),
            "eval_recall_qfinal_mode": qfinal_mode_eval,
            "eval_recall_pooling_mode": self.qfinal_pooling,
            "eval_recall_rollout_depth": float(active_rollout_depth),
            "eval_recall_runtime_sec": elapsed_sec,
        }
        if not orig_vecs:
            logger.info(
                "Finished eval recall computation with no valid rows: "
                f"runtime_sec={elapsed_sec:.2f}"
            )
            return metrics

        q_orig_mat = self._l2_norm_rows(np.stack(orig_vecs, axis=0).astype(np.float32))
        q_ref_mat = self._l2_norm_rows(np.stack(refine_vecs, axis=0).astype(np.float32))
        pos_rows_np = np.asarray(pos_rows, dtype=np.int64)
        video_mat = self._get_video_embeddings_norm()

        ranks_orig = self._compute_ranks(q_orig_mat, video_mat, pos_rows_np)
        ranks_ref = self._compute_ranks(q_ref_mat, video_mat, pos_rows_np)

        for k in self.eval_recall_ks:
            kk = int(min(max(1, k), video_mat.shape[0]))
            orig_r = float(np.mean(ranks_orig <= kk))
            ref_r = float(np.mean(ranks_ref <= kk))
            metrics[f"eval_recall_original_r{kk}"] = orig_r
            metrics[f"eval_recall_refined_r{kk}"] = ref_r
            metrics[f"eval_recall_delta_r{kk}"] = ref_r - orig_r

        mrr_orig = float(np.mean(1.0 / ranks_orig))
        mrr_ref = float(np.mean(1.0 / ranks_ref))
        metrics["eval_recall_original_mrr"] = mrr_orig
        metrics["eval_recall_refined_mrr"] = mrr_ref
        metrics["eval_recall_delta_mrr"] = mrr_ref - mrr_orig
        metrics["eval_recall_original_mean_rank"] = float(np.mean(ranks_orig))
        metrics["eval_recall_refined_mean_rank"] = float(np.mean(ranks_ref))
        logger.info(
            "Finished eval recall computation: "
            f"runtime_sec={elapsed_sec:.2f}, valid_rows={len(orig_vecs)}"
        )
        return metrics

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP/DeepSpeed style wrappers to access custom attributes."""
        unwrapped = model
        visited = set()
        while hasattr(unwrapped, "module") and id(unwrapped) not in visited:
            visited.add(id(unwrapped))
            unwrapped = unwrapped.module
        return unwrapped

    def _get_refine_modules(self, model: nn.Module):
        base_model = self._unwrap_model(model)
        projector = getattr(base_model, "refine_projector", None)
        gate = getattr(base_model, "refine_gate", None)
        if projector is None or gate is None:
            raise AttributeError(
                f"Refine modules are missing on model type {type(base_model)}. "
                "Expected attributes: refine_projector and refine_gate."
            )
        return projector, gate

    def _dummy_refine_loss(self, model: nn.Module, last_hidden: torch.Tensor) -> torch.Tensor:
        """
        Force refine modules to participate in each rank/step without affecting
        the objective when there is no valid <REFINE> sample.
        """
        if not self.enable_retrieval_optimization:
            return torch.zeros([], device=last_hidden.device, dtype=torch.float32)

        refine_projector, refine_gate = self._get_refine_modules(model)
        try:
            projector_dtype = next(refine_projector.parameters()).dtype
        except StopIteration:
            projector_dtype = last_hidden.dtype
        h_dummy = last_hidden[:, 0, :].detach().to(dtype=projector_dtype)
        delta_dummy = refine_projector(h_dummy)
        dummy = delta_dummy.sum()
        if self.use_refine_gate:
            try:
                gate_dtype = next(refine_gate.parameters()).dtype
            except StopIteration:
                gate_dtype = h_dummy.dtype
            alpha_dummy = torch.sigmoid(refine_gate(h_dummy.to(dtype=gate_dtype)))
            dummy = dummy + alpha_dummy.sum()
        base_model = self._unwrap_model(model)
        latent_input_projector = getattr(base_model, "refine_latent_input_projector", None)
        if latent_input_projector is not None:
            for p in latent_input_projector.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0
        append_input_projector = getattr(base_model, "refine_append_input_projector", None)
        if append_input_projector is not None:
            for p in append_input_projector.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0
        head = getattr(base_model, "query_embedder_head", None)
        if head is not None:
            for p in head.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0
        if self.use_query_embedder_path and self.tune_query_embedder and self.query_embedder_model is not None:
            for p in self.query_embedder_model.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0
        return dummy * 0.0

    def _get_video_vec(self, video_id: Optional[str]) -> Optional[np.ndarray]:
        if not video_id:
            return None
        row = self.video_id_to_row.get(str(video_id))
        if row is None:
            return None
        return np.asarray(self.video_embeddings[row], dtype=np.float32)

    def _sample_random_negatives(self, banned: set[str], need: int) -> List[str]:
        if need <= 0 or not self.video_ids:
            return []
        picked: List[str] = []
        attempts = 0
        max_attempts = max(need * 100, 1000)
        while len(picked) < need and attempts < max_attempts:
            vid = self.video_ids[self._rng.randrange(len(self.video_ids))]
            attempts += 1
            if vid in banned:
                continue
            banned.add(vid)
            picked.append(vid)
        if len(picked) < need:
            for vid in self.video_ids:
                if len(picked) >= need:
                    break
                if vid in banned:
                    continue
                banned.add(vid)
                picked.append(vid)
        return picked

    def _select_negative_ids(
        self, qid: str, pos_vid: str, retrieved_vid: Optional[str]
    ) -> List[str]:
        out: List[str] = []
        banned = {pos_vid}

        if (
            retrieved_vid
            and retrieved_vid != pos_vid
            and str(retrieved_vid) in self.video_id_to_row
        ):
            out.append(retrieved_vid)
            banned.add(retrieved_vid)

        for vid in self.hard_negatives.get(qid, []):
            if vid not in self.video_id_to_row:
                continue
            if vid in banned:
                continue
            out.append(vid)
            banned.add(vid)
            if len(out) >= self.negative_pool_size:
                break

        if len(out) < self.negative_pool_size:
            out.extend(
                self._sample_random_negatives(
                    banned=banned, need=self.negative_pool_size - len(out)
                )
            )
        return out[: self.negative_pool_size]

    def _select_strict_negative_indices(
        self,
        sim_pos_before: torch.Tensor,
        sim_neg_before: torch.Tensor,
        sim_neg_after: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, int | bool]]:
        total_count = int(sim_neg_before.numel())
        if total_count <= 0:
            empty = torch.zeros((0,), device=sim_neg_before.device, dtype=torch.long)
            return empty, {
                "total": 0,
                "ambiguous": 0,
                "strict": 0,
                "selected": 0,
                "topk_applied": False,
            }

        ambiguous_mask = torch.zeros_like(sim_neg_before, dtype=torch.bool)
        if self.retrieval_ignore_ambiguous_negatives:
            ambiguous_mask = sim_neg_before >= (
                sim_pos_before - self.ambiguous_negative_margin
            )
        strict_indices = torch.nonzero(~ambiguous_mask, as_tuple=False).squeeze(1)
        topk_applied = False
        if self.strict_negative_topk > 0 and strict_indices.numel() > self.strict_negative_topk:
            top_local = torch.topk(
                sim_neg_after[strict_indices], k=self.strict_negative_topk
            ).indices
            strict_indices = strict_indices[top_local]
            topk_applied = True

        return strict_indices, {
            "total": total_count,
            "ambiguous": int(ambiguous_mask.sum().item()),
            "strict": int((~ambiguous_mask).sum().item()),
            "selected": int(strict_indices.numel()),
            "topk_applied": topk_applied,
        }

    def _should_refresh_hard_negatives_online(self, is_training: bool) -> bool:
        if not is_training:
            return False
        if self.hard_negative_refresh_steps <= 0:
            return False
        step = int(getattr(self.state, "global_step", 0))
        if step <= 0:
            return False
        if step == self._last_hard_negative_refresh_step:
            return False
        return (step % self.hard_negative_refresh_steps) == 0

    def _mine_online_hard_negatives(
        self,
        query_vec: np.ndarray,
        banned_video_ids: set[str],
        topk: int,
    ) -> List[str]:
        if topk <= 0:
            return []
        video_mat = self._get_video_embeddings_norm()
        if video_mat.ndim != 2 or video_mat.shape[0] == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        sims = np.matmul(video_mat, q)
        sims = np.asarray(sims, dtype=np.float32).reshape(-1)

        for vid in banned_video_ids:
            row = self.video_id_to_row.get(str(vid))
            if row is None:
                continue
            row_idx = int(row)
            if 0 <= row_idx < sims.shape[0]:
                sims[row_idx] = -np.inf

        valid_count = int(np.isfinite(sims).sum())
        if valid_count <= 0:
            return []
        use_k = int(min(max(1, topk), valid_count))

        if use_k >= sims.shape[0]:
            top_idx = np.argsort(-sims)[:use_k]
        else:
            part = np.argpartition(-sims, kth=use_k - 1)[:use_k]
            top_idx = part[np.argsort(-sims[part])]

        out: List[str] = []
        for idx in top_idx:
            row_idx = int(idx)
            if not (0 <= row_idx < len(self.video_id_by_row)):
                continue
            vid = self.video_id_by_row[row_idx]
            if not vid or vid in banned_video_ids:
                continue
            out.append(vid)
            if len(out) >= use_k:
                break
        return out

    def _maybe_refresh_hard_negatives_online(
        self,
        q_final: torch.Tensor,
        valid_qids: Sequence[str],
        valid_pos_vids: Sequence[str],
        stats: Dict[str, float | int | str],
        is_training: bool,
    ) -> None:
        if not self._should_refresh_hard_negatives_online(is_training=is_training):
            return

        if q_final.dim() != 2 or q_final.size(0) <= 0:
            return

        t0 = time.perf_counter()
        q_np = q_final.detach().to(dtype=torch.float32).cpu().numpy()
        refresh_topk = int(max(1, self.negative_pool_size))
        rows = int(min(q_np.shape[0], len(valid_qids), len(valid_pos_vids)))

        updated = 0
        skipped = 0
        seen_qids: set[str] = set()
        for i in range(rows):
            qid = str(valid_qids[i]).strip()
            pos_vid = str(valid_pos_vids[i]).strip()
            if not qid or not pos_vid:
                skipped += 1
                continue
            if qid in seen_qids:
                continue
            seen_qids.add(qid)

            mined = self._mine_online_hard_negatives(
                query_vec=q_np[i],
                banned_video_ids={pos_vid},
                topk=refresh_topk,
            )
            if not mined:
                skipped += 1
                continue
            self.hard_negatives[qid] = mined
            updated += 1

        step = int(getattr(self.state, "global_step", 0))
        self._last_hard_negative_refresh_step = step
        elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
        stats["hard_neg_refresh_step"] = int(step)
        stats["hard_neg_refresh_rows"] = int(updated)
        stats["hard_neg_refresh_skip"] = int(skipped)
        stats["hard_neg_refresh_topk"] = int(refresh_topk)
        stats["hard_neg_refresh_ms"] = elapsed_ms

    def _compute_retrieval_loss(
        self,
        model: nn.Module,
        outputs,
        model_inputs: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        qids: Optional[Sequence[str]],
        metas: Optional[Sequence[dict]],
    ) -> tuple[torch.Tensor, Dict[str, float | int | str]]:
        stats: Dict[str, float | int | str] = {
            "reason": "ok",
            "gate_mode": "enabled" if self.use_refine_gate else "disabled",
            "qfinal_mode": "query_embedder" if self.use_query_embedder_path else "residual_add",
            "pooling_mode": self.qfinal_pooling,
            "embedder_used": bool(self.use_query_embedder_path),
            "batch_size": int(input_ids.size(0)),
            "rollout_depth_target": int(max(1, self.refine_rollout_depth)),
            "rollout_depth_active": 1,
            "candidate_rows": 0,
            "valid_rows": 0,
            "skip_qid_missing": 0,
            "skip_pos_missing": 0,
            "skip_neg_missing": 0,
            "skip_refine_pos_missing": 0,
            "skip_query_text_missing": 0,
        }
        if (
            not self.enable_retrieval_optimization
            or outputs.hidden_states is None
            or qids is None
            or metas is None
        ):
            stats["reason"] = "disabled_or_missing_inputs"
            if outputs.hidden_states is not None:
                return self._dummy_refine_loss(model=model, last_hidden=outputs.hidden_states[-1]), stats
            return torch.zeros([], device=input_ids.device, dtype=torch.float32), stats

        #1 refine 위치
        refine_mask = self._make_refine_mask(input_ids)
        # if labels is not None:
        #     #breakpoint()
        #     # Exclude system/user prompt tokens from refine candidates.
        #     refine_mask = refine_mask & labels.ne(-100)

        candidate_rows = torch.nonzero(refine_mask.any(dim=1), as_tuple=False).squeeze(1)
        stats["candidate_rows"] = int(candidate_rows.numel())
        if candidate_rows.numel() == 0:
            stats["reason"] = "no_refine_rows"
            return self._dummy_refine_loss(model=model, last_hidden=outputs.hidden_states[-1]), stats

        last_hidden = outputs.hidden_states[-1]
        update_list: List[torch.Tensor] = []
        query_texts: List[str] = []
        q_orig_list: List[np.ndarray] = []
        pos_list: List[np.ndarray] = []
        neg_list: List[np.ndarray] = []
        valid_qids: List[str] = []
        valid_pos_vids: List[str] = []
        delta_norm_mean_list: List[torch.Tensor] = []
        delta_norm_max_list: List[torch.Tensor] = []
        alpha_mean_list: List[torch.Tensor] = []
        alpha_min_list: List[torch.Tensor] = []
        alpha_max_list: List[torch.Tensor] = []
        latent_count_list: List[int] = []
        refine_projector, refine_gate = self._get_refine_modules(model)
        try:
            projector_dtype = next(refine_projector.parameters()).dtype
        except StopIteration:
            projector_dtype = last_hidden.dtype
        gate_dtype = projector_dtype
        try:
            gate_dtype = next(refine_gate.parameters()).dtype
        except StopIteration:
            pass
        active_rollout_depth = self._resolve_active_rollout_depth(for_training_step=bool(model.training))
        stats["rollout_depth_active"] = int(active_rollout_depth)

        for row_tensor in candidate_rows:
            row = int(row_tensor.item())
            if row >= len(qids) or row >= len(metas):
                continue

            qid = str(qids[row])
            meta = metas[row] if isinstance(metas[row], dict) else {}

            q_row = self.qid_to_query_row.get(qid)
            if q_row is None:
                stats["skip_qid_missing"] = int(stats["skip_qid_missing"]) + 1
                continue

            pos_vid = (
                meta.get("query_pos_video_id")
                or self.qid_to_pos_video.get(qid)
                or meta.get("pos_video_id")
            )
            pos_vec = self._get_video_vec(pos_vid)
            if pos_vid is None or pos_vec is None:
                stats["skip_pos_missing"] = int(stats["skip_pos_missing"]) + 1
                continue

            retrieved_vid = meta.get("video_id")
            neg_ids = self._select_negative_ids(
                qid=qid, pos_vid=str(pos_vid), retrieved_vid=retrieved_vid
            )
            neg_vecs = [self._get_video_vec(v) for v in neg_ids]
            neg_vecs = [v for v in neg_vecs if v is not None]
            if not neg_vecs:
                stats["skip_neg_missing"] = int(stats["skip_neg_missing"]) + 1
                continue

            positions = torch.nonzero(refine_mask[row], as_tuple=False).squeeze(1)

            # breakpoint()
            if positions.numel() == 0:
                stats["skip_refine_pos_missing"] = int(stats["skip_refine_pos_missing"]) + 1
                continue
            positions = positions[-1:]

            # 2) <REFINE> hidden에서 첫 latent(seed) 생성.
            h_refine = last_hidden[row, positions]
            h_ref_for_projector = h_refine.to(dtype=projector_dtype)
            delta = refine_projector(h_ref_for_projector).to(dtype=torch.float32)
            if self.use_refine_gate:
                h_ref_for_gate = h_refine.to(dtype=gate_dtype)
                alpha = torch.sigmoid(refine_gate(h_ref_for_gate)).to(dtype=torch.float32)
                update = delta * alpha
            else:
                alpha = torch.ones(
                    (delta.size(0), 1), device=delta.device, dtype=delta.dtype
                )
                update = delta

            query_text = None
            if self.use_query_embedder_path:
                query_text_val = self._get_query_text(qid=qid, meta=meta)
                if not query_text_val:
                    stats["skip_query_text_missing"] = int(stats["skip_query_text_missing"]) + 1
                    continue
                query_text = query_text_val
                query_texts.append(query_text)

            # 3) rollout: 이전 latent들을 누적 컨텍스트로 넣어 다음 latent들을 생성.
            row_rollout_inputs = self._build_row_rollout_inputs(model_inputs, row)
            update = self._rollout_refine_latents(
                model=model,
                first_update=update,
                query_text=query_text,
                rollout_depth=active_rollout_depth,
                row_model_inputs=row_rollout_inputs,
            )

            update_list.append(update)
            q_orig_list.append(np.asarray(self.query_embeddings[q_row], dtype=np.float32))
            pos_list.append(pos_vec)
            neg_list.append(np.stack(neg_vecs, axis=0).astype(np.float32))
            valid_qids.append(str(qid))
            valid_pos_vids.append(str(pos_vid))
            delta_norm = torch.norm(delta, dim=-1)
            delta_norm_mean_list.append(delta_norm.mean())
            delta_norm_max_list.append(delta_norm.max())
            alpha_scalar = alpha.squeeze(-1)
            alpha_mean_list.append(alpha_scalar.mean())
            alpha_min_list.append(alpha_scalar.min())
            alpha_max_list.append(alpha_scalar.max())
            latent_count_list.append(int(update.size(0)))

        stats["valid_rows"] = int(len(update_list))
        if not update_list:
            stats["reason"] = "no_valid_rows_after_filter"
            return self._dummy_refine_loss(model=model, last_hidden=outputs.hidden_states[-1]), stats

        device = input_ids.device
        q_orig = torch.from_numpy(np.stack(q_orig_list, axis=0)).to(device=device, dtype=torch.float32)
        pos_emb = torch.from_numpy(np.stack(pos_list, axis=0)).to(device=device, dtype=torch.float32)
        neg_emb = torch.from_numpy(np.stack(neg_list, axis=0)).to(device=device, dtype=torch.float32)

        q_orig = F.normalize(q_orig, p=2, dim=-1)
        pos_emb = F.normalize(pos_emb, p=2, dim=-1)
        neg_emb = F.normalize(neg_emb, p=2, dim=-1)
        q_final_rows: List[torch.Tensor] = []
        qfinal_mode = "query_embedder" if self.use_query_embedder_path else "residual_add"
        for row_idx in range(len(update_list)):
            q_text = query_texts[row_idx] if self.use_query_embedder_path else None
            q_final_row, qfinal_mode = self._build_q_final(
                model=model,
                q_orig=q_orig[row_idx],
                update=update_list[row_idx],
                query_text=q_text,
            )
            q_final_rows.append(q_final_row.to(device=device, dtype=torch.float32))
        q_final = torch.stack(q_final_rows, dim=0)
        if q_final.size(-1) != pos_emb.size(-1):
            raise ValueError(
                f"q_final dim {q_final.size(-1)} != video embedding dim {pos_emb.size(-1)}"
            )
        stats["qfinal_mode"] = qfinal_mode

        self._maybe_refresh_hard_negatives_online(
            q_final=q_final,
            valid_qids=valid_qids,
            valid_pos_vids=valid_pos_vids,
            stats=stats,
            is_training=bool(model.training),
        )

        q_shift = torch.norm(q_final - q_orig, dim=-1)
        pos_sim_before = torch.sum(q_orig * pos_emb, dim=-1)
        pos_sim_after = torch.sum(q_final * pos_emb, dim=-1)

        logits_pos = torch.sum(q_final * pos_emb, dim=-1)
        logits_neg = torch.einsum("bd,bkd->bk", q_final, neg_emb)
        neg_before = torch.einsum("bd,bkd->bk", q_orig, neg_emb)
        row_target = torch.zeros((1,), dtype=torch.long, device=device)
        per_row_losses: List[torch.Tensor] = []
        strict_counts: List[int] = []
        ambiguous_counts: List[int] = []
        selected_counts: List[int] = []
        strict_top_before: List[torch.Tensor] = []
        strict_top_after: List[torch.Tensor] = []
        strict_rows = 0
        all_ambiguous_rows = 0
        topk_applied_rows = 0

        for row_idx in range(logits_neg.size(0)):
            selected_idx, neg_stats = self._select_strict_negative_indices(
                sim_pos_before=pos_sim_before[row_idx],
                sim_neg_before=neg_before[row_idx],
                sim_neg_after=logits_neg[row_idx],
            )
            strict_counts.append(int(neg_stats["strict"]))
            ambiguous_counts.append(int(neg_stats["ambiguous"]))
            selected_counts.append(int(neg_stats["selected"]))
            if bool(neg_stats["topk_applied"]):
                topk_applied_rows += 1
            if selected_idx.numel() == 0:
                all_ambiguous_rows += 1
                continue

            strict_rows += 1
            row_logits = torch.cat(
                [logits_pos[row_idx].unsqueeze(0), logits_neg[row_idx, selected_idx]], dim=0
            ).unsqueeze(0) / self.retrieval_temperature
            per_row_losses.append(F.cross_entropy(row_logits, row_target))
            strict_top_before.append(neg_before[row_idx, selected_idx].max())
            strict_top_after.append(logits_neg[row_idx, selected_idx].max())

        if not per_row_losses:
            stats["reason"] = "no_strict_negatives_after_filter"
            stats["ambiguous_neg_count_mean"] = (
                float(np.mean(ambiguous_counts)) if ambiguous_counts else 0.0
            )
            stats["strict_neg_count_mean"] = (
                float(np.mean(strict_counts)) if strict_counts else 0.0
            )
            stats["selected_neg_count_mean"] = (
                float(np.mean(selected_counts)) if selected_counts else 0.0
            )
            stats["strict_neg_rows"] = int(strict_rows)
            stats["all_ambiguous_rows"] = int(all_ambiguous_rows)
            stats["strict_neg_topk_applied_rows"] = int(topk_applied_rows)
            return self._dummy_refine_loss(model=model, last_hidden=outputs.hidden_states[-1]), stats

        loss_ret = torch.stack(per_row_losses, dim=0).mean()

        with torch.no_grad():
            neg_after = logits_neg
            stats.update(
                {
                    "reason": "ok",
                    "delta_norm_mean": float(torch.stack(delta_norm_mean_list).mean().detach().cpu().item()),
                    "delta_norm_max": float(torch.stack(delta_norm_max_list).max().detach().cpu().item()),
                    "alpha_mean": float(torch.stack(alpha_mean_list).mean().detach().cpu().item()),
                    "alpha_min": float(torch.stack(alpha_min_list).min().detach().cpu().item()),
                    "alpha_max": float(torch.stack(alpha_max_list).max().detach().cpu().item()),
                    "latent_token_count_mean": float(np.mean(latent_count_list)),
                    "latent_token_count_max": float(max(latent_count_list)),
                    "latent_token_count_min": float(min(latent_count_list)),
                    "q_shift_mean": float(q_shift.mean().detach().cpu().item()),
                    "pos_sim_before_mean": float(pos_sim_before.mean().detach().cpu().item()),
                    "pos_sim_after_mean": float(pos_sim_after.mean().detach().cpu().item()),
                    "neg_top_before_mean": float(neg_before.max(dim=1).values.mean().detach().cpu().item()),
                    "neg_top_after_mean": float(neg_after.max(dim=1).values.mean().detach().cpu().item()),
                    "strict_neg_top_before_mean": float(
                        torch.stack(strict_top_before).mean().detach().cpu().item()
                    ),
                    "strict_neg_top_after_mean": float(
                        torch.stack(strict_top_after).mean().detach().cpu().item()
                    ),
                    "ambiguous_neg_count_mean": float(np.mean(ambiguous_counts)),
                    "strict_neg_count_mean": float(np.mean(strict_counts)),
                    "selected_neg_count_mean": float(np.mean(selected_counts)),
                    "strict_neg_rows": int(strict_rows),
                    "all_ambiguous_rows": int(all_ambiguous_rows),
                    "strict_neg_topk_applied_rows": int(topk_applied_rows),
                    "retrieval_ignore_ambiguous_negatives": bool(
                        self.retrieval_ignore_ambiguous_negatives
                    ),
                    "ambiguous_negative_margin": float(self.ambiguous_negative_margin),
                    "strict_negative_topk": int(self.strict_negative_topk),
                    "loss_ret_value": float(loss_ret.detach().cpu().item()),
                }
            )

        return loss_ret, stats

    def _log_retrieval_debug(
        self,
        model: nn.Module,
        loss_sft: torch.Tensor,
        loss_ret: torch.Tensor,
        stats: Dict[str, float | int | str],
    ) -> None:
        if not self.retrieval_debug:
            return
        if self.retrieval_debug_max_logs > 0 and self._debug_logged >= self.retrieval_debug_max_logs:
            return
        accel = getattr(self, "accelerator", None)
        if accel is not None and not accel.is_main_process:
            return

        payload = dict(stats)
        payload.update(
            {
                "global_step": int(getattr(self.state, "global_step", 0)),
                "epoch": float(getattr(self.state, "epoch", 0.0) or 0.0),
                "is_train_mode": bool(model.training),
                "loss_sft": float(loss_sft.detach().cpu().item()),
                "loss_ret": float(loss_ret.detach().cpu().item()),
                "loss_total": float((loss_sft + self.retrieval_loss_weight * loss_ret).detach().cpu().item()),
            }
        )
        logger.info("[RET_DEBUG] " + json.dumps(payload, ensure_ascii=False))

        if self.retrieval_debug_jsonl:
            path = self.retrieval_debug_jsonl
            if not os.path.isabs(path):
                path = os.path.join(self.args.output_dir, path)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self._debug_logged += 1

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, torch.OutOfMemoryError) or ("out of memory" in msg and "cuda" in msg)

    @staticmethod
    def _collect_batch_context(inputs) -> tuple[List[str], int, int]:
        qids_raw = inputs.get("qids", None) if isinstance(inputs, dict) else None
        qid_list: List[str] = []
        if isinstance(qids_raw, (list, tuple)):
            qid_list = [str(x) for x in qids_raw]
        elif qids_raw is not None:
            qid_list = [str(qids_raw)]

        seq_len = -1
        bsz = -1
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids", None)
            if torch.is_tensor(input_ids):
                bsz = int(input_ids.size(0))
                seq_len = int(input_ids.size(1))
        return qid_list, bsz, seq_len

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not (model.training and self.skip_oom_batch and self._is_cuda_oom(exc)):
                raise

            self._oom_skip_count += 1
            qid_list, bsz, seq_len = self._collect_batch_context(inputs)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # OOM 직전 partial grad가 남아 다음 step을 오염시키지 않게 정리
            if getattr(self, "optimizer", None) is not None:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()
            else:
                model.zero_grad(set_to_none=True)

            device = None
            if isinstance(inputs, dict):
                input_ids = inputs.get("input_ids", None)
                if torch.is_tensor(input_ids):
                    device = input_ids.device
            if device is None:
                device = next(model.parameters()).device

            stats = {
                "reason": "oom_skip_backward",
                "oom_skip_count": int(self._oom_skip_count),
                "oom_batch_size": int(bsz),
                "oom_seq_len": int(seq_len),
                "oom_qids": qid_list[:4],
            }
            zero = torch.zeros([], device=device, dtype=torch.float32)
            self._log_retrieval_debug(
                model=model,
                loss_sft=zero,
                loss_ret=zero,
                stats=stats,
            )
            logger.warning(
                "[OOM_SKIP_BACKWARD] skipped CUDA OOM batch at backward and continued training: "
                f"count={self._oom_skip_count}, batch_size={bsz}, seq_len={seq_len}, qids={qid_list[:4]}"
            )
            return zero

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = dict(inputs)
        qids = model_inputs.pop("qids", None)
        metas = model_inputs.pop("metas", None)

        input_ids = model_inputs.get("input_ids", None)
        device = input_ids.device if torch.is_tensor(input_ids) else next(model.parameters()).device
        qid_list: List[str] = []
        if isinstance(qids, (list, tuple)):
            qid_list = [str(x) for x in qids]
        elif qids is not None:
            qid_list = [str(qids)]
        if model.training and self.skip_qids and any(q in self.skip_qids for q in qid_list):
            loss_sft = torch.zeros([], device=device, dtype=torch.float32, requires_grad=True)
            loss_ret = torch.zeros([], device=device, dtype=torch.float32)
            ret_stats = {
                "reason": "qid_skip",
                "qid_skip_hit": [q for q in qid_list if q in self.skip_qids][:4],
            }
            self._log_retrieval_debug(
                model=model,
                loss_sft=loss_sft,
                loss_ret=loss_ret,
                stats=ret_stats,
            )
            logger.warning(
                "[QID_SKIP] skipped batch by SOFT_REFINE_SKIP_QIDS: "
                f"hit={ret_stats['qid_skip_hit']}"
            )
            return (loss_sft, None) if return_outputs else loss_sft

        use_retrieval_this_step = self.enable_retrieval_optimization and (
            model.training or self.retrieval_on_eval
        )
        model_inputs["output_hidden_states"] = bool(use_retrieval_this_step)
        try:
            outputs = model(**model_inputs)
            loss_sft = outputs.loss

            if use_retrieval_this_step:
                loss_ret, ret_stats = self._compute_retrieval_loss(
                    model=model,
                    outputs=outputs,
                    model_inputs=model_inputs,
                    input_ids=model_inputs["input_ids"],
                    labels=model_inputs.get("labels", None),
                    qids=qids,
                    metas=metas,
                )
            else:
                loss_ret = torch.zeros([], device=device, dtype=torch.float32)
                ret_stats = {"reason": "retrieval_disabled_for_phase"}
            loss = loss_sft + self.retrieval_loss_weight * loss_ret
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not (model.training and self.skip_oom_batch and self._is_cuda_oom(exc)):
                raise
            self._oom_skip_count += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loss_sft = torch.zeros([], device=device, dtype=torch.float32, requires_grad=True)
            loss_ret = torch.zeros([], device=device, dtype=torch.float32)
            seq_len = int(model_inputs["input_ids"].size(1)) if torch.is_tensor(model_inputs.get("input_ids")) else -1
            bsz = int(model_inputs["input_ids"].size(0)) if torch.is_tensor(model_inputs.get("input_ids")) else -1
            ret_stats = {
                "reason": "oom_skip",
                "oom_skip_count": int(self._oom_skip_count),
                "oom_batch_size": bsz,
                "oom_seq_len": seq_len,
                "oom_qids": qid_list[:4],
            }
            loss = loss_sft
            logger.warning(
                "[OOM_SKIP] skipped CUDA OOM batch and continued training: "
                f"count={self._oom_skip_count}, batch_size={bsz}, seq_len={seq_len}, qids={qid_list[:4]}"
            )
            outputs = None

        self._log_retrieval_debug(
            model=model,
            loss_sft=loss_sft,
            loss_ret=loss_ret,
            stats=ret_stats,
        )

        if not model.training:
            self._eval_loss_sft_values.append(float(loss_sft.detach().cpu().item()))
            self._eval_loss_ret_values.append(float(loss_ret.detach().cpu().item()))

        if self.state.global_step % max(int(self.args.logging_steps), 1) == 0:
            self.log(
                {
                    "loss_sft": float(loss_sft.detach().cpu().item()),
                    "loss_ret": float(loss_ret.detach().cpu().item()),
                }
            )

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        eval_dataset = kwargs.get("eval_dataset", None)
        self._eval_loss_ret_values = []
        self._eval_loss_sft_values = []
        metrics = super().evaluate(*args, **kwargs)
        if self._eval_loss_sft_values:
            metrics["eval_loss_sft_mean"] = float(np.mean(self._eval_loss_sft_values))
        if self._eval_loss_ret_values:
            metrics["eval_loss_ret_mean"] = float(np.mean(self._eval_loss_ret_values))
            active = [v for v in self._eval_loss_ret_values if abs(v) > 0.0]
            metrics["eval_loss_ret_active_ratio"] = float(len(active) / len(self._eval_loss_ret_values))
            if active:
                metrics["eval_loss_ret_active_mean"] = float(np.mean(active))
        if self.eval_recall_on_eval and self.enable_retrieval_optimization:
            try:
                recall_metrics = self._compute_eval_recall_metrics(eval_dataset=eval_dataset)
                if recall_metrics:
                    metrics.update(recall_metrics)
                    self.log(recall_metrics)
                    summary = {}
                    for key in (
                        "eval_loss",
                        "eval_recall_refined_r1",
                        "eval_recall_refined_r5",
                        "eval_recall_refined_r10",
                        "eval_recall_refined_r100",
                        "eval_recall_delta_r1",
                        "eval_recall_delta_r5",
                        "eval_recall_delta_r10",
                        "eval_recall_delta_r100",
                    ):
                        if key in metrics:
                            summary[key] = float(metrics[key])
                    if summary:
                        logger.info("[EVAL_RECALL_SUMMARY] " + json.dumps(summary, ensure_ascii=False))
            except Exception as exc:
                logger.warning(f"Failed to compute eval recall metrics: {exc}")
        return metrics

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            base_lr = optimizer_kwargs.pop("lr", self.args.learning_rate)

            base_params: List[torch.nn.Parameter] = []
            refine_params: List[torch.nn.Parameter] = []
            query_embedder_params: List[torch.nn.Parameter] = []
            model_param_ids: set[int] = set()
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                model_param_ids.add(id(param))
                if (
                    "refine_projector" in name
                    or "refine_gate" in name
                    or "refine_latent_input_projector" in name
                    or "refine_append_input_projector" in name
                    or "query_embedder_head" in name
                ):
                    refine_params.append(param)
                elif "query_embedder_model" in name:
                    query_embedder_params.append(param)
                else:
                    base_params.append(param)

            external_query_params: List[torch.nn.Parameter] = []
            if self.use_query_embedder_path and self.tune_query_embedder and self.query_embedder_model is not None:
                for param in self.query_embedder_model.parameters():
                    if param.requires_grad and id(param) not in model_param_ids:
                        external_query_params.append(param)
            if external_query_params:
                if bool(getattr(self.args, "deepspeed", None)):
                    logger.warning(
                        "Detected query_embedder params outside self.model under DeepSpeed; "
                        "freezing them to avoid ZeRO param-name mapping errors."
                    )
                    for param in external_query_params:
                        param.requires_grad = False
                else:
                    query_embedder_params.extend(external_query_params)

            param_groups = []
            if base_params:
                param_groups.append({"params": base_params, "lr": base_lr})
            if refine_params:
                param_groups.append({"params": refine_params, "lr": self.projector_lr})
            if query_embedder_params:
                param_groups.append({"params": query_embedder_params, "lr": self.query_embedder_lr})

            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer
