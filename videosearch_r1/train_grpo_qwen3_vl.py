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
import json
import shutil
import datetime
import logging
import torch
from torch import distributed as dist
from typing import Optional, Tuple
from transformers import AutoConfig, AutoProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from model.modeling_qwen3_vl_patched import Qwen3VLForConditionalGeneration
from model.monkey_patch import apply_qwen3_vl_monkey_patch
from reward.reward import reward_funcs_registry
from trainer.grpo_vllm_trainer_qwen3_vl import Qwen3_VL_GRPOVLLMTrainer
from utils.arguments import process_args
from utils.data_rl import LazyGRPODataset

logger = logging.getLogger("train_grpo")


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

    # in deepspeed, we need to count p.ds_numel instead of p.numel
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


def _resolve_text_hidden_size_from_model_path(
    model_path: Optional[str], default: int = 2048
) -> Tuple[int, str]:
    path = str(model_path or "").strip()
    if not path:
        return int(default), f"default({int(default)})"
    try:
        cfg = AutoConfig.from_pretrained(path)
    except Exception:
        return int(default), f"default({int(default)})"
    hidden, source = _resolve_text_hidden_size_from_config(cfg, default=default)
    return hidden, f"{path}:{source}"


def _init_refine_modules(
    model: torch.nn.Module,
    *,
    use_query_embedder_path: bool = False,
    query_embedder_model_path: Optional[str] = None,
) -> Tuple[Tuple[int, str], Tuple[int, str]]:
    hidden_size, hidden_source = _resolve_text_hidden_size(model, default=2048)
    model.refine_projector = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 2048),
        torch.nn.LayerNorm(2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 2048),
    )
    model.refine_gate = torch.nn.Linear(hidden_size, 1)
    model.refine_latent_input_projector = torch.nn.Sequential(
        torch.nn.LayerNorm(2048),
        torch.nn.Linear(2048, 2048),
    )
    model.refine_append_input_projector = torch.nn.Sequential(
        torch.nn.LayerNorm(2048),
        torch.nn.Linear(2048, 2048),
    )
    with torch.no_grad():
        torch.nn.init.eye_(model.refine_latent_input_projector[1].weight)
        torch.nn.init.zeros_(model.refine_latent_input_projector[1].bias)
        torch.nn.init.eye_(model.refine_append_input_projector[1].weight)
        torch.nn.init.zeros_(model.refine_append_input_projector[1].bias)
    query_hidden = 2048
    query_hidden_source = "disabled"
    if bool(use_query_embedder_path):
        query_hidden, query_hidden_source = _resolve_text_hidden_size_from_model_path(
            query_embedder_model_path, default=2048
        )
        if query_hidden != 2048:
            model.query_embedder_head = torch.nn.Linear(query_hidden, 2048)
        else:
            model.query_embedder_head = torch.nn.Identity()
    return (hidden_size, hidden_source), (query_hidden, query_hidden_source)


def _freeze_refine_modules(model: torch.nn.Module, use_refine_gate: bool = False):
    base = model.module if hasattr(model, "module") else model
    projector = getattr(base, "refine_projector", None)
    gate = getattr(base, "refine_gate", None)
    rollout_proj = getattr(base, "refine_latent_input_projector", None)
    append_proj = getattr(base, "refine_append_input_projector", None)
    q_head = getattr(base, "query_embedder_head", None)
    if projector is not None:
        for p in projector.parameters():
            p.requires_grad = False
    if gate is not None:
        for p in gate.parameters():
            p.requires_grad = False
    if rollout_proj is not None:
        for p in rollout_proj.parameters():
            p.requires_grad = False
    if append_proj is not None:
        for p in append_proj.parameters():
            p.requires_grad = False
    if q_head is not None:
        for p in q_head.parameters():
            p.requires_grad = False


def _build_refine_tokens(refine_token: str, refine_token_count: int) -> list[str]:
    token = str(refine_token or "<REFINE>").strip()
    if not token:
        token = "<REFINE>"
    # GRPO refine path now uses a single token; rollout depth controls latent steps.
    return [token]


def _maybe_load_refine_weights(
    model: torch.nn.Module, model_path: str, logger_: logging.Logger
) -> bool:
    """
    Load refine_projector/refine_gate/query_embedder_head weights from checkpoint directory when present.
    """
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
                logger_.info(
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
            logger_.warning(
                "Skipped incompatible refine tensors while loading checkpoint "
                f"(dropped={len(dropped)}; examples={preview})"
            )
        if not loadable:
            logger_.warning("No compatible refine tensors found in checkpoint.")
            return False
        missing, unexpected = model.load_state_dict(loadable, strict=False)
        logger_.info(
            "Loaded refine weights from checkpoint "
            f"({len(loadable)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return True
    logger_.warning("No refine weights found in checkpoint; latent improve reward may be unstable.")
    return False


def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))

    # parsing arguments
    model_args, data_args, training_args = process_args(is_grpo=True)
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # set up training args
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # set up loggers
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

    # monkey patch the Qwen model before initialization for handling the mixed modality dataset
    if model_args.apply_monkey_patch:
        apply_qwen3_vl_monkey_patch(model_args.apply_monkey_patch)
        logger.info(f"Qwen3-VL patch applied, mode: {model_args.apply_monkey_patch}")

    # build processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_path, padding_side="left"
    )
    refine_token = str(getattr(training_args, "refine_token", "<REFINE>"))
    refine_tokens = _build_refine_tokens(refine_token, 1)
    refine_rollout_depth = int(max(1, getattr(training_args, "refine_rollout_depth", 1)))
    needs_refine_vocab_resize = bool(getattr(training_args, "use_latent_improve_reward", False))
    missing_refine_tokens = [
        tok for tok in refine_tokens if tok not in processor.tokenizer.get_vocab()
    ]
    if missing_refine_tokens:
        if needs_refine_vocab_resize:
            processor.tokenizer.add_tokens(missing_refine_tokens, special_tokens=True)
            logger.info(f"Added refine tokens to tokenizer: {missing_refine_tokens}")
        else:
            logger.info(
                f"Refine tokens missing in tokenizer: {missing_refine_tokens}. "
                "Skipping tokenizer resize because latent improve reward is disabled."
            )
    logger.info(
        "Refine token config: "
        f"base={refine_token}, tokens={refine_tokens}, rollout_depth={refine_rollout_depth}"
    )

    # build model
    attn_impl = os.environ.get("ATTN_IMPL", "flash_attention_2")
    if not attn_impl:
        attn_impl = "flash_attention_2"
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if training_args.bf16 else None,
        "attn_implementation": attn_impl,
        # "use_cache": False if training_args.gradient_checkpointing else True,
    }

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_path, **model_kwargs
    )
    logger.info(f"Model loaded from {model_args.model_path}")

    # build reference model
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_path, **model_kwargs
    )
    logger.info(f"Reference model loaded from {model_args.model_path}")
    if needs_refine_vocab_resize:
        model.resize_token_embeddings(len(processor.tokenizer))
        ref_model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.use_latent_improve_reward:
        (model_hidden, model_hidden_src), (
            query_hidden,
            query_hidden_src,
        ) = _init_refine_modules(
            model,
            use_query_embedder_path=training_args.use_query_embedder_path,
            query_embedder_model_path=training_args.query_embedder_model_path,
        )
        (ref_hidden, ref_hidden_src), _ = _init_refine_modules(
            ref_model,
            use_query_embedder_path=training_args.use_query_embedder_path,
            query_embedder_model_path=training_args.query_embedder_model_path,
        )
        _maybe_load_refine_weights(model, model_args.model_path, logger)
        _maybe_load_refine_weights(ref_model, model_args.model_path, logger)
        _freeze_refine_modules(model, use_refine_gate=training_args.use_refine_gate)
        _freeze_refine_modules(ref_model, use_refine_gate=training_args.use_refine_gate)
        logger.info(
            "Latent improve mode: refine modules loaded and frozen "
            f"(use_refine_gate={training_args.use_refine_gate}, "
            f"model_hidden={model_hidden}@{model_hidden_src}, "
            f"ref_hidden={ref_hidden}@{ref_hidden_src}, "
            f"query_hidden={query_hidden}@{query_hidden_src}, "
            f"rollout_depth={refine_rollout_depth}, "
            f"use_sqr_latent_loss={bool(getattr(training_args, 'use_sqr_latent_loss', False))})"
        )

    # set model
    set_model(model_args, model, logger)

    # build dataset
    train_dataset = LazyGRPODataset(
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config,
        video_min_pixels=data_args.video_min_pixels,
        video_max_pixels=data_args.video_max_pixels,
        video_total_pixels=data_args.video_total_pixels,
        max_frames=data_args.max_frames,
        nframes=data_args.nframes,
        fps=data_args.fps,
        image_min_pixels=data_args.image_min_pixels,
        image_max_pixels=data_args.image_max_pixels,
        rl_mode=data_args.rl_mode,  # direct_rl, cot_rl, answer_twice_rl
    )

    # reward functions
    reward_funcs = [reward_funcs_registry[func] for func in training_args.reward_funcs]

    # initialize GRPO trainer
    trainer = Qwen3_VL_GRPOVLLMTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=processor,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
    )

    # train model
    trainer.train()
    logger.info("Training finished")

    # save model
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if (
        training_args.use_query_embedder_path
        and training_args.query_embedder_model_path
        and os.path.isdir(training_args.query_embedder_model_path)
    ):
        q_out_dir = os.path.join(training_args.output_dir, "query_embedder")
        if not os.path.exists(q_out_dir):
            shutil.copytree(training_args.query_embedder_model_path, q_out_dir)
        logger.info(f"Copied query embedder snapshot to {q_out_dir}")

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
