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

import torch
from torch import distributed as dist
from transformers import AutoProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers import HfArgumentParser
from trl import SFTTrainer, SFTConfig

from model.modeling_qwen2_5_vl_patched import Qwen2_5_VLForConditionalGeneration
from model.monkey_patch import apply_qwen2_5_vl_monkey_patch
from trainer.sft_share_gpt_trainer import ShareGPTSFTCollator
from utils.arguments import ModelArguments
from utils.data_sft_share_gpt import ShareGPTDataArguments, build_sharegpt_dataset

logger = logging.getLogger("train_sft_sharegpt")


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
    else:
        for p in model.visual.merger.parameters():
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

    parser = HfArgumentParser((ModelArguments, ShareGPTDataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    if model_args.apply_monkey_patch:
        apply_qwen2_5_vl_monkey_patch(model_args.apply_monkey_patch)
        logger.info(f"Qwen2.5-VL patch applied, mode: {model_args.apply_monkey_patch}")

    processor = AutoProcessor.from_pretrained(model_args.model_path)

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if training_args.bf16 else None,
        "attn_implementation": "flash_attention_2",
        "use_cache": False if training_args.gradient_checkpointing else True,
    }
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_path, **model_kwargs
    )
    logger.info(f"Model loaded from {model_args.model_path}")

    set_model(model_args, model, logger)

    train_dataset = build_sharegpt_dataset(data_args)

    collator = ShareGPTSFTCollator(
        processor,
        loss_type="only_assistant",
        max_length=model_args.model_max_length,
        image_patch_size=14,
        use_video_metadata=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    logger.info("Training finished")

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
