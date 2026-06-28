import logging
from typing import Any, Dict, List

import torch

from model.qwen_vl_utils.vision_process import cached_process_vision_info, process_vision_info

logger = logging.getLogger(__name__)


class ShareGPTSFTCollator:
    def __init__(
        self,
        processor,
        loss_type: str = "only_assistant",
        max_length: int = 16384,
        image_patch_size: int = 14,
        use_video_metadata: bool = False,
    ):
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.loss_type = loss_type
        self.max_length = max_length
        self.image_patch_size = image_patch_size
        self.use_video_metadata = use_video_metadata
        logger.info(f"Model max length: {self.max_length}")

        assert loss_type in [
            "exclude_visual",
            "only_assistant",
            "all_assistant",
            "format_answer",
        ], f"Loss type {loss_type} is not supported."
        logger.info(f"SFT loss type: {loss_type}")

        if loss_type in ["only_assistant", "all_assistant", "format_answer"]:
            self.assistant_prefix_ids = self.tokenizer(
                "<|im_start|>assistant\n", add_special_tokens=False
            ).input_ids
            logger.info(
                rf"id of <|im_start|>assistant\n: {self.assistant_prefix_ids}"
            )
        if loss_type == "all_assistant":
            self.user_prefix_ids = self.tokenizer(
                "<|im_start|>user\n", add_special_tokens=False
            ).input_ids
            self.system_prefix_ids = self.tokenizer(
                "<|im_start|>system\n", add_special_tokens=False
            ).input_ids

        if loss_type == "format_answer":
            self.think_open_ids = self.tokenizer(
                "<think>", add_special_tokens=False
            ).input_ids
            self.think_close_ids = self.tokenizer(
                "</think>", add_special_tokens=False
            ).input_ids
            logger.info(
                f"id of <think>: {self.think_open_ids}, </think>: {self.think_close_ids}"
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        qids = [example.get("qid", "") for example in examples]
        metas = [example.get("meta", {}) for example in examples]

        for idx in range(len(examples)):
            examples[idx]["messages"].append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": examples[idx]["response"]}
                    ],
                }
            )

        messages = [example["messages"] for example in examples]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False)
            for msg in messages
        ]

        if self.use_video_metadata:
            image_inputs, packed_video_inputs, video_kwargs = cached_process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
                image_patch_size=self.image_patch_size,
            )
            if packed_video_inputs is not None:
                video_inputs, video_metadatas = zip(*packed_video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)
            else:
                video_inputs = None
                video_metadatas = None
            batch = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
                **video_kwargs,
            )
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=self.image_patch_size,
            )
            batch = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
                **video_kwargs,
            )

        if self.loss_type == "exclude_visual":
            labels = batch["input_ids"].clone()

        elif self.loss_type == "only_assistant":
            labels = torch.full_like(batch["input_ids"], -100)
            pattern = torch.tensor(self.assistant_prefix_ids, device=labels.device)
            wins = batch["input_ids"].unfold(
                dimension=-1, size=pattern.numel(), step=1
            )
            match_mask = (wins == pattern).all(dim=-1)

            for b_idx in range(match_mask.size(0)):
                pos = torch.nonzero(match_mask[b_idx], as_tuple=False).squeeze(1)
                if pos.numel() == 0:
                    logger.warning(
                        f"[ShareGPTSFTCollator] No assistant prefix after truncation for sample {b_idx}. "
                        f"Message (truncated): {messages[b_idx]}"
                    )
                    continue
                start_idx = pos[-1].item() + pattern.numel()
                labels[b_idx, start_idx:] = batch["input_ids"][b_idx, start_idx:]

        elif self.loss_type == "all_assistant":
            labels = torch.full_like(batch["input_ids"], -100)
            seq_len = batch["input_ids"].size(1)

            def _match_positions(pattern_ids: List[int]):
                if seq_len < len(pattern_ids):
                    return [torch.tensor([], device=labels.device, dtype=torch.long) for _ in range(labels.size(0))]
                pattern = torch.tensor(pattern_ids, device=labels.device)
                wins = batch["input_ids"].unfold(
                    dimension=-1, size=pattern.numel(), step=1
                )
                match_mask = (wins == pattern).all(dim=-1)
                return [
                    torch.nonzero(match_mask[b], as_tuple=False).squeeze(1)
                    for b in range(match_mask.size(0))
                ]

            asst_pos = _match_positions(self.assistant_prefix_ids)
            user_pos = _match_positions(self.user_prefix_ids)
            sys_pos = _match_positions(self.system_prefix_ids)

            for b_idx in range(labels.size(0)):
                asst_list = asst_pos[b_idx].tolist()
                if not asst_list:
                    logger.warning(
                        f"[ShareGPTSFTCollator] No assistant prefix after truncation for sample {b_idx}. "
                        f"Message (truncated): {messages[b_idx]}"
                    )
                    continue
                boundary = sorted(set(asst_list + user_pos[b_idx].tolist() + sys_pos[b_idx].tolist()))
                for pos in asst_list:
                    start_idx = pos + len(self.assistant_prefix_ids)
                    next_candidates = [p for p in boundary if p > pos]
                    end_idx = min(next_candidates) if next_candidates else seq_len
                    if start_idx < end_idx:
                        labels[b_idx, start_idx:end_idx] = batch["input_ids"][b_idx, start_idx:end_idx]

        elif self.loss_type == "format_answer":
            labels = torch.full_like(batch["input_ids"], -100)
            pattern = torch.tensor(self.assistant_prefix_ids, device=labels.device)
            wins = batch["input_ids"].unfold(
                dimension=-1, size=pattern.numel(), step=1
            )
            match_mask = (wins == pattern).all(dim=-1)

            p_open = torch.tensor(self.think_open_ids, device=labels.device)
            p_close = torch.tensor(self.think_close_ids, device=labels.device)
            wins_close = batch["input_ids"].unfold(
                dimension=-1, size=p_close.numel(), step=1
            )
            match_close_mask = (wins_close == p_close).all(dim=-1)

            for b_idx in range(match_mask.size(0)):
                pos = torch.nonzero(match_mask[b_idx], as_tuple=False).squeeze(1)
                if pos.numel() == 0:
                    logger.warning(
                        f"[ShareGPTSFTCollator] No assistant prefix after truncation for sample {b_idx}. "
                        f"Message (truncated): {messages[b_idx]}"
                    )
                    continue

                think_open_start = pos[-1].item() + pattern.numel()
                think_open_end = think_open_start + p_open.numel()
                labels[b_idx, think_open_start:think_open_end] = batch["input_ids"][
                    b_idx, think_open_start:think_open_end
                ]

                close_pos = torch.nonzero(
                    match_close_mask[b_idx], as_tuple=False
                ).squeeze(1)
                if close_pos.numel() == 0:
                    logger.warning(
                        f"[ShareGPTSFTCollator] No </think> after truncation for sample {b_idx}. "
                        f"Message (truncated): {messages[b_idx]}"
                    )
                    continue

                think_close_start = close_pos[-1].item()
                labels[b_idx, think_close_start:] = batch["input_ids"][
                    b_idx, think_close_start:
                ]
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} is not supported.")

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        visual_token_ids: List[int] = []
        if hasattr(self.processor, "image_token"):
            visual_token_ids.append(
                self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            )
        if hasattr(self.processor, "video_token"):
            visual_token_ids.append(
                self.tokenizer.convert_tokens_to_ids(self.processor.video_token)
            )

        for visual_token_id in visual_token_ids:
            labels[labels == visual_token_id] = -100

        batch["labels"] = labels
        batch["qids"] = qids
        batch["metas"] = metas
        return batch
