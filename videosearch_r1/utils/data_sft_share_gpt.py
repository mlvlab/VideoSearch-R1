import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import yaml
from torch.utils.data import ConcatDataset, Dataset

from utils.video_metadata import (
    load_video_meta_index,
    resolve_video_meta_for_video_path,
)

logger = logging.getLogger(__name__)

_MEDIA_PATTERN = re.compile(r"(<video>|<image>)")


@dataclass
class ShareGPTDataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to ShareGPT-style JSON/JSONL file."},
    )
    dataset_info: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dataset yaml file for ShareGPT-style datasets."},
    )
    dataset_name: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of dataset names in the dataset config."},
    )
    max_samples: Optional[int] = field(
        default=None, metadata={"help": "Limit number of samples for debugging."}
    )
    messages_key: Optional[str] = field(
        default=None,
        metadata={"help": "Key for messages list. If None, auto-detect messages/conversations."},
    )
    role_key: Optional[str] = field(
        default=None,
        metadata={"help": "Key for role in each message. If None, auto-detect role/from."},
    )
    content_key: Optional[str] = field(
        default=None,
        metadata={"help": "Key for content in each message. If None, auto-detect content/value."},
    )
    videos_key: Optional[str] = field(
        default="videos",
        metadata={"help": "Key for video list in each sample."},
    )
    video_key: Optional[str] = field(
        default="video",
        metadata={"help": "Key for single video path in each sample."},
    )
    images_key: Optional[str] = field(
        default="images",
        metadata={"help": "Key for image list in each sample."},
    )
    image_key: Optional[str] = field(
        default="image",
        metadata={"help": "Key for single image path in each sample."},
    )
    video_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional root dir to prepend for video paths."},
    )
    video_root_override: Optional[str] = field(
        default=None,
        metadata={"help": "Optional override for dataset-config resolved video_root."},
    )
    video_meta_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional video meta jsonl path for npy temporal metadata."},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional root dir to prepend for image paths."},
    )
    strict_media_match: bool = field(
        default=True,
        metadata={"help": "Require all <video>/<image> tokens to match provided media list."},
    )
    skip_missing_files: bool = field(
        default=True,
        metadata={"help": "Skip samples if referenced media files are missing."},
    )

    # Visual sampling defaults (aligned with utils.arguments.DataArguments)
    image_min_pixels: int = field(default=4 * 28 * 28)
    image_max_pixels: int = field(default=16384 * 28 * 28)
    video_min_pixels: int = field(default=128 * 28 * 28)
    video_max_pixels: int = field(default=768 * 28 * 28)
    video_total_pixels: int = field(default=115200 * 28 * 28)
    max_frames: int = field(default=768)
    nframes: Optional[int] = field(default=None)
    fps: float = field(default=2.0)
    video_meta_warn_limit: int = field(
        default=20,
        metadata={"help": "Max warning count for missing video meta rows."},
    )

    dataset_config: Optional[Dict[str, Dict[str, Any]]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        if not self.dataset_info:
            return
        with open(self.dataset_info, "r", encoding="utf-8") as f:
            self.dataset_config = yaml.safe_load(f)
        if not isinstance(self.dataset_config, dict):
            raise ValueError("dataset_info must be a mapping of dataset configs.")

        if isinstance(self.dataset_name, str):
            self.dataset_name = [self.dataset_name]
        if self.dataset_name is None:
            self.dataset_name = list(self.dataset_config.keys())
        for name in self.dataset_name:
            if name not in self.dataset_config:
                raise ValueError(f"Dataset {name} not found in dataset config.")


def _apply_dataset_overrides(
    base_args: ShareGPTDataArguments, cfg: Dict[str, Any]
) -> ShareGPTDataArguments:
    args = replace(base_args)

    if "anno_path" in cfg:
        args.data_path = cfg["anno_path"]
    elif "data_path" in cfg:
        args.data_path = cfg["data_path"]
    elif "file_name" in cfg and "data_root" in cfg:
        args.data_path = os.path.join(cfg["data_root"], cfg["file_name"])
    elif "file_name" in cfg:
        args.data_path = cfg["file_name"]

    for key in (
        "messages_key",
        "role_key",
        "content_key",
        "videos_key",
        "video_key",
        "images_key",
        "image_key",
        "video_root",
        "video_root_override",
        "video_meta_path",
        "image_root",
        "strict_media_match",
        "skip_missing_files",
        "max_samples",
        "image_min_pixels",
        "image_max_pixels",
        "video_min_pixels",
        "video_max_pixels",
        "video_total_pixels",
        "max_frames",
        "nframes",
        "fps",
    ):
        if key in cfg:
            setattr(args, key, cfg[key])

    if "anno_path" in cfg and "data_path" in cfg:
        if "video_root" not in cfg:
            args.video_root = cfg["data_path"]
        if "image_root" not in cfg:
            args.image_root = cfg["data_path"]

    if args.video_root_override:
        args.video_root = args.video_root_override

    columns = cfg.get("columns", {})
    if isinstance(columns, dict):
        if "messages" in columns:
            args.messages_key = columns["messages"]
        if "videos" in columns:
            args.videos_key = columns["videos"]
        if "video" in columns:
            args.video_key = columns["video"]
        if "images" in columns:
            args.images_key = columns["images"]
        if "image" in columns:
            args.image_key = columns["image"]

    tags = cfg.get("tags", {})
    if isinstance(tags, dict):
        if "role_tag" in tags:
            args.role_key = tags["role_tag"]
        if "content_tag" in tags:
            args.content_key = tags["content_tag"]

    return args


def build_sharegpt_dataset(data_args: ShareGPTDataArguments) -> Dataset:
    if data_args.dataset_info:
        if not data_args.dataset_config:
            raise ValueError("dataset_info is set but dataset_config is empty.")
        datasets: List[Dataset] = []
        for name in data_args.dataset_name or []:
            cfg = data_args.dataset_config.get(name, {})
            local_args = _apply_dataset_overrides(data_args, cfg)
            if not local_args.data_path:
                raise ValueError(f"data_path missing for dataset {name}.")
            datasets.append(ShareGPTSFTDataset(local_args))
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    return ShareGPTSFTDataset(data_args)


class ShareGPTSFTDataset(Dataset):
    ROLE_MAP = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "environment": "user",
        "observation": "user",
        "tool": "user",
        "function": "assistant",
        "function_call": "assistant",
    }

    def __init__(self, data_args: ShareGPTDataArguments):
        super().__init__()
        if not data_args.data_path:
            raise ValueError("data_path is required for ShareGPTSFTDataset")
        self.data_args = data_args
        self._video_meta_index = load_video_meta_index(data_args.video_meta_path)
        self._video_meta_missing_warn_count = 0
        raw_samples = self._load_file(data_args.data_path)
        if data_args.max_samples is not None:
            raw_samples = raw_samples[: data_args.max_samples]

        self.samples: List[Dict[str, Any]] = []
        skipped = 0
        for idx, sample in enumerate(raw_samples):
            converted = self._convert_sample(sample)
            if converted is None:
                skipped += 1
                continue
            self.samples.append(converted)
        logger.info(
            f"Loaded {len(self.samples)} samples from {data_args.data_path} (skipped {skipped})."
        )

    def _load_file(self, path: str) -> List[Dict[str, Any]]:
        if path.endswith(".jsonl"):
            samples = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    samples.append(json.loads(line))
            return samples
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("Unsupported data format for ShareGPT dataset.")

    def _resolve_path(self, path: str, root: Optional[str]) -> str:
        if root and not os.path.isabs(path):
            path = os.path.join(root, path)
        return os.path.normpath(path)

    def _make_text(self, text: str) -> Dict[str, Any]:
        return {"type": "text", "text": text}

    def _make_image(self, path: str) -> Dict[str, Any]:
        path = self._resolve_path(path, self.data_args.image_root)
        return {
            "type": "image",
            "image": path,
            "min_pixels": self.data_args.image_min_pixels,
            "max_pixels": self.data_args.image_max_pixels,
        }

    def _make_video(self, path: str) -> Dict[str, Any]:
        path = self._resolve_path(path, self.data_args.video_root)
        payload = {
            "type": "video",
            "video": path,
            "min_pixels": self.data_args.video_min_pixels,
            "max_pixels": self.data_args.video_max_pixels,
            "total_pixels": self.data_args.video_total_pixels,
            "max_frames": self.data_args.max_frames,
            "fps": self.data_args.fps,
        }
        if self.data_args.nframes is not None:
            payload["nframes"] = self.data_args.nframes
            payload.pop("fps", None)

        meta_payload = resolve_video_meta_for_video_path(path, self._video_meta_index)
        if meta_payload:
            payload.update(meta_payload)
        elif (
            str(self.data_args.video_meta_path or "").strip()
            and self._video_meta_missing_warn_count < int(self.data_args.video_meta_warn_limit)
        ):
            self._video_meta_missing_warn_count += 1
            logger.warning(
                "Missing video meta for path=%s (meta=%s)",
                path,
                self.data_args.video_meta_path,
            )
        return payload

    def _split_text_with_media(
        self,
        text: str,
        media_state: Dict[str, List[str]],
        replace_media: bool,
    ) -> List[Dict[str, Any]]:
        if not replace_media or ("<video>" not in text and "<image>" not in text):
            return [self._make_text(text)]

        parts = _MEDIA_PATTERN.split(text)
        content: List[Dict[str, Any]] = []
        for part in parts:
            if part == "<video>":
                if not media_state["videos"]:
                    raise ValueError("Not enough videos for <video> tokens.")
                content.append(self._make_video(media_state["videos"].pop(0)))
            elif part == "<image>":
                if not media_state["images"]:
                    raise ValueError("Not enough images for <image> tokens.")
                content.append(self._make_image(media_state["images"].pop(0)))
            elif part:
                content.append(self._make_text(part))
        if not content:
            content.append(self._make_text(""))
        return content

    def _normalize_content(
        self,
        content: Any,
        media_state: Dict[str, List[str]],
        replace_media: bool,
    ) -> List[Dict[str, Any]]:
        if isinstance(content, list):
            normalized: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "text")
                    if item_type == "text":
                        text = item.get("text", item.get("value", ""))
                        normalized.extend(
                            self._split_text_with_media(text, media_state, replace_media)
                        )
                    elif item_type in ("image", "image_url"):
                        img = item.get("image", item.get("image_url", item.get("value")))
                        if img is None:
                            continue
                        normalized.append(self._make_image(img))
                    elif item_type == "video":
                        vid = item.get("video", item.get("value"))
                        if vid is None:
                            continue
                        normalized.append(self._make_video(vid))
                    else:
                        normalized.append(self._make_text(str(item)))
                elif isinstance(item, str):
                    normalized.extend(
                        self._split_text_with_media(item, media_state, replace_media)
                    )
                else:
                    normalized.append(self._make_text(str(item)))
            return normalized

        if isinstance(content, str):
            return self._split_text_with_media(content, media_state, replace_media)

        return [self._make_text(str(content))]

    def _extract_media_list(
        self, sample: Dict[str, Any], list_key: Optional[str], single_key: Optional[str]
    ) -> List[str]:
        media_list: List[str] = []
        if list_key and list_key in sample and sample[list_key] is not None:
            if isinstance(sample[list_key], list):
                media_list.extend(sample[list_key])
            elif isinstance(sample[list_key], str):
                media_list.append(sample[list_key])
        if single_key and single_key in sample and sample[single_key] is not None:
            if isinstance(sample[single_key], list):
                media_list.extend(sample[single_key])
            else:
                media_list.append(sample[single_key])
        return media_list

    def _check_media_files(self, media_list: List[str], root: Optional[str]) -> bool:
        if not self.data_args.skip_missing_files:
            return True
        for path in media_list:
            resolved = self._resolve_path(path, root)
            if not os.path.exists(resolved):
                logger.warning(f"Missing media file: {resolved}, skip sample.")
                return False
        return True

    def _convert_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages_key = self.data_args.messages_key
        if messages_key is None:
            if "messages" in sample:
                messages_key = "messages"
            elif "conversations" in sample:
                messages_key = "conversations"
            else:
                logger.warning("No messages/conversations field found, skip sample.")
                return None
        messages = sample.get(messages_key, [])
        if not isinstance(messages, list) or len(messages) == 0:
            logger.warning("Empty messages list, skip sample.")
            return None

        role_key = self.data_args.role_key
        content_key = self.data_args.content_key
        if role_key is None or content_key is None:
            first = messages[0]
            if role_key is None:
                if isinstance(first, dict) and "role" in first:
                    role_key = "role"
                elif isinstance(first, dict) and "from" in first:
                    role_key = "from"
                else:
                    logger.warning("Cannot detect role key, skip sample.")
                    return None
            if content_key is None:
                if isinstance(first, dict) and "content" in first:
                    content_key = "content"
                elif isinstance(first, dict) and "value" in first:
                    content_key = "value"
                else:
                    logger.warning("Cannot detect content key, skip sample.")
                    return None

        videos = self._extract_media_list(sample, self.data_args.videos_key, self.data_args.video_key)
        images = self._extract_media_list(sample, self.data_args.images_key, self.data_args.image_key)

        if not self._check_media_files(videos, self.data_args.video_root):
            return None
        if not self._check_media_files(images, self.data_args.image_root):
            return None

        media_state = {"videos": list(videos), "images": list(images)}

        processed_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning("Invalid message format, skip sample.")
                return None
            raw_role = msg.get(role_key)
            raw_content = msg.get(content_key, "")
            if raw_role is None:
                logger.warning("Missing role in message, skip sample.")
                return None
            norm_role = self.ROLE_MAP.get(str(raw_role), "user")
            replace_media = str(raw_role) != "system" and norm_role != "system"
            try:
                content_list = self._normalize_content(raw_content, media_state, replace_media)
            except ValueError as exc:
                logger.warning(f"{exc} skip sample.")
                return None
            processed_messages.append({"role": norm_role, "content": content_list})

        if self.data_args.strict_media_match:
            if media_state["videos"] or media_state["images"]:
                logger.warning("Unused media items after parsing, skip sample.")
                return None

        if not processed_messages or processed_messages[-1]["role"] != "assistant":
            logger.warning("Last message is not assistant, skip sample.")
            return None

        response_message = processed_messages[-1]
        response = self._content_to_text(response_message["content"])
        prompt_messages = processed_messages[:-1]
        if len(prompt_messages) == 0:
            logger.warning("No prompt messages after removing response, skip sample.")
            return None

        output = {
            "messages": prompt_messages,
            "response": response,
        }
        if "qid" in sample:
            output["qid"] = sample["qid"]
        if "meta" in sample:
            output["meta"] = sample["meta"]
        return output

    def _content_to_text(self, content_list: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in content_list:
            item_type = item.get("type", "text")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "image":
                parts.append("<image>")
            elif item_type == "video":
                parts.append("<video>")
            else:
                parts.append(str(item))
        return "".join(parts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]