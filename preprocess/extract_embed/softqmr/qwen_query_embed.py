import torch
import numpy as np
from typing import Dict, List, Optional
from .debug import dprint
from .device import resolve_device

from extract_embed.vendor.qwen3_vl_embedding import Qwen3VLEmbedder


def pool_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    flipped = attention_mask.flip(dims=[1])
    last_one_positions = flipped.argmax(dim=1)
    col = attention_mask.shape[1] - last_one_positions - 1
    row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row, col]


class QwenQueryEmbedder:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-Embedding-2B",
        device=None,
        default_instruction: str = "Represent the user's input.",
        video_num_frames: int | None = 64,
        video_max_frames: int | None = 64,
    ):
        self.embedder = Qwen3VLEmbedder(
            model_id,
            device=device,
            default_instruction=default_instruction,
            num_frames=video_num_frames,
            max_frames=video_max_frames,
        )
        self.model = self.embedder.model
        self.processor = self.embedder.processor
        dev = resolve_device(device)
        if dev is not None:
            self.model.to(dev)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        dprint("[QueryEmbedder] loaded", model_id, "device=", self.model.device)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        emb = self.embedder.process([{"text": text}], normalize=True)
        return emb.float().detach().cpu().numpy()

    @torch.no_grad()
    def embed_media(self, path: str, media_type: str = "image") -> np.ndarray:
        if media_type == "image":
            emb = self.embedder.process([{"image": path}], normalize=True)
        else:
            emb = self.embedder.process([{"video": path, "fps": 1, "max_frames": 16}], normalize=True)
        return emb.float().detach().cpu().numpy()

    @torch.no_grad()
    def embed_video_frames(
        self,
        frames: List["Image.Image"],
        instruction: str | None = None,
    ) -> np.ndarray:
        return self.embed_video_frames_batch([frames], instruction=instruction)

    @torch.no_grad()
    def embed_video_frames_batch(
        self,
        frames_batch: List[List["Image.Image"]],
        instruction: str | None = None,
    ) -> np.ndarray:
        payloads = []
        for frames in frames_batch:
            payload = {"video": frames}
            if instruction:
                payload["instruction"] = instruction
            payloads.append(payload)

        if not payloads:
            raise ValueError("frames_batch is empty")

        emb = self.embedder.process(payloads, normalize=True)
        return emb.float().detach().cpu().numpy()

    @torch.no_grad()
    def embed_video_paths_batch(
        self,
        video_paths: List[str],
        instruction: str | None = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        frame_size: Optional[int] = None,
        starts: Optional[List[float]] = None,
        ends: Optional[List[float]] = None,
    ) -> np.ndarray:
        if not video_paths:
            raise ValueError("video_paths is empty")
        if starts is not None and len(starts) != len(video_paths):
            raise ValueError("starts length must match video_paths length")
        if ends is not None and len(ends) != len(video_paths):
            raise ValueError("ends length must match video_paths length")

        payloads = []
        for i, path in enumerate(video_paths):
            payload = {"video": path}
            if instruction:
                payload["instruction"] = instruction
            if fps is not None and fps > 0:
                payload["fps"] = float(fps)
            if max_frames is not None and max_frames > 0:
                payload["max_frames"] = int(max_frames)
            if frame_size is not None and frame_size > 0:
                payload["resized_height"] = int(frame_size)
                payload["resized_width"] = int(frame_size)
            if starts is not None:
                payload["video_start"] = float(starts[i])
            if ends is not None:
                payload["video_end"] = float(ends[i])
            payloads.append(payload)

        emb = self.embedder.process(payloads, normalize=True)
        return emb.float().detach().cpu().numpy()

    def _make_base_inputs(self, query_text: str) -> Dict[str, torch.Tensor]:
        conv = self.embedder.format_model_input(
            text=query_text, instruction="Represent the query for retrieving relevant media."
        )
        inputs = self.embedder._preprocess_inputs([conv])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def _forward_with_soft_tokens(self, query_text: str, soft_tokens: torch.Tensor) -> torch.Tensor:
        """
        soft_tokens: (K, D=2048)
        return: (1, D) torch tensor
        """
        inputs = self._make_base_inputs(query_text)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        tok_emb = self.model.get_input_embeddings()(input_ids)

        K, D = soft_tokens.shape
        soft = soft_tokens.view(1, K, D).to(tok_emb.dtype).to(tok_emb.device)

        inputs_embeds = torch.cat([tok_emb, soft], dim=1)
        attn2 = torch.cat(
            [
                attention_mask,
                torch.ones((1, K), device=attention_mask.device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )

        dprint(
            "[EmbedInject] input_ids",
            tuple(input_ids.shape),
            "tok_emb",
            tuple(tok_emb.shape),
            "soft",
            tuple(soft.shape),
            "inputs_embeds",
            tuple(inputs_embeds.shape),
        )

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn2, return_dict=True)
        last_hidden = out.last_hidden_state

        pooled = pool_last(last_hidden, attn2)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        pooled = pooled.float()

        dprint(
            "[EmbedInject] last_hidden",
            tuple(last_hidden.shape),
            "pooled",
            tuple(pooled.shape),
            "norm",
            float(torch.norm(pooled)),
        )
        return pooled

    @torch.no_grad()
    def embed_text_with_soft_tokens(self, query_text: str, soft_tokens: torch.Tensor) -> np.ndarray:
        pooled = self._forward_with_soft_tokens(query_text, soft_tokens)
        return pooled.detach().cpu().numpy()

    @torch.no_grad()
    def embed_query_soft_nograd(self, query_text: str, soft_tokens: torch.Tensor) -> np.ndarray:
        pooled = self._forward_with_soft_tokens(query_text, soft_tokens)
        return pooled.detach().cpu().numpy()

    def forward_query_soft(self, query_text: str, soft_tokens: torch.Tensor) -> torch.Tensor:
        return self._forward_with_soft_tokens(query_text, soft_tokens)
