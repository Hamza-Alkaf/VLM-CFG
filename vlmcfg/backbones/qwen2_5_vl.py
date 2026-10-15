"""Qwen2.5-VL: loading, input building and one forward step per stream."""

import contextlib
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.cache_utils import DynamicCache

from vlmcfg.decoding import Stream
from vlmcfg.frames import record_visual_input_frame

QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 1605632


class Qwen25VL:
    """Qwen2.5-VL as a guidance backbone.

    Guidance needs the hidden state lm_head reads; a forward hook on lm_head captures it
    instead of `output_hidden_states=True`, which would keep every layer of the prefill.
    """

    model_cls = Qwen2_5_VLForConditionalGeneration
    blend_form = "reference"
    # whether the unguided baseline publishes its input frame
    # (the paper's Qwen2.5 one did)
    base_records_frame = True

    def __init__(
        self,
        pretrained: str,
        device_map: str = "auto",
        attn_implementation: str | None = None,
        max_pixels: int = QWEN_MAX_PIXELS,
        min_pixels: int = QWEN_MIN_PIXELS,
        dtype: str = "bfloat16",
    ) -> None:
        kwargs: dict = {"torch_dtype": dtype, "device_map": device_map}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        self.model = self.model_cls.from_pretrained(pretrained, **kwargs).eval()
        # The checkpoint ships repetition_penalty 1.05, which generate() would apply to
        # the unguided baseline only; guided decoding builds no logits processors.
        if self.model.generation_config is not None:
            self.model.generation_config.repetition_penalty = 1.0
        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.input_device = "cuda" if device_map == "auto" else device_map
        eos = {
            self.tokenizer.eos_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        }
        self.eos_ids = frozenset(i for i in eos if i is not None)
        self._hidden: torch.Tensor | None = None
        self.model.lm_head.register_forward_hook(self._capture_hidden)

    def _capture_hidden(self, _module, inputs, _output) -> None:
        self._hidden = inputs[0]

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.model.lm_head

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, image: Image.Image | None, text: str) -> dict:
        """Inputs for one stream; `image=None` builds the image-free negative stream.

        The image entry carries no max/min_pixels, as in the chat pipeline the paper's
        baseline ran on: passing them changes the patch grid and the image tokens.
        """
        content = [] if image is None else [{"type": "image", "image": image}]
        content.append({"type": "text", "text": text})
        msgs = [[{"role": "user", "content": content}]]
        texts = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(msgs)
        inputs = self.processor(
            text=texts, images=images, videos=videos, padding=True, return_tensors="pt"
        )
        if image is not None:
            self._record_frame(image.size, inputs)
        return inputs.to(self.input_device)

    def build_chat_inputs(self, messages: list[dict]) -> tuple[list[str], dict]:
        """Inputs for the unguided baseline from lmms-eval chat messages (HF format)."""
        batched = [messages]
        texts = self.processor.apply_chat_template(
            batched, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(batched)
        if videos is not None:
            raise NotImplementedError("video inputs are not supported")
        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )
        source = first_image(messages)
        if source is not None and self.base_records_frame:
            self._record_frame(source.size, inputs)
        return texts, inputs.to(self.input_device)

    def _record_frame(self, source_size: tuple[int, int], inputs: dict) -> None:
        if "image_grid_thw" not in inputs:
            return
        grid = inputs["image_grid_thw"][0]
        patch = self.processor.image_processor.patch_size
        record_visual_input_frame(
            source_size=source_size,
            input_size=(int(grid[2]) * patch, int(grid[1]) * patch),
        )

    # ------------------------------------------------------------------ guided decoding
    def new_stream(self, inputs: dict) -> Stream:
        """A fresh stream over processor `inputs`."""
        extra = {
            k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")
        }
        return Stream(
            ids=inputs["input_ids"],
            mask=inputs["attention_mask"],
            past=DynamicCache(),
            extra=extra,
        )

    def past_length(self, stream: Stream) -> int:
        return stream.past.get_seq_length()

    def step(self, stream: Stream, first: bool, feature: str) -> torch.Tensor:
        """One forward through the model's own `prepare_inputs_for_generation`.

        The mRoPE offsets (`rope_deltas`) live on the shared model, so each stream parks
        its own and restores them around the call; otherwise the two streams overwrite
        each other's positions after the prefill.
        """
        inner = self.model.model
        inner.rope_deltas = stream.state.get("rope_deltas")
        n_past = stream.past.get_seq_length()
        cache_position = torch.arange(
            n_past, stream.ids.shape[1], device=stream.ids.device
        )
        model_inputs = self.model.prepare_inputs_for_generation(
            stream.ids,
            past_key_values=stream.past,
            attention_mask=stream.mask,
            cache_position=cache_position,
            use_cache=True,
            **(stream.extra if first else {}),
        )
        context = stream.state.get("context")
        exit_layer = stream.state.get("exit_layer")
        # Logits are read out for the last position only, as HF generate does: lm_head
        # over the whole prompt is a different GEMM shape and rounds the last row
        # differently.
        keep = {"logits_to_keep": 1} if feature == "logits" else {}
        with context() if context is not None else contextlib.nullcontext():
            out = self.model(
                **model_inputs, output_hidden_states=exit_layer is not None, **keep
            )
        stream.state["rope_deltas"] = inner.rope_deltas
        stream.past = out.past_key_values
        if exit_layer is not None:
            return self._early_exit(out, exit_layer, feature)
        if feature == "hidden":
            assert self._hidden is not None
            return self._hidden[:, -1, :]
        return out.logits[:, -1, :]

    def _early_exit(self, out: Any, exit_layer: int, feature: str) -> torch.Tensor:
        """The residual stream after `exit_layer` layers,
        normed as the last layer is."""
        all_hidden = out.hidden_states  # [0] embeddings ... [N] the normed last layer
        n_layers = len(all_hidden) - 1
        k = exit_layer if exit_layer >= 0 else n_layers + 1 + exit_layer
        if not 0 <= k <= n_layers:
            raise ValueError(f"exit_layer {exit_layer} is outside [0, {n_layers}]")
        if k == n_layers:
            hidden = all_hidden[k]
        else:
            norm = self.model.model.language_model.norm
            hidden = norm(all_hidden[k].to(norm.weight.device))
        if feature == "hidden":
            return hidden[:, -1, :]
        return self.model.lm_head(hidden[:, -1:, :])[:, -1, :]

    def decode(self, ids: list[int]) -> str:
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()

    # ---------------------------------------------------------------- unguided baseline
    @torch.no_grad()
    def generate(self, inputs: dict, gen_kwargs: dict, **extra) -> str:
        """HF greedy generate, with the baseline's exact generation arguments.

        `extra` is passed through to `generate` (e.g. a method's logits processor).
        """
        out = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=gen_kwargs["do_sample"],
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            top_k=gen_kwargs.get("top_k"),
            use_cache=True,
            **extra,
        )
        trimmed = out[0][inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def first_image(messages: list[dict]) -> Image.Image | None:
    """The first image of lmms-eval chat messages, or None."""
    for message in messages:
        for part in message["content"]:
            if part["type"] == "image":
                return part["image"]
    return None
