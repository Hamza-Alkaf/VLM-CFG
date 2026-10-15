"""InternVL3.5 in its native transformers packaging (`OpenGVLab/InternVL3_5-*-HF`)."""

import torch
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration

from vlmcfg.decoding import Stream


class InternVL35HF:
    """InternVL3.5-HF as a guidance backbone.

    Plain 1D positions, so a stream step is a direct forward with the cache.
    `max_patches` tiles plus one thumbnail tile per image.
    """

    model_cls = InternVLForConditionalGeneration
    blend_form = "conditional"

    def __init__(
        self,
        pretrained: str,
        dtype: str = "bfloat16",
        max_patches: int = 12,
        attn_implementation: str | None = None,
        device_map: str | None = None,
        device: str = "cuda",
    ) -> None:
        kwargs: dict = {"dtype": getattr(torch, dtype)}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            kwargs["device_map"] = device_map
            self.model = self.model_cls.from_pretrained(pretrained, **kwargs).eval()
        else:
            self.model = (
                self.model_cls.from_pretrained(
                    pretrained, low_cpu_mem_usage=True, **kwargs
                )
                .eval()
                .to(device)
            )
        self.input_device = "cuda" if device_map == "auto" else device
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.tokenizer = self.processor.tokenizer
        self.max_patches = max_patches
        eos = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        self.eos_ids = frozenset(i for i in eos if i is not None and i >= 0)

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.model.lm_head

    # ------------------------------------------------------------------ guided decoding
    def build_inputs(self, images: list[Image.Image] | None, text: str) -> dict:
        """Inputs for one stream;
        `images=None` builds the image-free negative stream."""
        content = [{"type": "image", "image": im} for im in images or []]
        content.append({"type": "text", "text": text})
        tiling = (
            {"crop_to_patches": True, "max_patches": self.max_patches} if images else {}
        )
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **tiling,
        ).to(self.input_device)

    def new_stream(self, inputs: dict) -> Stream:
        return Stream(
            ids=inputs["input_ids"],
            mask=inputs["attention_mask"],
            extra={"pixel_values": inputs.get("pixel_values")},
        )

    def past_length(self, stream: Stream) -> int:
        return stream.past.get_seq_length() if stream.past is not None else 0

    def step(self, stream: Stream, first: bool, feature: str) -> torch.Tensor:
        kwargs = {
            "past_key_values": stream.past,
            "use_cache": True,
            "output_hidden_states": feature == "hidden",
        }
        if first:
            out = self.model(
                input_ids=stream.ids,
                attention_mask=stream.mask,
                pixel_values=stream.extra["pixel_values"],
                **kwargs,
            )
        else:
            out = self.model(
                input_ids=stream.ids[:, -1:], attention_mask=stream.mask, **kwargs
            )
        stream.past = out.past_key_values
        if feature == "hidden":
            return out.hidden_states[-1][:, -1, :]
        return out.logits[:, -1, :]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    # ---------------------------------------------------------------- unguided baseline
    def build_chat_inputs(
        self, messages: list[dict], images: list[Image.Image]
    ) -> tuple[list[str], dict]:
        texts = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )
        if isinstance(texts, str):
            texts = [texts]
        kwargs: dict = {"text": texts, "return_tensors": "pt"}
        if images:
            kwargs.update(
                images=images, crop_to_patches=True, max_patches=self.max_patches
            )
        return texts, self.processor(**kwargs).to(self.input_device)

    @torch.no_grad()
    def generate(self, inputs: dict, gen_kwargs: dict, **extra) -> str:
        pad = self.tokenizer.pad_token_id
        out = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad if pad is not None else self.tokenizer.eos_token_id,
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
