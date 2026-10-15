"""Molmo2 (allenai/Molmo2-4B, remote code), as a guidance backbone."""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlmcfg.decoding import Stream

# The checkpoint ships its own modelling code; pin the revision the paper ran.
MOLMO2_REVISION = "042abfa7a38879a376cec03d949eff0aefaa0600"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


class Molmo2:
    """Molmo2 as a guidance backbone.

    Plain 1D RoPE, so streams call the inner model directly. The prefill must carry
    every processor output (notably `token_type_ids`, which make attention among image
    tokens bidirectional). The inner model's `last_hidden_state` is what lm_head reads.
    """

    blend_form = "conditional"

    def __init__(
        self,
        pretrained: str,
        dtype: str = "float16",
        max_crops: int = 8,
        revision: str | None = MOLMO2_REVISION,
        device: str = "cuda",
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=True, revision=revision
        )
        self.tokenizer = self.processor.tokenizer
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                pretrained,
                trust_remote_code=True,
                revision=revision,
                dtype=getattr(torch, dtype),
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(device)
        )
        self.device = device
        self.max_crops = max_crops
        stop = {self.tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS}
        self.stop_ids = sorted(i for i in stop if i is not None and i >= 0)
        self.eos_ids = frozenset(self.stop_ids)

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.model.lm_head

    def build_inputs(self, images: list[Image.Image] | None, text: str) -> dict:
        """Inputs for one stream;
        `images=None` builds the image-free negative stream."""
        content = [{"type": "image", "image": im} for im in images or []]
        content.append({"type": "text", "text": text})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        image_kwargs = {"max_crops": self.max_crops} if images else {}
        return self.processor(
            text=prompt, images=images or None, return_tensors="pt", **image_kwargs
        ).to(self.device)

    def new_stream(self, inputs: dict) -> Stream:
        extra = {
            k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")
        }
        return Stream(
            ids=inputs["input_ids"], mask=inputs["attention_mask"], extra=extra
        )

    def past_length(self, stream: Stream) -> int:
        return stream.past.get_seq_length()

    def step(self, stream: Stream, first: bool, feature: str) -> torch.Tensor:
        inner = self.model.model
        if first:
            out = inner(
                input_ids=stream.ids,
                attention_mask=stream.mask,
                use_cache=True,
                **stream.extra,
            )
        else:
            out = inner(
                input_ids=stream.ids[:, -1:],
                attention_mask=stream.mask,
                past_key_values=stream.past,
                use_cache=True,
            )
        stream.past = out.past_key_values
        h = out.last_hidden_state[:, -1:, :]
        if feature == "hidden":
            return h[:, -1, :]
        return self.lm_head(h)[:, -1, :]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate(self, inputs: dict, max_new_tokens: int, **extra) -> str:
        """The unguided baseline: greedy HF generate on the same inputs.

        `extra` is passed through to `generate` (e.g. a method's logits processor).
        """
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            **extra,
        )
        return self.decode(out[0, inputs["input_ids"].shape[1] :].tolist())
