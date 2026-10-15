"""PAI (Paying Attention to Image, ECCV 2024) model class for Qwen3-VL.

PAI intervenes entirely from OUTSIDE `forward()` -- an attention wrapper
(`Qwen3VLPAIIntervention`) and a `CFGLogits` LogitsProcessor -- so there is deliberately
nothing else to override here. Using the plain HF class rather than this repo's CFG
subclass keeps PAI isolated from the CFG bookkeeping.

LAYER RANGE. Qwen3-VL-8B has 36 decoder layers. The paper's Qwen2.5-VL arm used `[2,28]`
of 28; the proportional rescaling here is `[2,36]`, matching what was done for
InternVL3.5-8B. Passing `[2,28]` would silently leave the top 8 layers unintervened.

NEGATIVE PROMPT. PAI's upstream `build_negative_prompt` strips images with a regex over
the RENDERED text (`<|vision_start|>...<|vision_end|>`). That is not used here: the
negative stream is built by structurally filtering the message dicts, the same way this
repo's CFG `no_img/drop` does, because the regex breaks on multi-image prompts and on
any chat-template change.
"""

from __future__ import annotations

from transformers import Qwen3VLForConditionalGeneration


class Qwen3VLForConditionalGenerationPAI(Qwen3VLForConditionalGeneration):
    """The plain HF class plus the one hook the chat wrapper calls."""

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Required no-op; PAI's per-sample state lives on the handle, rebuilt per
        document."""
