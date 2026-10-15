"""One place that knows where Qwen2.5-VL's mRoPE lives, across transformers versions.

WHY THIS EXISTS
---------------
`transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb` was
a module-level helper in transformers 4.x and was **removed in 5.x**. Every site that
reproduces a Qwen2.5-VL attention row outside the fused kernel imported it directly, so
all of them raise `ImportError` on 5.x.

The behavioural change is not just a rename:

* **4.x** -- `position_embeddings` carries the un-interleaved 3-way
  (temporal/height/width) frequencies, and the caller splits them via
  `apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)`.
* **5.x** -- `Qwen2_5_VLRotaryEmbedding` owns `mrope_section` and interleaves the
  frequencies itself, so `position_embeddings` arrives already mRoPE-combined and
  attention calls the PLAIN `apply_rotary_pos_emb(q, k, cos, sin)` (see
  `Qwen2_5_VLAttention.forward`).

Applying the section split twice -- or not at all -- does NOT raise. It yields finite,
fluent-looking output with silently wrong attention scores, which is exactly the failure
mode ACG/PAI/SID cannot tolerate. So this is centralised in one place; the ports that
use it are verified bitwise against the runs they reproduce.

Also handles the config rename `rope_scaling` -> `rope_parameters` (4.x -> 5.x).
"""

from collections.abc import Sequence
from typing import Any

import torch


def mrope_section(attn: Any) -> Sequence[int] | None:
    """Find `mrope_section` on an attention module or its config, across the 4.x/5.x
    rename.

    Returns None when neither spelling carries one (e.g. a plain-RoPE backbone).
    """
    for holder in (attn, getattr(attn, "config", None)):
        if holder is None:
            continue
        for key in ("rope_scaling", "rope_parameters"):
            cfg = getattr(holder, key, None)
            if isinstance(cfg, dict) and cfg.get("mrope_section") is not None:
                return cfg["mrope_section"]
    return None


def apply_qwen_rope(
    attn: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen2.5-VL rotary embeddings to (q, k) the way the installed version does.

    Use this instead of importing `apply_multimodal_rotary_pos_emb` directly.
    """
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb,
        )
    except ImportError:
        # transformers >= 5: mRoPE already folded into `position_embeddings`.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_rotary_pos_emb,
        )

        return apply_rotary_pos_emb(q, k, cos, sin)

    section = mrope_section(attn)
    if section is None:
        raise RuntimeError(
            "apply_qwen_rope found apply_multimodal_rotary_pos_emb (transformers 4.x) "
            "but no mrope_section on the attention module or its config (checked "
            "rope_scaling and rope_parameters). Refusing to guess: a wrong section "
            "split produces plausible-but-wrong attention scores rather than an error."
        )
    return apply_multimodal_rotary_pos_emb(q, k, cos, sin, section)


def model_forward_kwargs(model_forward, **candidate) -> dict:
    """Drop kwargs the installed `*Model.forward` does not declare.

    WHY: transformers 5.x REMOVED `cache_position` from `InternVLModel.forward` and
    `Qwen2_5_VLModel.forward`. Both still accept `**kwargs`, so passing it does NOT
    raise -- it is silently swallowed and never reaches the decoder layers. For any
    method that runs a second (auxiliary/negative) stream and then reads per-layer state
    back, that means the stream runs with different position bookkeeping than intended
    and the captured rows are misaligned. SID's CT2S selection degraded to
    `kept 0 of 256 image tokens` this way: finite, non-crashing, and completely inert.

    Passing an argument that lands in `**kwargs` unnoticed is the dangerous case, so
    filter against the real signature instead of assuming.

    Usage:
        kw = model_forward_kwargs(self.model.forward, cache_position=cache_position,
                                  image_sizes=image_sizes)
        out = self.model(input_ids=..., **kw)
    """
    import inspect

    params = inspect.signature(model_forward).parameters
    return {k: v for k, v in candidate.items() if k in params}
