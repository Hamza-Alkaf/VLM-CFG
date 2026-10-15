"""PAI's last-row image-attention amplification on InternVL3, spliced as a delta.

    A[:, :, -1, img_lo:img_hi] += alpha * |A[:, :, -1, img_lo:img_hi]|

THE DELTA FORM IS THE POINT. The splice is the DIFFERENCE of two identical-path
recomputes (`out_biased - out_plain`), not one recompute's absolute value, so the
fused-kernel-vs-manual noise appears in both with the same sign and cancels. At
`alpha = 0` the bias is zero, the two recomputes are the same tensor, and the delta is
exactly `0.0` -- bit-identical to an unintervened run, without a guard. See
`attention.py` for the Qwen2.5-VL original.

PORTING NOTES. The preamble is delegated to `InternVL3Adapter.recompute_qk` (Qwen3's
`q_norm`/`k_norm` before RoPE; plain RoPE, no mrope_section), and `_attend_and_project`
is imported from the Qwen module rather than copied -- it is pure tensor math over
`attn.o_proj.weight`, which is bias-free on both backbones.
"""

from collections.abc import Callable

import torch
from torch import nn

from vlmcfg.baselines.common.backbone import InternVL3Adapter, neg_inf
from vlmcfg.baselines.pai.attention import _attend_and_project


class InternVLPAIIntervention:
    """PAI's last-row image-attention amplification on InternVL3.

    `use_cfg` is PAI's own switch: `CFGLogits` raises it around the unconditional
    (text-only) forward, where the `[img_lo, img_hi)` columns no longer hold image
    tokens and boosting them would corrupt the negative branch.
    """

    def __init__(
        self, layer_lo: int, layer_hi: int, img_lo: int, img_hi: int, alpha: float
    ) -> None:
        if not layer_lo < layer_hi:
            raise ValueError(f"Empty layer range [{layer_lo}, {layer_hi}).")
        if not 0 <= img_lo < img_hi:
            raise ValueError(f"Invalid image span [{img_lo}, {img_hi}).")
        self.layer_lo, self.layer_hi = layer_lo, layer_hi
        self.img_lo, self.img_hi = img_lo, img_hi
        self.alpha = float(alpha)
        self.use_cfg = False
        self._patched: dict[nn.Module, Callable] = {}
        self._adapter = InternVL3Adapter()

    def install(self, layers: nn.ModuleList) -> None:
        for layer in layers:
            attn = layer.self_attn
            layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None or not (self.layer_lo <= layer_idx < self.layer_hi):
                continue
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def _make_wrapper(self, attn: nn.Module, original: Callable):
        img_lo, img_hi = self.img_lo, self.img_hi
        adapter = self._adapter

        def wrapper(hidden_states, *args, **kwargs):
            attn_output, attn_weights = original(hidden_states, *args, **kwargs)

            # PAI's guard: no bias while the CFG branch runs its text-only forward.
            if self.use_cfg:
                return attn_output, attn_weights

            attention_mask = kwargs.get("attention_mask")
            position_embeddings = kwargs.get("position_embeddings")
            past_key_values = kwargs.get("past_key_values")
            if past_key_values is None:
                raise RuntimeError(
                    "PAI requires use_cache=True with a real Cache; it targets "
                    "generate()'s incremental-decoding path."
                )
            if position_embeddings is None:
                raise RuntimeError(
                    "PAI requires position_embeddings (RoPE cos/sin) to be passed."
                )

            with torch.no_grad():
                bsz = hidden_states.shape[0]
                if bsz != 1:
                    raise RuntimeError("PAI only supports batch_size=1 (no padding).")

                q, _ = adapter.recompute_qk(attn, hidden_states, position_embeddings)
                keys, values = adapter.read_cache(past_key_values, attn.layer_idx)
                key_states = adapter.expand_kv(keys, attn)
                value_f32 = adapter.expand_kv(values, attn).float()
                s_kv = key_states.shape[-2]

                q_last = q[:, :, -1:, :]
                scores = torch.matmul(q_last, key_states.transpose(2, 3)) * attn.scaling
                if attention_mask is not None:
                    mask = attention_mask
                    if mask.dtype == torch.bool:
                        mask = torch.where(mask, 0.0, neg_inf(scores.dtype)).to(
                            scores.dtype
                        )
                    scores = scores + mask[:, :, -1:, :s_kv]

                # fp32 from here: the bias, both softmaxes, both p@V and both o_proj, so
                # the ONLY dtype boundary is the final cast of the delta.
                scores_f32 = scores.float()

                ### PAI's modification
                a_img = scores_f32[:, :, :, img_lo:img_hi]
                scores_biased = scores_f32.clone()
                scores_biased[:, :, :, img_lo:img_hi] = a_img + self.alpha * a_img.abs()
                ### PAI's modification

                p_plain = nn.functional.softmax(scores_f32, dim=-1, dtype=torch.float32)
                p_biased = nn.functional.softmax(
                    scores_biased, dim=-1, dtype=torch.float32
                )
                out_plain = _attend_and_project(attn, p_plain, value_f32, bsz)
                out_biased = _attend_and_project(attn, p_biased, value_f32, bsz)
                delta = out_biased - out_plain

                new_output = attn_output.clone()
                new_output[:, -1, :] = attn_output[:, -1, :] + delta[:, 0, :].to(
                    attn_output.dtype
                )

            return new_output, attn_weights

        return wrapper

    def remove(self) -> None:
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()

    def __enter__(self) -> "InternVLPAIIntervention":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
