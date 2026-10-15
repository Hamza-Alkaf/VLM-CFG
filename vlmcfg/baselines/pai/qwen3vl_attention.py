"""PAI's attention intervention, retargeted to Qwen3-VL.

Like ACG's, `InternVLPAIIntervention` is written against the `BackboneAdapter` protocol,
so the only backbone-specific line is the adapter it builds. This is a one-line subclass
rather than a copy, for the same reason: two copies means two places to fix the
amplification.

WHY THE ADAPTER MATTERS HERE SPECIFICALLY. PAI amplifies the last row's attention over
the image columns, which requires recomputing that row. On Qwen3-VL that recompute must
apply `q_norm`/`k_norm` before the (plain, pre-interleaved) rotary. InternVL3's adapter
would apply a different rotary and no norms -- the boost would then be applied to a
wrongly-ranked row, silently. `Qwen3VLAdapter` is verified bit-exact in
`common/test_qwen3vl_recompute_unit.py`.

NEUTRAL POINT. `pai_alpha = 0.0` is bitwise neutral by construction: the
amplification is
`attn_weights[..., -1, img_lo:img_hi] += alpha * |attn_weights[..., -1, img_lo:img_hi]|`
(or the equivalent in the upstream form), so alpha 0 adds exactly +-0.0 while every line
still executes.
"""

from __future__ import annotations

from vlmcfg.baselines.common.backbone import Qwen3VLAdapter
from vlmcfg.baselines.pai.internvl_attention import InternVLPAIIntervention


class Qwen3VLPAIIntervention(InternVLPAIIntervention):
    """PAI on Qwen3-VL: identical mechanism, Qwen3-VL's q_norm/k_norm + mRoPE
    preamble."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._adapter = Qwen3VLAdapter()
