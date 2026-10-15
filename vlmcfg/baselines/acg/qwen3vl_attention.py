"""ACG's attention intervention, retargeted to Qwen3-VL.

`InternVLACGIntervention` is already written against the `BackboneAdapter` protocol --
every backbone-specific operation (recompute q/k, read the KV cache, GQA-expand) is
delegated, and the rest of the class touches only `layer.self_attn`,
`attn.o_proj.weight`, `attn.layer_idx` and `attn.config.layer_types`, all of which exist
on Qwen3-VL. The ONLY backbone-specific line is the adapter it constructs.

So this is a one-line subclass rather than a 200-line copy. That is deliberate:
duplicating the intervention would mean two places to fix the orthogonalisation or the
splice, and this repo already carries the scar of four divergent `find_image_span`
copies.

PRECONDITIONS VERIFIED ON Qwen3-VL-8B (config.json), because ACG's bitwise neutral point
depends on both:
  * `text_config.attention_bias = False` -> `o_proj.bias is None`, so
    `O_final W_O = O_cond W_O + gamma*(dO_perp W_O)` is an ADDITIVE splice and
    `gamma = 0` contributes exactly +-0.0 while every line still executes.
  * `layer_types` absent -> all 36 layers are full attention, so the cache-history
    recompute is always valid. (This is what made Qwen3.5 unportable for ACG: 24 of its
    32 layers are linear-attention and hold no KV at all.)
"""

from __future__ import annotations

from vlmcfg.baselines.acg.internvl_attention import InternVLACGIntervention
from vlmcfg.baselines.common.backbone import Qwen3VLAdapter


class Qwen3VLACGIntervention(InternVLACGIntervention):
    """ACG on Qwen3-VL: identical mechanism, Qwen3-VL's q_norm/k_norm + mRoPE preamble.

    Using the inherited (InternVL3) adapter here would be the silent-failure case: it
    applies plain `qwen3.apply_rotary_pos_emb` with no mRoPE section handling, so the
    recomputed scores would be wrong while the method still ran. `Qwen3VLAdapter` is
    verified bit-exact against the real attention by
    `common/test_qwen3vl_recompute_unit.py`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._adapter = Qwen3VLAdapter()
