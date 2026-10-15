"""ACG (Attention-space Contrastive Guidance) on InternVL3, spliced as an additive
correction.

The operator is unchanged from `acg_attention.py` -- `orthogonalize`, `_flatten_heads`,
the `ORTH_SCOPES` vocabulary and the `NEG_INF` sentinel are IMPORTED from it rather than
copied, so the two backbones cannot drift apart in the math. What changes is only the
recompute preamble and the layer/cache plumbing, both of which come from
`common.backbone.InternVL3Adapter`.

THE ADDITIVE SPLICE SURVIVES THE MOVE, which is the fact the whole design rests on.
`Qwen3Attention.o_proj` is `nn.Linear(..., bias=config.attention_bias)` and InternVL3's
`text_config.attention_bias` is `false`, so `W_O` is linear and::

    Z_final = O_final W_O = O_cond W_O + gamma * (dO_perp W_O)

with `O_cond W_O` exactly what the real sdpa kernel already returned. The module
therefore never replaces the attention output; it adds one term to its last row. At
`gamma = 0` that term is `0.0 * finite == +-0.0` and `x + +-0.0 == x` for every float,
so every line can execute -- both softmaxes, the mask, the orthogonalization, the
projection -- and still be bit-identical.

THE ONE GENUINELY NEW TRAP IS `q_norm`. Qwen3 applies an RMSNorm over `head_dim` to q
and k (and not to v) BEFORE RoPE, which Qwen2.5-VL does not have. Omitting it still runs
and still yields plausible logits; it just computes the wrong attention. The preamble is
therefore not written here at all but delegated to `InternVL3Adapter.recompute_qk`,
which `common/test_internvl_recompute_unit.py` pins against the real kernel (agreement
4e-7 relative, versus 5e-2 when the norms are dropped).

Scope, asserted rather than assumed: `batch_size == 1`, a real `Cache`, and a contiguous
image span (InternVL's video prompt is N disjoint per-frame runs; the adapter raises on
those rather than masking the interleaved text).
"""

from collections.abc import Callable

import torch
from torch import nn

from vlmcfg.baselines.acg.attention import (
    ORTH_SCOPES,
    _flatten_heads,
    neg_inf,
    orthogonalize,
)
from vlmcfg.baselines.common.backbone import InternVL3Adapter


class InternVLACGIntervention:
    """ACG's per-layer last-row correction on InternVL3, installed as an additive
    splice.

    Wraps `self_attn.forward` rather than replacing it -- a forward *hook* cannot be
    used because `query_states`/`key_states` are locals of `Qwen3Attention.forward`.
    """

    def __init__(
        self,
        layer_lo: int,
        layer_hi: int,
        img_lo: int,
        img_hi: int,
        gamma: float,
        orth_scope: str = "concat",
        eps: float = 1e-6,
        record_stats: bool = False,
    ) -> None:
        if not layer_lo < layer_hi:
            raise ValueError(f"Empty layer range [{layer_lo}, {layer_hi}).")
        if not 0 <= img_lo < img_hi:
            raise ValueError(f"Invalid image span [{img_lo}, {img_hi}).")
        if orth_scope not in ORTH_SCOPES:
            raise ValueError(
                f"acg_orth_scope must be one of {ORTH_SCOPES}, got {orth_scope!r}."
            )
        self.layer_lo, self.layer_hi = layer_lo, layer_hi
        self.img_lo, self.img_hi = img_lo, img_hi
        self.gamma = float(gamma)
        self.orth_scope = orth_scope
        self.eps = float(eps)
        self.record_stats = record_stats
        self.stats: dict[int, dict[str, float]] = {}
        self._patched: dict[nn.Module, Callable] = {}
        self._adapter = InternVL3Adapter()

    def install(self, layers: nn.ModuleList) -> None:
        for layer in layers:
            attn = layer.self_attn
            layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None or not (self.layer_lo <= layer_idx < self.layer_hi):
                continue
            layer_types = getattr(attn.config, "layer_types", None)
            if layer_types is not None and layer_types[layer_idx] != "full_attention":
                raise ValueError(
                    f"ACG only supports full_attention layers; layer {layer_idx} is "
                    f"{layer_types[layer_idx]!r}. The recompute reads "
                    "the whole key history back out of the cache, which a windowed "
                    "layer may not hold."
                )
            if attn in self._patched:
                continue  # never stack: install() runs once per forward
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def _make_wrapper(self, attn: nn.Module, original: Callable):
        img_lo, img_hi = self.img_lo, self.img_hi
        adapter = self._adapter

        def wrapper(hidden_states, *args, **kwargs):
            # The REAL forward first, untouched, on the model's real kernel: every row,
            # every head, and this layer's one and only cache write. Only the last row
            # is corrected.
            attn_output, attn_weights = original(hidden_states, *args, **kwargs)

            attention_mask = kwargs.get("attention_mask")
            position_embeddings = kwargs.get("position_embeddings")
            past_key_values = kwargs.get("past_key_values")
            if past_key_values is None:
                raise RuntimeError(
                    "ACG requires use_cache=True with a real Cache; "
                    "it reads the accumulated K/V back out of it."
                )
            if position_embeddings is None:
                raise RuntimeError(
                    "ACG requires position_embeddings (RoPE cos/sin) to be passed."
                )

            with torch.no_grad():
                bsz = hidden_states.shape[0]
                if bsz != 1:
                    raise RuntimeError(
                        "ACG only supports batch_size=1 "
                        "(no padding, one well-defined last row)."
                    )

                # q_norm/k_norm-before-RoPE lives in the adapter; see the module
                # docstring.
                q, _ = adapter.recompute_qk(attn, hidden_states, position_embeddings)
                keys, values = adapter.read_cache(past_key_values, attn.layer_idx)
                key_states = adapter.expand_kv(keys, attn)
                value_f32 = adapter.expand_kv(values, attn).float()
                s_kv = key_states.shape[-2]

                q_last = q[:, :, -1:, :]  # [B, H, 1, D] -- ACG's row i*
                scores = torch.matmul(q_last, key_states.transpose(2, 3)) * attn.scaling

                # The last row is the newest position, so every key column is in its
                # causal past and the mask term is identically zero. sdpa elides the
                # mask on its is_causal fast path and passes None; adding nothing is
                # correct HERE PRECISELY BECAUSE only the last row is ever touched.
                # Anything widening the row scope must first materialise a real causal
                # mask, or every non-final row attends to its future.
                if attention_mask is not None:
                    mask = attention_mask
                    if mask.dtype == torch.bool:
                        # sdpa masks are boolean in 4.57.1 (eager's are additive
                        # floats).
                        mask = torch.where(mask, 0.0, neg_inf(scores.dtype)).to(
                            scores.dtype
                        )
                    scores = scores + mask[:, :, -1:, :s_kv]

                s_cond = (
                    scores.float()
                )  # fp32 from here; exactly one dtype boundary, at the end

                ### ACG's modification
                s_uncond = s_cond.clone()
                s_uncond[:, :, :, img_lo:img_hi] = neg_inf(
                    s_uncond.dtype
                )  # M[i*, j] = -inf for visual j
                p_cond = nn.functional.softmax(s_cond, dim=-1, dtype=torch.float32)
                p_uncond = nn.functional.softmax(s_uncond, dim=-1, dtype=torch.float32)

                n_heads, head_dim = p_cond.shape[1], value_f32.shape[-1]
                o_cond = _flatten_heads(
                    torch.matmul(p_cond, value_f32), bsz
                )  # [1, 1, H*D]
                o_uncond = _flatten_heads(torch.matmul(p_uncond, value_f32), bsz)
                delta_o = o_cond - o_uncond
                delta_perp = orthogonalize(
                    delta_o, o_uncond, self.orth_scope, self.eps, n_heads, head_dim
                )

                # ONE o_proj, and gamma sits OUTSIDE it so gamma == 0 yields exactly
                # +-0.0 rather than a small residue.
                delta = self.gamma * nn.functional.linear(
                    delta_perp, attn.o_proj.weight.float(), None
                )
                ### ACG's modification

                if self.record_stats:
                    self._record(
                        attn.layer_idx,
                        o_cond,
                        o_uncond,
                        delta_o,
                        delta_perp,
                        delta,
                        attn_output,
                        p_uncond,
                        n_heads,
                        head_dim,
                    )

                # Out-of-place: every row but the last must survive byte-for-byte.
                new_output = attn_output.clone()
                new_output[:, -1, :] = attn_output[:, -1, :] + delta[:, 0, :].to(
                    attn_output.dtype
                )

            return new_output, attn_weights

        return wrapper

    def _record(
        self,
        layer_idx,
        o_cond,
        o_uncond,
        delta_o,
        delta_perp,
        delta,
        attn_output,
        p_uncond,
        n_heads,
        head_dim,
    ) -> None:
        """Per-layer diagnostics for the effect smoke test.

        `dO_over_O` near 0 means the image contributes almost nothing to the last row,
        so ACG has nothing to work with -- structurally inert, not broken.
        `perp_over_dO` near 0 means orthogonalization is annihilating the correction;
        near 1 means it does nothing. `uncond_mass_on_image` must be exactly 0.0,
        proving the mask landed. None of these are recoverable from an accuracy column
        after the fact.
        """
        n_dO = float(delta_o.norm())
        if self.orth_scope == "head":
            x = delta_perp.view(1, 1, n_heads, head_dim).transpose(1, 2)
            v = o_uncond.view(1, 1, n_heads, head_dim).transpose(1, 2)
        else:
            x, v = delta_perp, o_uncond
        u = v / (v.norm(dim=-1, keepdim=True) + self.eps)
        residual = float((x * u).sum(dim=-1).abs().max())
        self.stats[layer_idx] = {
            "dO_over_O": n_dO / max(float(o_cond.norm()), 1e-12),
            "perp_over_dO": float(delta_perp.norm()) / max(n_dO, 1e-12),
            "delta_over_out": float(delta.norm())
            / max(float(attn_output[:, -1, :].norm()), 1e-12),
            "orth_residual": residual / max(n_dO, 1e-12),
            "uncond_mass_on_image": float(
                p_uncond[:, :, :, self.img_lo : self.img_hi].sum()
            ),
        }

    def remove(self) -> None:
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()

    def __enter__(self) -> "InternVLACGIntervention":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
