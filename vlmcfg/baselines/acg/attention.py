"""ACG's attention-space contrastive guidance on Qwen2.5-VL, spliced as an additive
correction.

THE OPERATOR IS ACG'S, UNCHANGED (Attention-space Contrastive Guidance, Jo/Bae/Kim,
arXiv:2601.13707, Algorithm 1). For each decoder layer in ``[start_layer, end_layer)``,
on the LAST text token's query row ``i*`` only::

    A_cond   = softmax(S)                    O_cond   = A_cond   @ V
    M[i*, j] = -inf if key j is visual
    A_uncond = softmax(S + M)                O_uncond = A_uncond @ V
    dO       = O_cond - O_uncond
    u        = O_uncond / (||O_uncond||_2 + eps)
    dO_perp  = dO - <dO, u> u                # textual orthogonalization
    O_final  = O_cond + gamma * dO_perp
    Z        = O_final W_O

Only row ``i*`` of ``M`` is non-zero, so ``O_final`` differs from ``O_cond`` in that row
and no other -- which is what makes a last-row-only recompute *exact* rather than an
approximation.

WHY THE SPLICE IS ADDITIVE, AND WHY THAT MATTERS. ``Qwen2_5_VLAttention.o_proj`` is
``nn.Linear(..., bias=False)`` (verified against transformers 4.57.1), so ``W_O`` is
linear and

    Z_final = O_final W_O = O_cond W_O + gamma * (dO_perp W_O)

and ``O_cond W_O`` is EXACTLY what the real sdpa kernel already returned. So this module
never replaces the attention output; it adds one term to its last row::

    out[-1] += gamma * (dO_perp W_O)

Two consequences. (1) One ``o_proj`` per layer instead of PAI's two -- ACG is genuinely
cheaper than the delta form it is modelled on. (2) At ``gamma = 0`` the added term is
``0.0 * finite == +-0.0``, the cast preserves it, and ``x + +-0.0 == x`` for every float
-- so the intervention can run EVERY line (both softmaxes, the mask, the
orthogonalization, the projection) and still be bit-identical to an unintervened run.
That is what makes the neutral point a statement about the arithmetic rather than about
a branch. See the module docstring of ``qwen_acg_model.py``.

REJECTED ALTERNATIVE: recomputing the whole last row as ``O_final W_O`` and overwriting
it. That injects recompute-vs-fused-kernel noise (~1e-3) on every layer even at
``gamma = 0``, 28 layers deep, which this repo has already measured to flip greedy
argmax (``vlmcfg/negatives/attention.py``). The additive form has no such term.

THE PLUMBING IS PAI'S. ``vlmcfg.baselines.pai.attention.PAIIntervention`` already solved
the Qwen2.5-VL specifics -- mRoPE via ``apply_multimodal_rotary_pos_emb`` with
``rope_scaling["mrope_section"]``, GQA via ``repeat_kv`` (28 query heads over 4 KV
heads), ``attn.scaling``, and reading the already-RoPE'd already-accumulated K/V
straight back out of the ``Cache`` the real forward just wrote. All of that is
reproduced here. Crucially, this module NEVER sets ``attn_implementation="eager"``: an
all-eager text tower diverges from sdpa by as much as ``max|dlogit| ~ 1.22`` on a fixed
prompt, far more than ACG itself does, so an eager arm would be measuring the kernel
swap rather than the method.

Scope (asserted, not assumed): ``batch_size == 1``, an actual ``Cache`` in
``past_key_values``, and ``full_attention`` layers.
"""

from collections.abc import Callable

import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

# `apply_multimodal_rotary_pos_emb` was REMOVED in transformers 5.x (mRoPE moved into
# Qwen2_5_VLRotaryEmbedding). Go through the compat shim so this file works in both.
from vlmcfg.baselines.common.rope_compat import apply_qwen_rope

# `<|image_pad|>` for every Qwen2.5-VL checkpoint. Only the DEFAULT for
# `find_image_span`; callers holding a config should pass `config.image_token_id`.
IMAGE_TOKEN_ID = 151655

# NOT `-inf`. A row that ended up fully masked would softmax to NaN and poison the whole
# generation silently; -1e30 underflows to exactly 0.0 after exp() and fails safe. Same
# constant, same reason, as `attn_intervention.py` and SID's `Ct2sImageMask`.
NEG_INF = -1e30


def neg_inf(dtype: "torch.dtype") -> float:
    """Additive-mask fill value that is representable in `dtype`, and safe to add twice.

    `-1e30` is below fp16's floor of -65504, so in fp16 it either raises outright
    (`masked_fill`: "value cannot be converted to type at::Half without overflow") or,
    worse, silently becomes `-inf` when it arrives via `torch.where(...).to(half)`. A
    row that ends up fully `-inf` softmaxes to NaN and poisons generation without an
    error -- exactly the failure `-1e30` was chosen to avoid in the first place.

    `finfo(dtype).min / 2` keeps the original guarantee in every dtype: it still
    underflows to exactly 0.0 after exp(), and the halving leaves room for the one `+=`
    accumulation these masks perform (kill-columns added on top of a causal mask) to
    stay finite.
    """
    return torch.finfo(dtype).min / 2


ORTH_SCOPES = ("concat", "head", "none")


def get_language_model_layers(model) -> nn.ModuleList:
    """The decoder-layer list, at ``model.model.language_model.layers``.

    On transformers >= 4.52 the Qwen2.5-VL text stack lives one level deeper than the
    old ``model.model.layers``; the flat path is kept as a fallback. Copied from PAI's
    port so both methods resolve the same 28 modules.
    """
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers
    return inner.layers


def find_image_span(
    input_ids: torch.Tensor, image_token_id: int = IMAGE_TOKEN_ID
) -> tuple[int, int] | None:
    """``(img_lo, img_hi)``: the half-open ``<|image_pad|>`` span, or ``None`` if there
    is none.

    The VISUAL keys are the image-pad run itself -- not the ``<|vision_start|>`` /
    ``<|vision_end|>`` sentinels, which are text tokens marking its boundaries. Masking
    the sentinels too would remove text the paper's ``M`` leaves alone.

    Returns ``None`` rather than raising on a text-only prompt: ACG is undefined there
    (there is no visual key to mask, so ``O_uncond == O_cond`` and ``dO == 0``), and the
    caller skips the intervention entirely. Non-contiguous or multi-image spans DO raise
    -- masking only the first of two images would leave the "unconditional" path still
    looking at the second, which is silently wrong rather than loudly broken.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"ACG expects a single unpadded sequence [1, L], got "
            f"{tuple(input_ids.shape)}."
        )
    positions = (input_ids[0] == image_token_id).nonzero().flatten()
    if positions.numel() == 0:
        return None
    img_lo = int(positions[0])
    img_hi = int(positions[-1]) + 1
    if img_hi - img_lo != positions.numel():
        raise ValueError(
            "ACG found a non-contiguous image-token span (multi-image prompt?). It "
            "masks one [img_lo, img_hi) slice; with two images the 'unconditional' "
            "path would still attend to the second one."
        )
    return img_lo, img_hi


def orthogonalize(
    delta_o: torch.Tensor,
    o_uncond: torch.Tensor,
    scope: str,
    eps: float,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """ACG Eq. 6-7: remove from ``delta_o`` the component along the text-only direction.

    ``scope`` selects what "the" direction means, which the paper leaves implicit:

      * ``"concat"`` (DEFAULT, and the paper's reading) -- one direction over the whole
        ``[1, 1, num_heads*head_dim]`` vector. Algorithm 1 applies ``W_O`` to the same
        symbol it takes ``||.||_2`` and ``<dO, u>`` of, and in standard multi-head
        attention the object ``W_O`` left-multiplies IS the concatenation of the heads.
        So the norm is over all ``hidden`` dims and the inner product is a single
        scalar.
      * ``"head"`` -- one direction per head (28 norms, 28 scalars). This removes
        strictly more and is a DIFFERENT operator, not a formatting choice. Ours, not
        the paper's; kept as an ablation and labelled a deviation in the README.
      * ``"none"`` -- return ``delta_o`` unchanged, reproducing the paper's own Table 5
        ablation ("ACG w/o Ortho").

    DO NOT move this after ``o_proj``. ``W_O`` is not orthogonal, so
    ``(dO - <dO,u>u) W_O != (dO W_O) - <dO W_O, u W_O>(u W_O)``; projecting in the
    3584-dim residual space silently implements a different method.
    """
    if scope == "none":
        return delta_o
    if scope == "concat":
        x, v = delta_o, o_uncond
    elif scope == "head":
        x = delta_o.view(1, 1, num_heads, head_dim).transpose(1, 2)  # [1, H, 1, D]
        v = o_uncond.view(1, 1, num_heads, head_dim).transpose(1, 2)
    else:
        raise ValueError(f"acg_orth_scope must be one of {ORTH_SCOPES}, got {scope!r}.")
    # `eps` is the paper's, and guards the degenerate ||O_uncond|| == 0 case into u == 0
    # (which makes the projection a no-op) instead of a NaN.
    u = v / (v.norm(dim=-1, keepdim=True) + eps)
    out = x - (x * u).sum(dim=-1, keepdim=True) * u
    if scope == "head":
        out = out.transpose(1, 2).reshape(1, 1, num_heads * head_dim)
    return out


class ACGIntervention:
    """ACG's per-layer last-row correction, installed as an additive splice.

    ``install()`` / ``remove()`` / context-manager shape, wrapping ``self_attn.forward``
    rather than replacing it -- a forward *hook* cannot be used because
    ``query_states``/``key_states`` are locals of ``Qwen2_5_VLAttention.forward`` (see
    ``lastrow_attn.py``).

    ``record_stats`` turns on the per-layer diagnostics the effect smoke test reports
    (``||dO||/||O_c||``, ``||dO_perp||/||dO||``, ``||delta||/||out[-1]||``). It is off
    during sweeps: these are the numbers that distinguish "ACG is structurally inert on
    this benchmark" from "ACG is broken", and both look like a flat accuracy column.
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
        self.layer_lo = layer_lo
        self.layer_hi = layer_hi
        self.img_lo = img_lo
        self.img_hi = img_hi
        self.gamma = float(gamma)
        self.orth_scope = orth_scope
        self.eps = float(eps)
        self.record_stats = record_stats
        self.stats: dict[int, dict[str, float]] = {}
        self._patched: dict[nn.Module, Callable] = {}

    def install(self, layers: nn.ModuleList) -> None:
        for layer in layers:
            attn = layer.self_attn
            layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None or not (self.layer_lo <= layer_idx < self.layer_hi):
                continue
            layer_types = getattr(attn.config, "layer_types", None)
            if layer_types is not None and layer_types[layer_idx] != "full_attention":
                raise ValueError(
                    f"ACGIntervention only supports full_attention layers; layer "
                    f"{layer_idx} is {layer_types[layer_idx]!r} (sliding-window "
                    f"checkpoint?). The recompute reads the whole key history back out "
                    f"of the cache, which a windowed layer may not hold."
                )
            if attn in self._patched:
                continue  # never stack: install() is called once per forward
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def _make_wrapper(self, attn: nn.Module, original: Callable):
        img_lo, img_hi = self.img_lo, self.img_hi

        def wrapper(hidden_states, *args, **kwargs):
            # The REAL forward first, untouched, on the model's real kernel: every row,
            # every head, and this layer's one and only cache write. Only the last row
            # is corrected.
            attn_output, attn_weights = original(hidden_states, *args, **kwargs)

            attention_mask = kwargs.get("attention_mask")
            position_embeddings = kwargs.get("position_embeddings")
            past_key_values = kwargs.get("past_key_values")
            if args:
                # Qwen2_5_VLDecoderLayer always calls with keyword args; guarded for
                # safety.
                attention_mask = args[0] if attention_mask is None else attention_mask
            if past_key_values is None:
                raise RuntimeError(
                    "ACGIntervention requires use_cache=True with a real Cache "
                    "(past_key_values); it reads the accumulated K/V back out of it."
                )
            if position_embeddings is None:
                raise RuntimeError(
                    "ACGIntervention requires position_embeddings "
                    "(mRoPE cos/sin) to be passed."
                )

            with torch.no_grad():
                bsz, q_len, _ = hidden_states.size()
                if bsz != 1:
                    raise RuntimeError(
                        "ACGIntervention only supports batch_size=1 "
                        "(no padding, one well-defined last row)."
                    )

                # Project + RoPE this step's queries exactly as the real forward did.
                # `k_dummy` is only a same-op RoPE argument; its result is discarded,
                # because the real, already-accumulated K comes from the cache below.
                q = (
                    attn.q_proj(hidden_states)
                    .view(bsz, q_len, -1, attn.head_dim)
                    .transpose(1, 2)
                )
                k_dummy = (
                    attn.k_proj(hidden_states)
                    .view(bsz, q_len, -1, attn.head_dim)
                    .transpose(1, 2)
                )
                cos, sin = position_embeddings
                q, _ = apply_qwen_rope(attn, q, k_dummy, cos, sin)

                cached = past_key_values.layers[attn.layer_idx]
                key_states = repeat_kv(cached.keys, attn.num_key_value_groups)
                value_f32 = repeat_kv(cached.values, attn.num_key_value_groups).float()
                s_kv = key_states.shape[-2]

                q_last = q[:, :, -1:, :]  # [B, H, 1, D] -- ACG's row i*
                scores = torch.matmul(q_last, key_states.transpose(2, 3)) * attn.scaling

                # The last row is the newest position, so every key column 0..s_kv-1 is
                # in its causal past and the mask term is identically zero. sdpa elides
                # the mask on its is_causal fast path and passes None; adding nothing is
                # correct HERE PRECISELY BECAUSE only the last row is ever touched.
                # Anything that widens the row scope must first port
                # attn_intervention.py's causal-mask reconstruction, or every non-final
                # row will attend to its own future.
                if attention_mask is not None:
                    scores = scores + attention_mask[:, :, -1:, :s_kv]

                # fp32 from here on, so there is exactly one dtype boundary: the final
                # cast.
                s_cond = scores.float()

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

                # ONE o_proj (PAI's delta form needs two), and the gamma factor sits
                # outside it so gamma == 0 yields exactly +-0.0 rather than a small
                # residue.
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
        """Per-layer diagnostics, for the effect smoke test and for unit tests A3/A4/A5.

        ``dO_over_O`` near 0 means the image contributes almost nothing to the last
        row's attention output, so ACG has nothing to work with on this input --
        structurally inert, not broken. ``perp_over_dO`` near 0 means orthogonalization
        is annihilating the correction; near 1 means it is doing nothing and
        concat-vs-head is moot. ``orth_residual`` is the relative component of
        ``dO_perp`` still lying along the text-only direction, i.e. a direct check of
        Eq. 7 -- it must be ~0 for any scope but ``"none"``. ``uncond_mass_on_image``
        must be exactly 0.0, proving the mask actually landed. None of these are
        recoverable from an accuracy column after the fact.
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

    def __enter__(self) -> "ACGIntervention":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def _flatten_heads(x: torch.Tensor, bsz: int) -> torch.Tensor:
    """``[B, H, R, D] -> [B, R, H*D]``: the concatenation ``W_O`` left-multiplies."""
    return x.transpose(1, 2).contiguous().reshape(bsz, x.shape[2], -1)
