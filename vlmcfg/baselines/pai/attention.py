"""PAI's inference intervention on Qwen2.5-VL, spliced as a delta.

THE OPERATOR IS PAI'S, UNCHANGED. For decoder layers ``[start_layer, end_layer)``, the
LAST query row's raw (pre-softmax) attention scores at the image-token columns are
amplified by the paper's rule, and the bias is skipped while the CFG branch runs its
text-only forward (PAI's ``use_attn and not use_cfg`` guard):

    A[:, :, -1, img_start:img_end] += alpha * |A[:, :, -1, img_start:img_end]|

THE PLUMBING IS NOT PAI'S, DELIBERATELY. ``PAI-master/attention.py`` installs a
hand-written EAGER forward over the whole layer. Doing that here would make the PAI arm
non-comparable to this repo's baselines, which run on sdpa/flash: an all-eager text
tower diverges from the fused kernel by as much as ``max|dlogit| ~ 1.22`` on a fixed
prompt (measured with `vlmcfg/baselines/common/lastrow_attn.py`), which is far larger
than anything PAI itself does at alpha=0.2 -- so an eager-vs-sdpa arm would be measuring
the kernel swap, not the method.

Instead this module uses the delta form from
``vlmcfg/negatives/attention.py``:

  * the layer's REAL forward runs first, untouched, on the model's real kernel -- every
    row, every layer, including the one this module then corrects;
  * only the last row is recomputed, twice, through the IDENTICAL code path
    (``_attend_and_project``): once with PAI's bias and once without;
  * their DIFFERENCE is added to the real kernel's output.

        delta    = o_proj(softmax(A + bias) @ V) - o_proj(softmax(A) @ V)
        out[-1] += delta

Because both branches are the same ops on the same tensors, the
recompute-vs-fused-kernel noise (different matmul accumulation order, materialising
``repeat_kv``, dtype rounding) appears in both with the same sign and cancels. With
``alpha = 0`` the bias is zero, the two branches are literally identical, ``delta`` is
BITWISE zero, and the output is bit-identical to an unintervened run -- so ``alpha=0``
is an exact control arm rather than "close enough". Everything downstream of the scores
runs in fp32, leaving exactly one dtype boundary: the final cast of ``delta``.

Qwen2.5-VL specifics handled here (none of which PAI's LLaMA code had): mRoPE via
``apply_multimodal_rotary_pos_emb`` with ``rope_scaling["mrope_section"]``, GQA via
``repeat_kv`` (28 query heads over 4 KV heads), ``attn.scaling`` for
``1/sqrt(head_dim)``, and reading the real (already RoPE'd, already accumulated) K/V
straight back out of the ``Cache`` the real forward just wrote.

Scope (asserted, not assumed): ``batch_size == 1`` (no padding, one well-defined last
row) and an actual ``Cache`` in ``past_key_values`` (this targets ``generate()``'s
incremental decoding path).
"""

from collections.abc import Callable

import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

# `apply_multimodal_rotary_pos_emb` was REMOVED in transformers 5.x (mRoPE moved into
# Qwen2_5_VLRotaryEmbedding). Go through the compat shim so this file works in both.
from vlmcfg.baselines.common.rope_compat import apply_qwen_rope

# `<|image_pad|>` for every Qwen2.5-VL checkpoint. Only the DEFAULT for
# `find_image_span`; callers that hold a config should pass `config.image_token_id`.
IMAGE_TOKEN_ID = 151655


def get_language_model_layers(model) -> nn.ModuleList:
    """The decoder-layer list PAI indexes as ``model.model.layers``.

    On transformers >= 4.52 the Qwen2.5-VL text stack lives one level deeper, under
    ``model.model.language_model``; the older flat path is kept as a fallback.
    """
    inner = model.model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers
    return inner.layers


def find_image_span(
    input_ids: torch.Tensor, image_token_id: int = IMAGE_TOKEN_ID
) -> tuple[int, int]:
    """``(img_start_idx, img_end_idx)``: the half-open ``<|image_pad|>`` span.

    LLaVA has a fixed 576 image tokens, so PAI computes the span arithmetically.
    Qwen2.5-VL emits ``(H/28)*(W/28)`` tokens after the 2x2 spatial merge, so it has to
    be read off the processed ``input_ids``.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"Expected a single unpadded sequence [1, L], got {tuple(input_ids.shape)}."
        )
    positions = (input_ids[0] == image_token_id).nonzero().flatten()
    if positions.numel() == 0:
        raise ValueError("No image tokens found in the prompt.")
    img_start_idx = int(positions[0])
    img_end_idx = int(positions[-1]) + 1
    if img_end_idx - img_start_idx != positions.numel():
        raise ValueError(
            "Image tokens are not contiguous; PAI perturbs one "
            "[img_start, img_end) slice."
        )
    return img_start_idx, img_end_idx


def _attend_and_project(
    attn: nn.Module, probs: torch.Tensor, value_f32: torch.Tensor, bsz: int
) -> torch.Tensor:
    """``o_proj(probs @ V)`` for the query row, entirely in fp32.

    Factored out so the biased and plain branches go through provably IDENTICAL ops --
    that identity is what makes the recompute noise cancel and a zero bias a bitwise
    no-op.
    """
    out = torch.matmul(probs, value_f32)  # [B, H, R, D]
    out = (
        out.transpose(1, 2).contiguous().reshape(bsz, probs.shape[2], -1)
    )  # [B, R, H*D]
    # o_proj is bf16; run it in fp32 so the two branches share one dtype boundary
    # (the final cast) instead of two.
    return nn.functional.linear(out, attn.o_proj.weight.float(), None)


class PAIIntervention:
    """PAI's last-row image-attention amplification, installed as a delta splice.

    ``install()``/``remove()``/context-manager shape. Unlike ``PAI-master``'s
    ``llama_modify``, this WRAPS ``self_attn.forward`` rather than replacing it: the
    real forward runs first, on the real kernel, and only the last row's output is
    corrected.

    ``use_cfg`` is PAI's own switch: ``CFGLogits`` raises it around the unconditional
    (text-only) forward, where the ``[img_start, img_end)`` columns no longer hold image
    tokens and boosting them would corrupt the negative branch. PAI sets it per
    attention module; here it is one flag on the handle, which is the same thing with
    one owner.
    """

    def __init__(
        self, layer_lo: int, layer_hi: int, img_lo: int, img_hi: int, alpha: float
    ) -> None:
        if not layer_lo < layer_hi:
            raise ValueError(f"Empty layer range [{layer_lo}, {layer_hi}).")
        if not 0 <= img_lo < img_hi:
            raise ValueError(f"Invalid image span [{img_lo}, {img_hi}).")
        self.layer_lo = layer_lo
        self.layer_hi = layer_hi
        self.img_lo = img_lo
        self.img_hi = img_hi
        self.alpha = alpha
        self.use_cfg = False
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
                    "PAIIntervention only supports full_attention layers; "
                    f"layer {layer_idx} is "
                    f"{layer_types[layer_idx]!r} (sliding-window checkpoint?)."
                )
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def _make_wrapper(self, attn: nn.Module, original: Callable):
        img_lo, img_hi = self.img_lo, self.img_hi

        def wrapper(hidden_states, *args, **kwargs):
            # Real forward first, untouched: real kernel, every row, this layer's one
            # and only cache write. Only the last row's output is corrected below.
            attn_output, attn_weights = original(hidden_states, *args, **kwargs)

            # PAI's guard: no bias while the CFG branch runs its text-only forward.
            if self.use_cfg:
                return attn_output, attn_weights

            attention_mask = kwargs.get("attention_mask")
            position_embeddings = kwargs.get("position_embeddings")
            past_key_values = kwargs.get("past_key_values")
            if args:
                # Qwen2_5_VLDecoderLayer always calls with keyword args; guarded for
                # safety.
                attention_mask = args[0] if attention_mask is None else attention_mask
            if past_key_values is None:
                raise RuntimeError(
                    "PAIIntervention requires use_cache=True with a real Cache "
                    "(past_key_values); it targets generate()'s "
                    "incremental-decoding path."
                )
            if position_embeddings is None:
                raise RuntimeError(
                    "PAIIntervention requires position_embeddings "
                    "(mRoPE cos/sin) to be passed."
                )

            with torch.no_grad():
                bsz, q_len, _ = hidden_states.size()
                assert bsz == 1, (
                    "PAIIntervention only supports batch_size=1 (no padding)."
                )

                # Project + RoPE this step's queries exactly like the real forward did.
                # `k_dummy` is only a same-op RoPE argument; its result is discarded --
                # the real, already-accumulated K comes from the cache below.
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

                # The real forward already wrote this step's K/V into the cache (RoPE'd,
                # accumulated with everything before it), so read it back rather than
                # re-deriving it.
                cached = past_key_values.layers[attn.layer_idx]
                key_states = repeat_kv(cached.keys, attn.num_key_value_groups)
                value_f32 = repeat_kv(cached.values, attn.num_key_value_groups).float()
                s_kv = key_states.shape[-2]

                q_last = q[:, :, -1:, :]  # [B, H, 1, D]
                scores = torch.matmul(q_last, key_states.transpose(2, 3)) * attn.scaling

                # The LAST row is the newest position, so every key column 0..s_kv-1 is
                # in its causal past and the mask term is all zeros. Added anyway when
                # HF supplies a mask (it is the same tensor the real forward used); when
                # sdpa takes its is_causal fast path and passes None, there is nothing
                # to add -- correct here precisely because only the last row is ever
                # touched.
                if attention_mask is not None:
                    scores = scores + attention_mask[:, :, -1:, :s_kv]

                # fp32 from here on: the bias, both softmaxes, both p@V and both o_proj,
                # so the ONLY dtype boundary is the final cast of the delta.
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

                # THE DELTA FORM: splice the DIFFERENCE of two identical-path
                # recomputes, not one recompute's absolute value. The
                # fused-kernel-vs-manual noise cancels, so alpha=0 is bitwise zero
                # rather than ~1e-3 of drift that greedy argmax would eventually
                # discretise into a different continuation.
                out_plain = _attend_and_project(attn, p_plain, value_f32, bsz)
                out_biased = _attend_and_project(attn, p_biased, value_f32, bsz)
                delta = out_biased - out_plain  # [B, 1, hidden_size], fp32

                # Out-of-place: every row but the last must survive byte-for-byte.
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

    def __enter__(self) -> "PAIIntervention":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def qwen_modify(
    model, start_layer, end_layer, use_attn, alpha, use_cfg, img_start_idx, img_end_idx
):
    """PAI's ``llama_modify``, for Qwen2.5-VL. Returns the handle (needed to uninstall).

    Same call shape as the paper's, so the ported eval scripts read the same. Unlike the
    original -- which rebinds a whole eager forward and is therefore idempotent -- this
    one WRAPS the forward, so any handle already installed on this model is removed
    first; otherwise re-calling per sample (as the eval scripts do) would stack
    wrappers.

    ``use_attn=False`` uninstalls and returns ``None``, matching the original's
    behaviour of leaving the bias switched off.
    """
    previous = getattr(model, "_pai_intervention", None)
    if previous is not None:
        previous.remove()
        model._pai_intervention = None

    if not use_attn:
        return None

    handle = PAIIntervention(start_layer, end_layer, img_start_idx, img_end_idx, alpha)
    handle.use_cfg = use_cfg
    handle.install(get_language_model_layers(model))
    model._pai_intervention = handle
    return handle
