"""One adapter per backbone, holding every fact a hallucination-mitigation port needs
about it.

WHY THIS EXISTS. The seven Qwen2.5-VL ports contain FOUR independent copies of
`find_image_span` with three different behaviours on a text-only prompt and two
different span conventions (three scan for `image_token_id`, SID scans for
`VISION_START/END_TOKEN_ID`). Adding a second backbone by the same method would produce
eight. Each port instead asks an adapter.

SCOPE, deliberately. This is NOT a refactor of the Qwen ports -- they are producing
paper numbers and churning them risks the existing table. `QwenAdapter` exists so the
InternVL code can be written against one interface and so a later consolidation is
cheap; the Qwen ports keep their own copies until someone decides to migrate them on
purpose.

THE PART THAT EARNS ITS KEEP is `recompute_qk`. ACG, PAI and SID all need to reproduce a
layer's attention scores outside the fused kernel, and the exact op sequence differs
between the two backbones in a way that is invisible at review time and silent at
runtime:

    Qwen2.5-VL : q_proj -> view -> transpose
                 -> apply_MULTIMODAL_rotary_pos_emb(mrope_section)
    InternVL3  : q_proj -> view -> q_NORM   -> transpose
                 -> apply_rotary_pos_emb   (plain)

Qwen3 (InternVL3's text tower) applies an RMSNorm over `head_dim` to q and k -- and NOT
to v -- before RoPE. A port that omits it still runs and still produces plausible
logits; it just computes the wrong attention. `test_internvl_recompute_unit.py` pins it
against the real kernel.
"""

from typing import Protocol

import torch
from torch import nn

# NOT `-inf`: a row that ended up fully masked would softmax to NaN and poison
# generation silently. -1e30 underflows to exactly 0.0 after exp() and fails safe. Same
# constant and same reason as `attn_intervention.py` and SID's `Ct2sImageMask`.
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


class BackboneAdapter(Protocol):
    """What a method needs to know about a backbone. Implementations are stateless."""

    def text_layers(self, model) -> nn.ModuleList: ...
    def vision_layers(self, model) -> nn.ModuleList: ...
    def image_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def image_span(self, input_ids: torch.Tensor) -> tuple[int, int] | None: ...
    def recompute_qk(self, attn, hidden_states, position_embeddings): ...
    def expand_kv(self, x: torch.Tensor, attn) -> torch.Tensor: ...
    def read_cache(self, past_key_values, layer_idx): ...


def _span_from_mask(mask: torch.Tensor, who: str) -> tuple[int, int] | None:
    """`[lo, hi)` from a boolean mask, or `None` if empty. Raises if the run is not
    contiguous.

    The boolean mask is the primitive and this is the convenience derived from it.
    Methods that genuinely need an interval (ACG masks a contiguous key slice) call this
    and get a loud failure on a disjoint layout -- InternVL's video prompt is N
    per-frame runs separated by text, and masking only `[first, last)` there would
    blanket the interleaved text as well. Failing beats silently masking the wrong
    columns.
    """
    positions = mask.nonzero().flatten()
    if positions.numel() == 0:
        return None
    lo, hi = int(positions[0]), int(positions[-1]) + 1
    if hi - lo != positions.numel():
        raise ValueError(
            f"{who}: the image-token positions are not contiguous "
            f"({positions.numel()} tokens spread over [{lo},{hi})). This is a "
            f"multi-image or video prompt; a single [lo,hi) slice would also cover "
            f"the text between the runs."
        )
    return lo, hi


class InternVL3Adapter:
    """`transformers.InternVLForConditionalGeneration` with a Qwen3 text tower.

    Verified against transformers 4.57.1 and InternVL3.5-8B-HF:
    36 text layers (hidden 4096, 32 heads / 8 KV, head_dim 128, `attention_bias=False`),
    24 vision layers, `image_token_id=151671`, plain 1-D RoPE with no `rope_deltas`.
    """

    name = "internvl3"

    def text_layers(self, model) -> nn.ModuleList:
        """`model.model.language_model.layers` from outside the HF class, one level in
        from within.

        `InternVLModel._checkpoint_conversion_mapping` maps `language_model.model` ->
        `language_model`, so the decoder list is directly under `language_model`.
        """
        inner = model.model if hasattr(model, "model") else model
        if hasattr(inner, "language_model"):
            return inner.language_model.layers
        return inner.layers

    def vision_layers(self, model) -> nn.ModuleList:
        """`vision_tower.encoder.layer` -- note `.layer`, SINGULAR.

        `InternVLVisionEncoder` names its ModuleList `layer`, not `layers` (Qwen2.5-VL
        uses `visual.blocks`). Truncating it is safe in a way Qwen's was not automatic:
        the encoder's forward is a plain `for layer_module in self.layer`, with no
        index-based branching.
        """
        inner = model.model if hasattr(model, "model") else model
        return inner.vision_tower.encoder.layer

    def image_token_id(self, model) -> int:
        return int(model.config.image_token_id)

    def image_token_mask(
        self, input_ids: torch.Tensor, image_token_id: int = 151671
    ) -> torch.Tensor:
        """Boolean mask over positions holding `<IMG_CONTEXT>`, shape `[L]`.

        This is the same test the model itself uses to place the vision features
        (`InternVLModel.get_placeholder_mask`:
        `input_ids == self.config.image_token_id`), so the masked positions are exactly
        the ones carrying image information -- not the `<img>` / `</img>` sentinels,
        which are text tokens marking the boundary.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"Expected a single unpadded sequence [1, L], got "
                f"{tuple(input_ids.shape)}."
            )
        return input_ids[0] == image_token_id

    def image_span(
        self, input_ids: torch.Tensor, image_token_id: int = 151671
    ) -> tuple[int, int] | None:
        return _span_from_mask(
            self.image_token_mask(input_ids, image_token_id),
            "InternVL3Adapter.image_span",
        )

    def recompute_qk(self, attn, hidden_states: torch.Tensor, position_embeddings):
        """Reproduce a Qwen3 layer's post-RoPE q and k from its INPUT hidden states.

        `hidden_states` must be what the layer's `self_attn` was called with, i.e. POST
        `input_layernorm`, not the residual stream.

        The op order is `Qwen3Attention.forward`'s, and all three details matter:
          * `.view(hidden_shape)` BEFORE the norm, so the RMSNorm reduces over
            `head_dim` (128) per head rather than over the flat 4096.
          * `q_norm`/`k_norm` applied to q and k but NOT to v.
          * plain `apply_rotary_pos_emb` -- there is no mRoPE here and no
            `mrope_section`.
        """
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k

    def expand_kv(self, x: torch.Tensor, attn) -> torch.Tensor:
        """GQA expansion, 8 KV heads -> 32 query heads on the 8B checkpoint."""
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv

        return repeat_kv(x, attn.num_key_value_groups)

    def read_cache(self, past_key_values, layer_idx):
        """`(keys, values)` for one layer, pre-GQA-expansion, shape
        `(B, n_kv_heads, S, head_dim)`.

        transformers 4.57.1 stores K/V on per-layer objects;
        `past_key_values.key_cache[idx]` (the pre-4.56 spelling several upstream method
        releases still use) does NOT exist and raises `AttributeError`.
        """
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values


class QwenAdapter:
    """Qwen2.5-VL, for parity. The shipped ports do not use this yet -- see the module
    docstring."""

    name = "qwen2_5_vl"

    def text_layers(self, model) -> nn.ModuleList:
        inner = model.model if hasattr(model, "model") else model
        if hasattr(inner, "language_model"):
            return inner.language_model.layers
        return inner.layers

    def vision_layers(self, model) -> nn.ModuleList:
        inner = model.model if hasattr(model, "model") else model
        return inner.visual.blocks

    def image_token_id(self, model) -> int:
        return int(model.config.image_token_id)

    def image_token_mask(
        self, input_ids: torch.Tensor, image_token_id: int = 151655
    ) -> torch.Tensor:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"Expected a single unpadded sequence [1, L], got "
                f"{tuple(input_ids.shape)}."
            )
        return input_ids[0] == image_token_id

    def image_span(
        self, input_ids: torch.Tensor, image_token_id: int = 151655
    ) -> tuple[int, int] | None:
        return _span_from_mask(
            self.image_token_mask(input_ids, image_token_id), "QwenAdapter.image_span"
        )

    def recompute_qk(self, attn, hidden_states: torch.Tensor, position_embeddings):
        """Qwen2.5-VL's preamble: no q/k norm, and mRoPE rather than plain RoPE.

        WHERE mRoPE is applied moved between transformers 4.x and 5.x, so this delegates
        to `lmms_eval.models.model_utils.rope_compat.apply_qwen_rope`, which is the
        single place that knows the difference. See that module for the full
        explanation; the short version is that a wrong section split does NOT raise, it
        silently produces wrong scores, so it lives in one tested place rather than
        being duplicated per call site.
        """
        from vlmcfg.baselines.common.rope_compat import apply_qwen_rope

        bsz, q_len, _ = hidden_states.shape
        q = (
            attn.q_proj(hidden_states)
            .view(bsz, q_len, -1, attn.head_dim)
            .transpose(1, 2)
        )
        k = (
            attn.k_proj(hidden_states)
            .view(bsz, q_len, -1, attn.head_dim)
            .transpose(1, 2)
        )
        cos, sin = position_embeddings
        return apply_qwen_rope(attn, q, k, cos, sin)

    def expand_kv(self, x: torch.Tensor, attn) -> torch.Tensor:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

        return repeat_kv(x, attn.num_key_value_groups)

    def read_cache(self, past_key_values, layer_idx):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values


class Qwen3VLAdapter:
    """Qwen3-VL (4B / 8B dense).

    Structurally a hybrid of the two backbones already supported, and BOTH halves
    matter:

      * Like InternVL3/Qwen3, it has `q_norm`/`k_norm` -- RMSNorm over `head_dim`,
        applied after `.view(hidden_shape)` and before RoPE
        (`Qwen3VLTextAttention.forward`, modeling_qwen3_vl.py:428-429). `QwenAdapter`
        has no such norm, so reusing it here would compute q@k.T from UN-normalised q/k:
        that still runs, still produces plausible logits, and is wrong. This is the
        single most likely silent bug in an ACG/PAI/SID port onto this backbone.

      * Unlike InternVL3, its RoPE is mRoPE -- but `apply_rotary_pos_emb` here is the
        PLAIN two-argument form (modeling_qwen3_vl.py:358). The `mrope_section` split is
        already folded into `cos`/`sin` by `Qwen3VLRotaryEmbedding` (config carries
        `rope_scaling.mrope_interleaved=True`, `mrope_section=[24,20,20]`), so no
        `apply_multimodal_rotary_pos_emb` call belongs here. Passing a section list to
        this function would be a TypeError; recomputing the interleave by hand would
        double-apply it.

    DEEPSTACK is invisible to this adapter but not to the methods built on it. The
    vision tower emits features at `config.vision_config.deepstack_visual_indexes`
    ([8,16,24] on 8B) from INSIDE its block loop (modeling_qwen3_vl.py:737-749), and the
    text tower re-injects them at the matching text layers (:862-866). Two consequences
    worth stating once here:
      * Truncating `vision_layers()` below index 8 (LayerCD) yields an EMPTY deepstack
        list, so that arm loses deepstack injection as well as depth. It does not crash
        -- the text side guards with `layer_idx in range(len(deepstack_visual_embeds))`
        -- but it is a stronger degradation than the method intends, and must be
        recorded as a deviation.
      * `text_layers()` returns the plain decoder list; deepstack merging happens
        outside it.
    """

    name = "qwen3_vl"

    def text_layers(self, model) -> nn.ModuleList:
        inner = model.model if hasattr(model, "model") else model
        if hasattr(inner, "language_model"):
            return inner.language_model.layers
        return inner.layers

    def vision_layers(self, model) -> nn.ModuleList:
        inner = model.model if hasattr(model, "model") else model
        return inner.visual.blocks

    def image_token_id(self, model) -> int:
        return int(model.config.image_token_id)

    def image_token_mask(
        self, input_ids: torch.Tensor, image_token_id: int = 151655
    ) -> torch.Tensor:
        """Qwen3-VL keeps Qwen2.5-VL's `image_token_id` (151655), verified against the
        8B config."""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"Expected a single unpadded sequence [1, L], got "
                f"{tuple(input_ids.shape)}."
            )
        return input_ids[0] == image_token_id

    def image_span(
        self, input_ids: torch.Tensor, image_token_id: int = 151655
    ) -> tuple[int, int] | None:
        return _span_from_mask(
            self.image_token_mask(input_ids, image_token_id),
            "Qwen3VLAdapter.image_span",
        )

    def recompute_qk(self, attn, hidden_states: torch.Tensor, position_embeddings):
        """Reproduce a Qwen3-VL text layer's post-RoPE q and k from its INPUT hidden
        states.

        `hidden_states` must be what `self_attn` was called with (post
        `input_layernorm`), not the residual stream. Op order is
        `Qwen3VLTextAttention.forward`'s verbatim: view -> q_norm/k_norm -> transpose ->
        plain rotary.
        """
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k

    def expand_kv(self, x: torch.Tensor, attn) -> torch.Tensor:
        """GQA expansion: 8 KV heads -> 32 query heads on the 8B checkpoint
        (groups = 4)."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

        return repeat_kv(x, attn.num_key_value_groups)

    def read_cache(self, past_key_values, layer_idx):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values


class Molmo2Adapter:
    """allenai/Molmo2-4B (remote code; Qwen3-4B text tower, SigLIP vision).

    Three facts no other adapter here shares, each of which a port gets silently wrong:

      * FUSED qkv. `self_attn.att_proj` emits q|k|v concatenated (`attn.fused_dims`),
        and the output projection is `attn_out`, not `o_proj`. Code written against
        `q_proj`/`o_proj` raises at best, and at worst picks up a same-named attribute
        from elsewhere.
      * qk-norm placement follows `attn.qk_norm_type`: "qwen3" (the 4B checkpoint) norms
        each head AFTER the view, any other type norms the flat projection BEFORE it.
        Mirrored exactly from `Molmo2Attention.forward`.
      * The image block is NOT contiguous in image-PATCH tokens. Each crop row ends in a
        column token (`image_col_id`), and start / low-res-start / end markers sit
        between the views, all marked as image tokens by `token_type_ids`. Only the
        PATCH positions (`config.image_patch_id`, 151938) carry vision features
        (`build_input_embeddings` adds them at exactly those ids), so `image_token_mask`
        returns those -- and `image_span`, which requires contiguity, raises on a real
        prompt by design.

    Plain 1-D RoPE (`position_ids = cache_position`), no rope_deltas, decoder layers at
    `model.model.transformer.blocks`.
    """

    name = "molmo2"

    def text_layers(self, model) -> nn.ModuleList:
        inner = model.model if hasattr(model, "model") else model
        return inner.transformer.blocks

    def vision_layers(self, model) -> nn.ModuleList:
        inner = model.model if hasattr(model, "model") else model
        return inner.vision_backbone.image_vit.transformer.resblocks

    def image_token_id(self, model) -> int:
        return int(model.config.image_patch_id)

    def image_token_mask(
        self, input_ids: torch.Tensor, image_token_id: int = 151938
    ) -> torch.Tensor:
        """Boolean mask over the image-PATCH positions, shape `[L]` (not contiguous)."""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"Expected a single unpadded sequence [1, L], got "
                f"{tuple(input_ids.shape)}."
            )
        return input_ids[0] == image_token_id

    def image_span(
        self, input_ids: torch.Tensor, image_token_id: int = 151938
    ) -> tuple[int, int] | None:
        return _span_from_mask(
            self.image_token_mask(input_ids, image_token_id), "Molmo2Adapter.image_span"
        )

    def recompute_qk(self, attn, hidden_states: torch.Tensor, position_embeddings):
        """Post-RoPE q and k from the layer's INPUT (post `attn_norm`), as
        `Molmo2Attention.forward`."""
        import sys

        rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb
        hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
        q, k, _ = attn.att_proj(hidden_states).split(attn.fused_dims, dim=-1)
        normed = attn.q_norm is not None and attn.k_norm is not None
        if normed and attn.qk_norm_type != "qwen3":
            q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = q.view(hidden_shape), k.view(hidden_shape)
        if normed and attn.qk_norm_type == "qwen3":
            q, k = attn.q_norm(q), attn.k_norm(k)
        cos, sin = position_embeddings
        return rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)

    def expand_kv(self, x: torch.Tensor, attn) -> torch.Tensor:
        """GQA expansion, 8 KV heads -> 32 query heads on the 4B checkpoint."""
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv

        return repeat_kv(x, attn.num_key_value_groups)

    def read_cache(self, past_key_values, layer_idx):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
