"""InternVL3 with SID (Self-Introspective Decoding, Huo et al.) -- CT2S + contrastive
decoding.

Two nested seams, as on Qwen2.5-VL: (1) capture layer K-1's head-averaged last attention
row over the image tokens, (2) from layer K onward, mask out every image column EXCEPT
the least-attended `rank` of them, building a deliberately hallucination-amplified
"amateur" stream, then contrast against it with the shared `contrastive_combine`.

WHY THE PER-LAYER WRAPPER SURVIVES THE PORT. transformers 4.57.1 offers a much cleaner
mask hook than the hand-rebuilt mask this needs -- `Qwen3Model.forward` accepts a dict
`causal_mask_mapping` and short-circuits all mask creation. It is NOT usable here: that
dict applies to EVERY layer, whereas CT2S must mask only from layer K onward. Layers < K
have to see the whole image, because that is what makes layer K-1's attention row
meaningful in the first place. So the mask is still injected per layer, and still has to
be built by hand -- on the sdpa path with an unpadded prompt `create_causal_mask` elides
the mask entirely and returns `None`, letting `F.scaled_dot_product_attention`'s own
`is_causal=True` supply causality.

ONE NEW WRINKLE ON THIS BACKBONE: when transformers DOES materialise an sdpa mask it is
now a BOOLEAN tensor (eager's is an additive float). A boolean mask cannot carry
`-1e30`, so it is converted to the additive form before the image columns are
suppressed. Getting this wrong would not raise -- it would silently mask nothing,
because `bool + (-1e30)` is a type promotion, not a suppression.

THE RANK DOES NOT TRANSFER, and that is worth saying plainly. `sid_attention_rank=100`
was tuned on LLaVA-1.5's fixed 576 image tokens. InternVL emits 256 tokens per tile and
1-13 tiles, so 100 is anywhere from "almost all of them" to "under a tenth".
`sid_attention_rank_ratio` expresses the same intent proportionally and is the better
arm here; the absolute default is kept only because it is the paper's.
"""

import torch
from torch import nn
from transformers import InternVLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.common.backbone import InternVL3Adapter, neg_inf
from vlmcfg.baselines.common.rope_compat import model_forward_kwargs
from vlmcfg.baselines.vcd.combine import contrastive_combine

_ADAPTER = InternVL3Adapter()


class InternVLLastRowCapture:
    """Capture one layer's head-averaged last-row attention over the image tokens.

    An InternVL sibling of
    `lmms_eval.models.model_utils.lastrow_attn.LastRowAttentionCapture`, which cannot be
    reused: that module imports `repeat_kv` from the Qwen2.5-VL modeling file and
    applies mRoPE. The row is computed as a matrix-vector product rather than via
    `output_attentions=True`, which would force `attn_implementation="eager"` and
    materialise a `[B, H, seq, seq]` tensor.
    """

    def __init__(self, img_lo: int, img_hi: int, adapter=None) -> None:
        self.img_lo, self.img_hi = img_lo, img_hi
        self.rows: dict[int, torch.Tensor] = {}
        self._patched: dict[nn.Module, object] = {}
        # Backbone is injectable so Qwen3-VL can reuse this class unchanged; the default
        # is the InternVL3 adapter, so existing callers are bit-for-bit unaffected. The
        # adapter is what knows q_norm/k_norm and which rotary to apply -- using the
        # wrong one here does not raise, it silently computes the wrong attention row.
        self._adapter = adapter if adapter is not None else _ADAPTER

    def install(self, layers) -> None:
        for layer in layers:
            attn = layer.self_attn
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def _make_wrapper(self, attn: nn.Module, original):
        def wrapper(hidden_states, *args, **kwargs):
            # Real attention first: it writes this step's keys into the cache, so
            # afterwards the cache holds the whole sequence even when `generate` has
            # narrowed hidden_states to one new token. Reading the cache keeps the row
            # indexed by absolute position.
            out = original(hidden_states, *args, **kwargs)
            position_embeddings = kwargs.get("position_embeddings")
            past_key_values = kwargs.get("past_key_values")
            if position_embeddings is None or past_key_values is None:
                return out

            with torch.no_grad():
                q, _ = self._adapter.recompute_qk(
                    attn, hidden_states, position_embeddings
                )
                keys, _ = self._adapter.read_cache(past_key_values, attn.layer_idx)
                key_states = self._adapter.expand_kv(keys, attn)
                q_last = q[:, :, -1:, :]
                row = torch.matmul(q_last, key_states.transpose(2, 3)) * attn.scaling
                mask = kwargs.get("attention_mask")
                if mask is not None:
                    if mask.dtype == torch.bool:
                        mask = torch.where(mask, 0.0, neg_inf(row.dtype)).to(row.dtype)
                    row = row + mask[:, :, -1:, : key_states.shape[-2]]
                row = nn.functional.softmax(row, dim=-1, dtype=torch.float32)
                self.rows[attn.layer_idx] = (
                    row[0, :, 0, self.img_lo : self.img_hi].mean(dim=0).detach().cpu()
                )
            return out

        return wrapper

    def remove(self) -> None:
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()


class InternVLCt2sImageMask:
    """From layer K onward, keep only the LEAST-attended `rank` image tokens visible."""

    def __init__(
        self,
        img_lo: int,
        img_hi: int,
        rank: int,
        agg_layer: int,
        capture: InternVLLastRowCapture,
    ) -> None:
        self.img_lo, self.img_hi = img_lo, img_hi
        self.rank = rank
        self.agg_layer = agg_layer
        self.capture = capture
        self._patched: dict[nn.Module, object] = {}
        self._kill_cols: torch.Tensor | None = None
        self.n_selected = 0
        self.n_image_tokens = img_hi - img_lo

    def reset_pass(self) -> None:
        """Drop the previous forward's selection AND the row it came from.

        Both halves matter. A stale `_kill_cols` would freeze the selection at prefill,
        turning SID into a static token-drop and destroying the "text aware" half of
        CT2S; a stale row would let a forward in which layer K-1 did not fire silently
        reuse the previous step's.
        """
        self._kill_cols = None
        self.capture.rows.clear()

    def _select(self, device) -> torch.Tensor:
        row = self.capture.rows.get(self.agg_layer - 1)
        if row is None:
            raise RuntimeError(
                f"SID: layer {self.agg_layer - 1}'s attention row was not captured "
                f"before layer {self.agg_layer} ran; CT2S cannot select tokens."
            )
        n_img = row.shape[-1]
        # Clamp rather than crash: the image-token count is per-sample (tile count
        # varies), so a single-tile image can legitimately have fewer tokens than the
        # requested rank.
        k = min(self.rank, n_img)
        # `largest=False` -- the LEAST-attended tokens. This is SID, not FastV: upstream
        # deliberately inverted FastV's top-k to build a hallucination-amplified stream.
        keep = torch.topk(row, k, largest=False).indices
        kill = torch.ones(n_img, dtype=torch.bool)
        kill[keep] = False
        self.n_selected, self.n_image_tokens = int(k), int(n_img)
        return (kill.nonzero(as_tuple=True)[0] + self.img_lo).to(device)

    def install(self, layers) -> None:
        for layer in layers:
            attn = layer.self_attn
            layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None or layer_idx < self.agg_layer:
                continue
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn.forward)

    @staticmethod
    def _resolve_cache_position(kwargs, hidden_states):
        """The query rows' absolute positions, across the 4.x/5.x attention-kwarg
        change.

        transformers 4.x passes `cache_position` down to `self_attn.forward`; 5.x does
        NOT (verified: the only kwarg that disappears between the two). CT2S needs those
        positions to rebuild the causal mask, and the old code bailed out when the kwarg
        was missing -- so on 5.x the whole selection silently never ran and CT2S kept 0
        of 256 image tokens: inert, finite, and no error.

        `position_ids` IS still passed and is identical in both versions ((1, L),
        0..L-1), and InternVL3 uses plain 1-D RoPE (no mRoPE), so row `0` of
        `position_ids` is exactly `cache_position`. Fall back to it, and only then to a
        contiguous range derived from the cache length.
        """
        cp = kwargs.get("cache_position")
        if cp is not None:
            return cp
        pos_ids = kwargs.get("position_ids")
        if pos_ids is not None:
            # (B, L) for plain RoPE; mRoPE backbones would be (3, B, L) -- take the
            # temporal axis, which for text positions equals the absolute index.
            return pos_ids[0] if pos_ids.dim() == 2 else pos_ids[0, 0]
        if hidden_states is None:
            return None
        # Last resort: derive from what the cache already holds plus this call's rows.
        pkv = kwargs.get("past_key_values")
        past = 0
        if pkv is not None:
            try:
                past = int(pkv.get_seq_length())
            except Exception:  # noqa: BLE001 - a cache without that API
                past = 0
        q_len = hidden_states.shape[1]
        return torch.arange(past, past + q_len, device=hidden_states.device)

    def _make_wrapper(self, original):
        def wrapper(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            cache_position = self._resolve_cache_position(kwargs, hidden_states)
            if hidden_states is None or cache_position is None:
                return original(*args, **kwargs)
            if self._kill_cols is None:
                self._kill_cols = self._select(hidden_states.device)
            kwargs["attention_mask"] = self._biased_mask(
                kwargs.get("attention_mask"), hidden_states, cache_position
            )
            return original(*args, **kwargs)

        return wrapper

    def _biased_mask(self, mask, hidden_states, cache_position) -> torch.Tensor:
        """A 4-D ADDITIVE mask with the non-selected image columns pushed to -inf.

        `mask` arrives as `None` on the sdpa path with an unpadded prompt, so the causal
        structure is rebuilt here from `cache_position` (the query rows' absolute
        positions) -- the same quantity transformers' own mask builder uses. When a mask
        IS supplied it may be BOOLEAN on this backbone (sdpa) rather than additive float
        (eager); converting first is essential, since adding -1e30 to a bool tensor
        promotes rather than suppresses.
        """
        q_pos = cache_position.view(-1, 1)
        kv_len = int(cache_position[-1]) + 1
        if mask is None:
            k_idx = torch.arange(kv_len, device=cache_position.device).view(1, -1)
            blocked = k_idx > q_pos
            mask = torch.zeros(
                blocked.shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            mask = mask.masked_fill(blocked, neg_inf(mask.dtype)).view(
                1, 1, *blocked.shape
            )
            mask = mask.expand(hidden_states.shape[0], 1, *blocked.shape).clone()
        else:
            # Never mutate the model's own mask: it is built once per forward and shared
            # by every layer, so an in-place edit would leak CT2S's masking into layers
            # < K.
            mask = mask[:, :, :, :kv_len]
            if mask.dtype == torch.bool:
                mask = torch.where(mask, 0.0, neg_inf(hidden_states.dtype)).to(
                    hidden_states.dtype
                )
            else:
                mask = mask.clone()
        mask[:, :, :, self._kill_cols] += neg_inf(mask.dtype)
        return mask

    def remove(self) -> None:
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()


class InternVLForConditionalGenerationSID(InternVLForConditionalGeneration):
    """`InternVLForConditionalGeneration` + SID, configured post-construction."""

    _force_compute: bool = False
    sid_cd_alpha: float = 0.0
    sid_cd_beta: float = 0.0
    sid_agg_layer: int = 2
    sid_attention_rank: int = 100
    sid_attention_rank_ratio: float = 0.0

    def __init__(self, config):
        super().__init__(config)
        self._adapter = InternVL3Adapter()
        self._aux_cache = None
        self._sid_span = None
        self.sid_last_n_selected = 0
        self.sid_last_n_image_tokens = 0

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Required no-op; `_aux_cache` and the span rebuild at each main-stream
        prefill."""

    def _resolve_rank(self, n_img: int) -> int:
        """`sid_attention_rank_ratio` wins when set; see the module docstring on why it
        is better."""
        if self.sid_attention_rank_ratio > 0.0:
            return max(1, int(round(self.sid_attention_rank_ratio * n_img)))
        return int(self.sid_attention_rank)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        labels=None,
        cache_position=None,
        logits_to_keep=0,
        image_sizes=None,
        **kwargs,
    ):
        main_out = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            image_sizes=image_sizes,
            **kwargs,
        )

        if (
            self.sid_cd_alpha == 0.0
            and self.sid_cd_beta == 0.0
            and not self._force_compute
        ):
            return main_out

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            self._aux_cache = DynamicCache(config=self.config)
            # The span is a property of the PROMPT, resolved once while input_ids still
            # holds the whole sequence; on decode steps it is a single new token.
            self._sid_span = (
                None
                if input_ids is None
                else self._adapter.image_span(
                    input_ids, int(self.config.image_token_id)
                )
            )

        if self._sid_span is None:
            # Text-only prompt: CT2S has no image tokens to select from, so there is no
            # amateur stream. Returning the main logits unchanged is the only faithful
            # thing.
            return main_out

        img_lo, img_hi = self._sid_span
        rank = self._resolve_rank(img_hi - img_lo)
        layers = self._adapter.text_layers(self)
        if not (0 < self.sid_agg_layer < len(layers)):
            raise ValueError(
                f"sid_agg_layer must be in [1, {len(layers) - 1}] (layer K-1's "
                f"attention must exist); got {self.sid_agg_layer!r}."
            )

        capture = InternVLLastRowCapture(img_lo, img_hi)
        ct2s = InternVLCt2sImageMask(img_lo, img_hi, rank, self.sid_agg_layer, capture)
        ct2s.reset_pass()
        # Layer K-1 only: that is the single row CT2S reads.
        capture.install([layers[self.sid_agg_layer - 1]])
        ct2s.install(layers)
        try:
            # transformers 5.x REMOVED `cache_position` from `InternVLModel.forward`
            # while keeping `**kwargs`, so passing it does not raise -- it is swallowed
            # and never reaches the decoder layers. That silently misaligned this
            # auxiliary stream and CT2S selected 0 of 256 image tokens (inert, but
            # non-crashing). Filter against the real signature instead of assuming it is
            # accepted.
            aux_kwargs = model_forward_kwargs(
                self.model.forward,
                cache_position=cache_position,
                image_sizes=image_sizes,
            )
            aux_out = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self._aux_cache,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                use_cache=True,
                **aux_kwargs,
            )
        finally:
            # Unconditional, and BEFORE anything can raise past this point: the main
            # stream must never run with CT2S wrappers or the capture attached.
            ct2s.remove()
            capture.remove()

        self._aux_cache = aux_out.past_key_values
        self.sid_last_n_selected = ct2s.n_selected
        self.sid_last_n_image_tokens = ct2s.n_image_tokens

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        aux_logits = self.lm_head(aux_out.last_hidden_state[:, slice_indices, :])

        main_out.logits = contrastive_combine(
            main_out.logits,
            aux_logits,
            self.sid_cd_alpha,
            self.sid_cd_beta,
            force=self._force_compute,
        )
        return main_out
