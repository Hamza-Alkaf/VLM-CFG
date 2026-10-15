"""SID (Self-Introspective Decoding) on Qwen3-VL -- CT2S + contrastive decoding.

Ported from `internvl_sid_model.py`. Both helper classes (`InternVLLastRowCapture`,
`InternVLCt2sImageMask`) are REUSED, not copied: the capture now takes an injectable
adapter (defaulting to the InternVL3 one, so existing callers are untouched) and the
CT2S mask is already backbone-agnostic -- it only rewrites `attention_mask` columns.

THE ADAPTER IS THE WHOLE PORT, AND GETTING IT WRONG IS SILENT. CT2S selects which image
tokens to keep from layer K-1's attention row, recomputed by hand. With InternVL3's
adapter that recompute would use plain `qwen3.apply_rotary_pos_emb` and would skip
nothing else -- Qwen3-VL DOES have q_norm/k_norm, so the row would be computed from
un-normalised q and would rank the image tokens wrongly. The method would still run,
still mask `rank` columns, and still produce fluent output. `Qwen3VLAdapter` is verified
bit-exact against the real attention in `common/test_qwen3vl_recompute_unit.py`.

TWO THINGS THAT DO NOT TRANSFER, stated rather than silently inherited:
  * `sid_attention_rank=100` was tuned on LLaVA-1.5's fixed 576 image tokens. Qwen3-VL's
    dynamic resolution emits anywhere from a few hundred to a few thousand, so 100 is
    "almost all of them" on a small image and "a few percent" on a large one.
    `sid_attention_rank_ratio` expresses the same intent proportionally and is the
    better arm here; the absolute default is kept only because it is the paper's.
  * `sid_agg_layer=2` was likewise chosen on a 32-layer LLaVA; Qwen3-VL has 36.

DEEPSTACK INTERACTION, worth recording. CT2S masks image COLUMNS in the text tower's
attention, but Qwen3-VL also injects visual features directly into the text residual
stream at layers [8,16,24] via deepstack. Those injections are NOT masked by CT2S --
they enter as hidden-state additions, not as attention keys. So the amateur stream here
is less visually starved than the same method on a non-deepstack backbone, which likely
WEAKENS the contrast. That is a property of the backbone, not a porting bug, and it is
why an effect check (is the amateur stream actually different?) matters more here than
elsewhere.
"""

from __future__ import annotations

from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.common.backbone import Qwen3VLAdapter
from vlmcfg.baselines.sid.internvl3_5_hf import (
    InternVLCt2sImageMask,
    InternVLLastRowCapture,
)
from vlmcfg.baselines.vcd.combine import contrastive_combine


class Qwen3VLForConditionalGenerationSID(Qwen3VLForConditionalGeneration):
    """`Qwen3VLForConditionalGeneration` + SID, configured post-construction."""

    _force_compute: bool = False
    sid_cd_alpha: float = 0.0
    sid_cd_beta: float = 0.0
    sid_agg_layer: int = 2
    sid_attention_rank: int = 100
    sid_attention_rank_ratio: float = 0.0

    def __init__(self, config):
        super().__init__(config)
        self._adapter = Qwen3VLAdapter()
        self._aux_cache = None
        self._sid_span = None
        self.sid_last_n_selected = 0
        self.sid_last_n_image_tokens = 0

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Required no-op; `_aux_cache` and the span rebuild at each main-stream
        prefill."""

    def _resolve_rank(self, n_img: int) -> int:
        """`sid_attention_rank_ratio` wins when set -- see the module docstring on
        why."""
        if self.sid_attention_rank_ratio > 0.0:
            return max(1, int(round(self.sid_attention_rank_ratio * n_img)))
        return int(self.sid_attention_rank)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        main_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
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
            # amateur stream. Returning the main logits unchanged is the faithful thing.
            return main_out

        img_lo, img_hi = self._sid_span
        rank = self._resolve_rank(img_hi - img_lo)
        layers = self._adapter.text_layers(self)
        if not (0 < self.sid_agg_layer < len(layers)):
            raise ValueError(
                f"sid_agg_layer must be in [1, {len(layers) - 1}] (layer K-1's "
                f"attention must exist); got {self.sid_agg_layer!r}."
            )

        capture = InternVLLastRowCapture(img_lo, img_hi, adapter=self._adapter)
        ct2s = InternVLCt2sImageMask(img_lo, img_hi, rank, self.sid_agg_layer, capture)
        ct2s.reset_pass()
        capture.install([layers[self.sid_agg_layer - 1]])  # layer K-1 only
        ct2s.install(layers)
        try:
            aux_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self._aux_cache,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                use_cache=True,
            )
        finally:
            # Unconditional, and BEFORE anything can raise past this point: the main
            # stream must never run with CT2S wrappers or the capture still attached.
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
