"""VCD (Visual Contrastive Decoding) on Qwen3-VL.

Ported from `qwen_vcd_model.py` (Qwen2.5-VL). The method itself is unchanged -- a second
forward over diffusion-noised pixels, contrasted with the standard adaptive-plausibility
formula -- so `contrastive_combine` and `_noised_pixel_values` are imported rather than
duplicated. What differs is entirely backbone plumbing, and each difference is
load-bearing:

1. FORWARD SIGNATURE. `Qwen3VLForConditionalGeneration.forward` does NOT name
   `use_cache`, `output_attentions`, `output_hidden_states`, `rope_deltas` or
   `second_per_grid_ts` -- all five moved into `**kwargs` (they are named params on
   Qwen2.5-VL). Passing them positionally, or forwarding `second_per_grid_ts=None` into
   `Qwen3VLModel`, is how this port would break. This class therefore mirrors the
   Qwen3-VL signature exactly and lets `**kwargs` carry the rest.

2. `Qwen3VLModel.forward` accepts (input_ids, attention_mask, position_ids,
   past_key_values, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw,
   video_grid_thw, cache_position, **kwargs) -- note NO `second_per_grid_ts`.

3. PIXEL LAYOUT IS UNCHANGED. Qwen3-VL uses the same `Qwen2VLImageProcessorFast` with
   `temporal_patch_size=2` and `merge_size=2`; only `patch_size` differs (16 vs 14) and
   the normalisation stats (0.5/0.5 vs CLIP's). `_noised_pixel_values` reads
   `patch_size`, `temporal_patch_size` and `in_channels` off `vision_config` at call
   time, so it computes `expected = 3*2*16*16 = 1536` here without modification -- the
   frame-duplication argument in its docstring holds identically.

4. DEEPSTACK is handled inside `Qwen3VLModel`: it calls `get_image_features`, which
   returns `(embeds, deepstack_features)`, and routes both. Because the aux pass goes
   through the SAME `self.model(...)` entry point with noised pixels, its deepstack
   features are derived from the noised image too -- which is what VCD intends (the
   whole visual pathway is degraded, not just its first injection).
"""

from __future__ import annotations

from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.vcd.combine import contrastive_combine
from vlmcfg.baselines.vcd.qwen2_5_vl import _noised_pixel_values


class Qwen3VLForConditionalGenerationVCD(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with VCD's image-noise contrastive decoding.

    `cd_alpha`/`cd_beta`/`noise_step` are plain instance attributes defaulted to the
    neutral point, set by the lmms-eval wrapper right after `from_pretrained` builds
    this class -- the same "configure post-construction" pattern every other port in
    this repo uses.
    """

    # EARNED-GUARD BYPASS. The neutral guard below is an optimisation, not proof of
    # neutrality: with it in place `max|dlogit| == 0` shows only that the BRANCH works,
    # and a wrong implementation passes untouched. `_force_compute = True` runs the full
    # method at strength zero so the neutral test measures the real arithmetic. Kept
    # permanently.
    _force_compute: bool = False
    cd_alpha: float = 0.0
    cd_beta: float = 0.0
    noise_step: int = 500

    def __init__(self, config):
        super().__init__(config)
        self._aux_cache = None

    def reset_sample_tracking(self, doc_id=None):
        """No-op. `chat/qwen3_vl.py::generate_until` calls this on every `model_fn` at
        the start of each chunk because the CFG class uses it to flush per-sample state.
        Nothing to flush here: `_aux_cache` resets itself in `forward()` at each
        prefill.
        """

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

        # Neutral point: skip the aux pass entirely, so the logits are bit-for-bit the
        # base model's -- not merely numerically close to it.
        if self.cd_alpha == 0.0 and self.cd_beta == 0.0 and not self._force_compute:
            return main_out

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            self._aux_cache = DynamicCache(config=self.config)
            impaired_pixel_values = (
                None
                if pixel_values is None
                else _noised_pixel_values(
                    pixel_values,
                    self.noise_step,
                    vision_config=getattr(self.config, "vision_config", None),
                )
            )
        else:
            # `pixel_values` is already None on every non-prefill step (see
            # `prepare_inputs_for_generation`); nothing to noise.
            impaired_pixel_values = pixel_values

        aux_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self._aux_cache,
            inputs_embeds=inputs_embeds,
            pixel_values=impaired_pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            use_cache=True,
        )
        self._aux_cache = aux_out.past_key_values

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        aux_logits = self.lm_head(aux_out.last_hidden_state[:, slice_indices, :])

        main_out.logits = contrastive_combine(
            main_out.logits,
            aux_logits,
            self.cd_alpha,
            self.cd_beta,
            force=self._force_compute,
        )
        return main_out
