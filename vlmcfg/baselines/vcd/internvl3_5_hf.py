"""InternVL3 with VCD (Visual Contrastive Decoding, Leng et al., CVPR 2024).

Same method, same `contrastive_combine`, same dual-cache lockstep as the Qwen2.5-VL port
in `qwen_vcd_model.py`. Only the degradation and the forward signature change.

THE NOISE IS MUCH SIMPLER HERE, AND THAT IS A REAL DIFFERENCE WORTH RECORDING. On
Qwen2.5-VL the processor hands back a patch tensor of shape
`(n_patches, C * temporal_patch_size * ph * pw)` in which a single image is DUPLICATED
along a temporal axis, so the port has to unflatten to `(n, C, T, ph, pw)`, noise the
`T=0` slice and re-expand -- otherwise the two copies get independent noise,
`patch_embed`'s Conv3d averages them, and the effective noise shrinks by ~sqrt(2).
InternVL's processor returns `(total_tiles, 3, 448, 448)` with no temporal duplication
at all, so the diffusion forward process applies directly to the whole tensor. Nothing
to unflatten, and nothing to get wrong.

THE THUMBNAIL TILE IS NOISED TOO, deliberately. With `crop_to_patches=True` InternVL
appends a downscaled whole-image tile after the crops (`use_thumbnail`, when there is
more than one tile). VCD's premise is that the negative stream sees a globally corrupted
image; leaving the thumbnail clean would hand it an intact global view, which is
precisely the signal the contrast is supposed to remove. Noising the full `pixel_values`
tensor covers every tile including the thumbnail.

NEUTRAL POINT. `cd_alpha == cd_beta == 0` skips the second forward entirely, so the
logits are bit-for-bit the base model's. That guard is an OPTIMISATION, not the proof --
`_force_compute` bypasses it (and is threaded into `contrastive_combine` as `force=`) so
the neutral test measures the arithmetic rather than the branch. See
`qwen_vcd_model.py`'s "EARNED-GUARD BYPASS".
"""

from transformers import InternVLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.vcd.combine import contrastive_combine
from vlmcfg.baselines.vcd.noise import add_diffusion_noise


class InternVLForConditionalGenerationVCD(InternVLForConditionalGeneration):
    """`InternVLForConditionalGeneration` + VCD, configured by attributes set
    post-construction."""

    _force_compute: bool = False
    cd_alpha: float = 0.0
    cd_beta: float = 0.0
    noise_step: int = 500

    def __init__(self, config):
        super().__init__(config)
        self._aux_cache = None

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Required no-op -- the chat wrapper calls this on `self.model` once per chunk.

        VCD carries no cross-forward state that needs flushing here: `_aux_cache` is
        rebuilt inside `forward()` whenever the main stream is at its own prefill
        (`cache_position[0] == 0`).
        """

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

        if self.cd_alpha == 0.0 and self.cd_beta == 0.0 and not self._force_compute:
            return main_out

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            self._aux_cache = DynamicCache(config=self.config)
            impaired_pixel_values = (
                None
                if pixel_values is None
                else add_diffusion_noise(pixel_values, self.noise_step)
            )
        else:
            # `prepare_inputs_for_generation` only passes pixel_values on the prefill
            # step, so this is already None on every decode step; nothing to noise.
            impaired_pixel_values = pixel_values

        aux_out = self.model(
            input_ids=input_ids,
            pixel_values=impaired_pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self._aux_cache,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            cache_position=cache_position,
            image_sizes=image_sizes,
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
