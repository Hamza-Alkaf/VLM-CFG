"""LayerCD ported to Qwen2.5-VL.

Despite the name, this repo's actual method (`cd_utils.py`) does not contrast against an
earlier LLM decoder layer (no DoLa-style logit lens). It contrasts against a second
forward pass through the SAME image with the vision tower forcibly truncated to its
shallowest block -- LLaVA: `vision_tower.select_layer=1`; Molmo: `vit_layers=[1,0]`;
Cambrian: each sub-tower forced to its first block. Qwen2.5-VL's vision tower has no
discrete `select_layer` attribute (it's a flat `nn.ModuleList` of blocks, `config.depth`
deep, no separate CLIP-style layer selection), so the equivalent here is temporarily
replacing `visual.blocks` with `visual.blocks[:immature_depth]` for the impaired pass
only.

Same lockstep-dual-cache `forward()` override as VCD's port (see
`../../VCD/vcd_utils/qwen_vcd_model.py` for the full rationale): both streams see the
identical token history at every decode step, avoiding both the original repo's
version-fragile monkey-patch of `transformers.GenerationMixin` and this project's own
CFG pipeline's independent-pre-computed-trajectory approximation.
"""

import contextlib

from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.vcd.combine import contrastive_combine


@contextlib.contextmanager
def truncate_vision_blocks(visual_module, depth: int):
    """Temporarily keep only the first `depth` blocks of a Qwen2.5-VL vision tower.

    `visual_module.forward` (`Qwen2_5_VisionTransformerPretrainedModel.forward`)
    iterates `for layer_num, blk in enumerate(self.blocks): if layer_num in
    self.fullatt_block_indexes: ...` -- a membership test, not an index into a
    fixed-length structure, so truncating `blocks` is safe even though
    `fullatt_block_indexes` may name indices beyond the truncated length: they simply
    never come up during the truncated pass. No other part of vision-tower forward
    depends on `len(self.blocks)` matching `config.depth`.
    """
    original_blocks = visual_module.blocks
    if depth >= len(original_blocks):
        yield
        return
    visual_module.blocks = original_blocks[:depth]
    try:
        yield
    finally:
        visual_module.blocks = original_blocks


class Qwen2_5_VLForConditionalGenerationLayerCD(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with LayerCD's shallow-vision-tower contrastive decoding.

    `cd_alpha`/`cd_beta`/`immature_depth` are plain instance attributes (defaulted to
    LayerCD's neutral point below), set by the lmms-eval wrapper model
    (`layercd_lmms_plugin/models/qwen2_5_vl_layercd.py`) right after `from_pretrained`
    loads this class -- same "configure post-construction" pattern as the VCD port and
    this project's own `attn_intervene_*`.
    """

    # EARNED-GUARD BYPASS. The neutral guards below are an optimisation, not the proof
    # that a method is neutral: with the guard in place, `max|dlogit| == 0` only
    # demonstrates that the BRANCH works, and an implementation whose arithmetic is
    # wrong passes it untouched. Setting `_force_compute = True` runs the whole method
    # with its strength at zero, so the neutral test measures the real arithmetic. Kept
    # permanently so that check stays a repeatable regression test rather than something
    # done once during development.
    _force_compute: bool = False
    cd_alpha: float = 0.0
    cd_beta: float = 0.0
    immature_depth: int = 1

    def __init__(self, config):
        super().__init__(config)
        self._aux_cache = None

    def reset_sample_tracking(self, doc_id=None):
        """No-op: `chat/qwen2_5_vl.py::generate_until` calls this unconditionally on
        every `model_fn` at the start of each chunk (regardless of `cfg_mode`), since
        the CFG model class (`Qwen2_5_VLForConditionalGenerationWithCFG`) uses it to
        flush per-sample bookkeeping. This wrapper needs none of that: `_aux_cache`
        already resets itself inside `forward()` whenever the main stream is at its own
        prefill step (`cache_position[0] == 0`), so there is nothing to flush between
        samples here.
        """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
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
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Neutral point: skip the second forward pass entirely, so the returned logits
        # are bit-for-bit the base model's -- not merely numerically close to it.
        if self.cd_alpha == 0.0 and self.cd_beta == 0.0 and not self._force_compute:
            return main_out

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            self._aux_cache = DynamicCache(config=self.config)
        # `pixel_values` is already None on every non-prefill step (see
        # `Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation`); the
        # impaired pass reuses the real image unchanged -- only the vision tower's depth
        # changes, and only for whichever call (prefill only) actually has pixel_values
        # to encode.
        with truncate_vision_blocks(self.model.visual, self.immature_depth):
            aux_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self._aux_cache,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                cache_position=cache_position,
                return_dict=True,
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
