"""LayerCD (shallow-vision-tower contrastive decoding) on Qwen3-VL.

Ported from `qwen_layercd_model.py`. `contrastive_combine` and `truncate_vision_blocks`
are imported unchanged -- the latter only swaps `visual.blocks`, and Qwen3-VL's vision
forward iterates `for layer_num, blk in enumerate(self.blocks)` exactly as Qwen2.5-VL's
does, so the truncation is structurally safe here too.

DEVIATION THAT MUST BE REPORTED WITH ANY LayerCD NUMBER ON THIS BACKBONE -- deepstack.
Qwen3-VL re-injects visual features into the TEXT tower at three depths, and it collects
them from inside the vision block loop (modeling_qwen3_vl.py:737-749):

    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(...)
        if layer_num in self.deepstack_visual_indexes:      # [8, 16, 24] on the 8B
            deepstack_feature_lists.append(merger(hidden_states))

With `immature_depth=1` the loop never reaches index 8, so `deepstack_feature_lists`
comes back EMPTY. The text tower then skips injection entirely, because it guards with
`layer_idx in range(len(deepstack_visual_embeds))` (:862) -- so nothing crashes, and
nothing warns. The impaired stream is therefore degraded along TWO axes at once (shallow
vision AND no deepstack), where the paper's method degrades only depth.

This is a property of running LayerCD on a deepstack backbone, not a porting bug: any
truncation below `min(deepstack_visual_indexes)` has it. It is recorded here, and
surfaced by `deepstack_lost` on the instance so a caller can assert on it rather than
rediscover it. An `immature_depth >= 9` arm would keep the first deepstack injection and
is the closer analogue of the paper's setting on this backbone -- worth running as a
sensitivity check before any LayerCD conclusion is drawn here.
"""

from __future__ import annotations

from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.layercd.qwen2_5_vl import truncate_vision_blocks
from vlmcfg.baselines.vcd.combine import contrastive_combine


class Qwen3VLForConditionalGenerationLayerCD(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with LayerCD's shallow-vision-tower contrastive decoding."""

    # EARNED-GUARD BYPASS -- see qwen3vl_vcd_model.py for the full rationale.
    _force_compute: bool = False
    cd_alpha: float = 0.0
    cd_beta: float = 0.0
    immature_depth: int = 1

    def __init__(self, config):
        super().__init__(config)
        self._aux_cache = None
        self.deepstack_lost = None  # set on first impaired pass; see module docstring

    def reset_sample_tracking(self, doc_id=None):
        """No-op; `_aux_cache` resets itself at each prefill."""

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

        if self.cd_alpha == 0.0 and self.cd_beta == 0.0 and not self._force_compute:
            return main_out

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            self._aux_cache = DynamicCache(config=self.config)

        # Record whether this truncation costs the deepstack injections, so the
        # deviation is observable from outside instead of being inferred from the
        # config.
        ds_idx = list(
            getattr(self.config.vision_config, "deepstack_visual_indexes", []) or []
        )
        self.deepstack_lost = bool(ds_idx) and self.immature_depth <= min(ds_idx)

        with truncate_vision_blocks(self.model.visual, self.immature_depth):
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
