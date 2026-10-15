"""InternVL3 with LayerCD -- contrast against a forward through a TRUNCATED vision
tower.

Despite the name, the method contrasts against a second pass through the same image with
the vision encoder forcibly cut to its shallowest block(s), not against an earlier LLM
decoder layer. See `qwen_layercd_model.py` for the full provenance.

WHY TRUNCATE THE MODULELIST RATHER THAN USE `vision_feature_layer`.
`InternVLModel.get_image_features` looks like it supports selecting an intermediate
vision layer::

    if vision_feature_layer == -1:
        vision_features = self.vision_tower(pixel_values=pixel_values).last_hidden_state
    else:
        vision_features = self.vision_model(pixel_values=pixel_values).hidden_states[vision_feature_layer]

but `self.vision_model` does not exist on that class -- only `self.vision_tower` is ever
assigned (`modeling_internvl.py:534`). So any `vision_feature_layer != -1` raises
`AttributeError` in transformers 4.57.1. Truncating the encoder's `ModuleList` is
therefore not merely the analogue of the Qwen port's approach, it is the only one that
works.

TRUNCATION IS CLEAN ON THIS BACKBONE. `InternVLVisionEncoder.forward` is a plain
`for layer_module in self.layer` with no index-based branching (Qwen2.5-VL needed an
argument about `fullatt_block_indexes` being a membership test), and
`InternVLVisionModel.layernorm` is `nn.Identity()` whenever `use_mean_pooling=True` --
which it is for every InternVL3 checkpoint -- so the shallow readout is not passed
through a norm calibrated for the full stack. Note the attribute is `encoder.layer`,
SINGULAR, not `.layers` or `.blocks`.
"""  # noqa: E501

import contextlib

from transformers import InternVLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.vcd.combine import contrastive_combine


@contextlib.contextmanager
def truncate_vision_layers(vision_tower, depth: int):
    """Temporarily keep only the first `depth` layers of an InternVL vision encoder."""
    encoder = vision_tower.encoder
    original = encoder.layer
    if depth >= len(original):
        yield
        return
    encoder.layer = original[:depth]
    try:
        yield
    finally:
        # Unconditional: a forward outside this block must never run on a truncated
        # tower.
        encoder.layer = original


class InternVLForConditionalGenerationLayerCD(InternVLForConditionalGeneration):
    """`InternVLForConditionalGeneration` + LayerCD, configured post-construction."""

    _force_compute: bool = False
    cd_alpha: float = 0.0
    cd_beta: float = 0.0
    immature_depth: int = 1

    def __init__(self, config):
        super().__init__(config)
        self._aux_cache = None

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Required no-op; `_aux_cache` rebuilds itself at each main-stream prefill."""

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

        # Only the prefill step actually runs the vision tower (`pixel_values` is None
        # afterwards), so the truncation only bites where it is meant to.
        with truncate_vision_layers(self.model.vision_tower, self.immature_depth):
            aux_out = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
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
