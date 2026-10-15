"""ACG (Attention-Consistency Guidance) on Qwen3-VL.

Ported from `internvl_acg_model.py`; the mechanism is unchanged and the intervention
itself is reused via `Qwen3VLACGIntervention` (a one-line adapter swap -- see that
module).

NEUTRAL POINT -- proven, not asserted. At `acg_gamma == 0.0` this class STILL installs
the intervention and still runs the whole recompute when `_force_compute` is set; the
splice is additive and bias-free, so the contribution is exactly +-0.0. A guard exists
on top purely as an optimisation, and the neutral test runs both arms.

LAYER RANGE. Qwen3-VL-8B has 36 text layers, so ACG-Full is `[0,36]` -- the same
rescaling already applied for InternVL3.5-8B, and NOT the `[0,28]` used for Qwen2.5-VL.
Passing a 28-layer range here would silently leave the top 8 layers unintervened.
"""

from __future__ import annotations

from transformers import Qwen3VLForConditionalGeneration

from vlmcfg.baselines.acg.qwen3vl_attention import Qwen3VLACGIntervention
from vlmcfg.baselines.common.backbone import Qwen3VLAdapter


class Qwen3VLForConditionalGenerationACG(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with ACG's orthogonal attention-consistency splice."""

    acg_gamma: float = 0.0
    acg_layer_range: tuple | None = None
    acg_orth_scope: str = "concat"
    acg_eps: float = 1e-6
    acg_record_stats: bool = False
    _force_compute: bool = False

    def __init__(self, config):
        super().__init__(config)
        self._adapter = Qwen3VLAdapter()
        self._acg_span = None
        self._acg_handle: Qwen3VLACGIntervention | None = None

    def reset_sample_tracking(self, doc_id=None) -> None:
        """No-op; the span is recomputed at every prefill and the handle removed per
        forward."""
        self._acg_span = None

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
        def _plain():
            return super(Qwen3VLForConditionalGenerationACG, self).forward(
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

        if self.acg_layer_range is None:
            return _plain()
        if self.acg_gamma == 0.0 and not self._force_compute:
            return _plain()

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            if input_ids is None:
                raise RuntimeError(
                    "ACG needs input_ids at prefill to locate the image span."
                )
            # Raises on a disjoint span (video / multi-image) rather than masking the
            # text between the runs.
            self._acg_span = self._adapter.image_span(
                input_ids, int(self.config.image_token_id)
            )
            if self._acg_span is None:
                self._acg_handle = None
            else:
                if int(input_ids[0, -1]) == int(self.config.image_token_id):
                    raise RuntimeError(
                        "ACG intervenes on the LAST TEXT token's row, but this prompt "
                        "ends with an image token; row i* is not defined."
                    )
                self._acg_handle = Qwen3VLACGIntervention(
                    *self.acg_layer_range,
                    *self._acg_span,
                    gamma=self.acg_gamma,
                    orth_scope=self.acg_orth_scope,
                    eps=self.acg_eps,
                    record_stats=self.acg_record_stats,
                )

        # Text-only prompt: no visual key to mask, so O_uncond == O_cond and ACG is the
        # identity by construction. Skipped rather than run.
        if self._acg_handle is None:
            return _plain()

        self._acg_handle.install(self._adapter.text_layers(self))
        try:
            return _plain()
        finally:
            # Unconditional: no forward outside this call may run with a wrapper (and a
            # possibly stale span) still attached.
            self._acg_handle.remove()
