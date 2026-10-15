"""InternVL3 with ACG. Single-pass: no second model call, no second cache, no position
state.

NEUTRAL POINT -- proven, not asserted. At `acg_gamma == 0.0` this class STILL installs
the intervention and STILL runs every line of it. The correction it splices is
`0.0 * finite == +-0.0` and `x + +-0.0 == x` for every float, so the result is
bit-for-bit the base model's. That makes `max|dlogit| == 0.0` a statement about ACG'S
ARITHMETIC rather than about an `if` that skipped it. A short-circuit is a legitimate
optimisation once that has been demonstrated, so one is installed below, gated on
`_force_compute`.

INSTALL/UNINSTALL IS PER FORWARD, in a `try/finally`. The tempting alternative --
install at prefill, uninstall in `reset_sample_tracking` -- leaks three ways:
`reset_sample_tracking` fires at the START of a chunk (so the wrapper outlives the final
chunk), `loglikelihood` never calls it, and an exception inside `generate()` leaves it
attached. Each leaves a STALE image span silently masking whatever tokens sit at the
previous image's offsets.
"""

from transformers import InternVLForConditionalGeneration

from vlmcfg.baselines.acg.internvl_attention import InternVLACGIntervention
from vlmcfg.baselines.common.backbone import InternVL3Adapter


class InternVLForConditionalGenerationACG(InternVLForConditionalGeneration):
    """`InternVLForConditionalGeneration` + ACG, configured post-construction."""

    acg_gamma: float = 0.0
    acg_layer_range: tuple[int, int] | None = None
    acg_orth_scope: str = "concat"
    acg_eps: float = 1e-6
    _force_compute: bool = False
    acg_record_stats: bool = False

    def __init__(self, config):
        super().__init__(config)
        self._adapter = InternVL3Adapter()
        self._acg_span: tuple[int, int] | None = None
        self._acg_handle: InternVLACGIntervention | None = None

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Required no-op. ACG carries no per-sample state across forwards -- the
        intervention is installed and removed inside a single `forward`. Clearing the
        span is belt-and-braces."""
        self._acg_span = None
        self._acg_handle = None

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
        def _plain():
            return super(InternVLForConditionalGenerationACG, self).forward(
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

        if self.acg_layer_range is None:
            return _plain()

        # EARNED short-circuit: valid only because the unguarded arithmetic has been
        # shown bit-identical. `_force_compute` disables it so that claim stays
        # testable.
        if self.acg_gamma == 0.0 and not self._force_compute:
            return _plain()

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            if input_ids is None:
                raise RuntimeError(
                    "ACG needs input_ids at prefill to locate the image span."
                )
            # Raises on a disjoint span (video / multi-image) rather than masking the
            # text between the runs -- see `InternVL3Adapter.image_span`.
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
                self._acg_handle = InternVLACGIntervention(
                    *self.acg_layer_range,
                    *self._acg_span,
                    gamma=self.acg_gamma,
                    orth_scope=self.acg_orth_scope,
                    eps=self.acg_eps,
                    record_stats=self.acg_record_stats,
                )

        # Text-only prompt: no visual key to mask, so `O_uncond == O_cond` and ACG is
        # the identity by construction. Skipped rather than run.
        if self._acg_handle is None:
            return _plain()

        self._acg_handle.install(self._adapter.text_layers(self))
        try:
            return _plain()
        finally:
            # Unconditional, and before anything downstream can raise: no forward
            # outside this call may ever run with a wrapper (and a possibly stale span)
            # attached.
            self._acg_handle.remove()
