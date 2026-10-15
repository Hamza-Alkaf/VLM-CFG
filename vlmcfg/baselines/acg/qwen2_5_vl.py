"""Qwen2.5-VL with ACG (Attention-space Contrastive Guidance, arXiv:2601.13707).

ACG is SINGLE-PASS. Unlike VCD/LayerCD/SID (a degraded second forward) or MARINE/M3ID
(an enriched or image-free second forward), its "unconditional" path is a re-softmax of
the SAME score matrix with the visual keys masked out, computed inside each attention
layer. There is no second model call, no second KV cache, and no `rope_deltas` to juggle
-- which is why this class is short and why the whole method lives in
`acg_attention.ACGIntervention`.

NEUTRAL POINT -- proven, not asserted. At `acg_gamma == 0.0` this class STILL installs
the intervention and STILL runs every line of it: both softmaxes, the visual mask, the
orthogonalization, the `o_proj`. The correction it splices is `0.0 * finite == +-0.0`,
and `x + +-0.0 == x` for every float, so the result is bit-for-bit the base model's.
That makes `max|dlogit| == 0.0` a statement about ACG'S ARITHMETIC rather than about an
`if` that skipped it.

A short-circuit is a legitimate optimisation once that has been demonstrated -- it saves
28 layers of recompute on control arms -- so one is installed below, gated on
`_force_compute`. The neutral smoke test runs BOTH arms (`_force_compute=True` and the
default), and only the first one means anything. Do not add a guard to a method whose
unguarded neutral test has not been run.

INSTALL/UNINSTALL is per forward, in a `try/finally`, following SID
(`sid_utils_lmms/qwen_sid_model.py`). The tempting alternative -- install once at
prefill and uninstall in `reset_sample_tracking` -- leaks three ways:
`reset_sample_tracking` fires at the START of a chunk (so the wrapper outlives the final
chunk), `loglikelihood` never calls it at all, and an exception inside `generate()`
leaves it attached. Each leaves a STALE image span silently masking whatever tokens
happen to sit at the previous image's offsets. Rebinding 28 wrappers per decode step
costs ~3 us against a ~30 ms forward, so there is nothing to buy by being clever.
"""

from transformers import Qwen2_5_VLForConditionalGeneration

from vlmcfg.baselines.acg.attention import (
    ACGIntervention,
    find_image_span,
    get_language_model_layers,
)


class Qwen2_5_VLForConditionalGenerationACG(Qwen2_5_VLForConditionalGeneration):
    """`Qwen2_5_VLForConditionalGeneration` + ACG, configured by attributes set
    post-construction.

    The lmms-eval wrapper (`acg_lmms_plugin/models/qwen2_5_vl_acg.py`) sets `acg_gamma`,
    `acg_layer_range`, `acg_orth_scope` and `acg_eps` on the instance after
    `from_pretrained`. Class-level defaults are the NEUTRAL point, so a bare
    `from_pretrained` of this class behaves exactly like the stock model.
    """

    # Neutral defaults: gamma 0 and no layer range => ACG requested of nothing.
    acg_gamma: float = 0.0
    acg_layer_range: tuple[int, int] | None = None
    acg_orth_scope: str = "concat"
    acg_eps: float = 1e-6
    # Bypasses the gamma==0 short-circuit so the neutral test exercises the real
    # arithmetic.
    _force_compute: bool = False
    # Turns on ACGIntervention's per-layer diagnostics (effect smoke test only).
    acg_record_stats: bool = False

    def __init__(self, config):
        super().__init__(config)
        self._acg_span: tuple[int, int] | None = None
        self._acg_handle: ACGIntervention | None = None

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Required no-op.

        `chat/qwen2_5_vl.py::generate_until` calls this on `self.model` once per chunk
        regardless of `cfg_mode`, so the attribute must exist. ACG carries NO per-sample
        state across forwards -- the intervention is installed and removed inside a
        single `forward` -- so there is deliberately nothing to reset here. Clearing the
        span is belt-and-braces: it is recomputed at every prefill anyway.
        """
        self._acg_span = None
        self._acg_handle = None

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
        def _plain():
            return super(Qwen2_5_VLForConditionalGenerationACG, self).forward(
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

        # Nothing requested.
        if self.acg_layer_range is None:
            return _plain()

        # EARNED short-circuit (see the module docstring): only valid because the
        # unguarded arithmetic has been shown to be bit-identical. `_force_compute`
        # disables it so that claim stays testable rather than becoming folklore.
        if self.acg_gamma == 0.0 and not self._force_compute:
            return _plain()

        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError(
                "ACG masks the image-token span; "
                "a video's visual tokens fall outside it, so the 'unconditional' path "
                "would still be conditioned on the video."
            )

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            if input_ids is None:
                raise RuntimeError(
                    "ACG needs input_ids at prefill to locate the image span."
                )
            self._acg_span = find_image_span(input_ids, self.config.image_token_id)
            if self._acg_span is None:
                self._acg_handle = None
            else:
                if int(input_ids[0, -1]) == self.config.image_token_id:
                    raise RuntimeError(
                        "ACG intervenes on the LAST TEXT token's row, but this prompt "
                        "ends with an image token; row i* is not defined."
                    )
                self._acg_handle = ACGIntervention(
                    *self.acg_layer_range,
                    *self._acg_span,
                    gamma=self.acg_gamma,
                    orth_scope=self.acg_orth_scope,
                    eps=self.acg_eps,
                    record_stats=self.acg_record_stats,
                )

        # Text-only prompt: there is no visual key to mask, so `O_uncond == O_cond` and
        # ACG is the identity by construction. Skipped rather than run, matching SID's
        # precedent.
        if self._acg_handle is None:
            return _plain()

        self._acg_handle.install(get_language_model_layers(self))
        try:
            return _plain()
        finally:
            # Unconditional, and before anything downstream can raise: no forward
            # outside this call may ever run with a wrapper (and a possibly stale span)
            # attached.
            self._acg_handle.remove()
