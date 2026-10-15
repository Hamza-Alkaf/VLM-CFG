"""Qwen2.5-VL with M3ID (Favero et al., CVPR 2024, arXiv:2403.14003), decoding half
only.

M3ID contrasts the image-conditioned distribution against the SAME model on the SAME
prompt with the image removed, and amplifies the difference by a weight that grows with
the number of tokens generated -- the paper's "conditioning dilution": the longer the
generation, the more the model falls back on its language prior. The DPO half of the
paper is deliberately not implemented.

THE SECOND STREAM IS A DIFFERENT LENGTH, which is the whole difficulty. It is the same
text with the image tokens gone, so it is ~1300 tokens shorter, and Qwen2.5-VL's mRoPE
is stateful. This port therefore keeps three things separate, exactly as
`MARINE/marine_utils_lmms/
qwen_marine_model.py` does for its longer guided stream:

  1. THE KV CACHE. The unconditional stream owns a `DynamicCache` (`self._m3id_cache`),
     rebuilt whenever the main stream prefills.
  2. THE POSITIONS. After `t` generated tokens the main stream is at `L_cond + t` while
     this one is at `L_uncond + t`. `self._m3id_len` tracks its own length and its
     `cache_position` is built from that, never from the main stream's.
  3. THE mRoPE STATE. `Qwen2_5_VLModel.rope_deltas` is ONE mutable attribute on the
     shared inner model, owned by the main stream. This class borrows it for each
     unconditional call and gives it back in a `finally`. Dropping that restore is the
     nastiest failure mode available here: the main stream would silently use the wrong
     mRoPE offset from its second token onward and produce fluent, wrong text with no
     error anywhere. Unit test M3 pins it.

`pixel_values` is always `None` for this stream, so the cost is far below the paper's
stated "two forward passes": the unconditional prefill has no vision tower and ~60
tokens instead of ~1300, so prefill overhead is ~5%. Only decode is genuinely 2x.

NEUTRAL POINT -- proven, not asserted. At `m3id_lambda == 0.0` the weight is
`expm1(0) == 0.0` exactly, so `m3id_combine` returns the conditional logits BIT-FOR-BIT
(see its module docstring for why it adds to raw logits rather than to log-probs). Built
that way so the neutral smoke test exercises the second stream, its cache, its positions
and its `rope_deltas` handling -- the code most likely to be wrong -- rather than an
`if` that skipped all of it. A short-circuit is a legitimate optimisation ONCE THAT HAS
BEEN DEMONSTRATED, and one is installed below gated on `_force_compute`; the neutral
test runs both arms and only the unguarded one means anything.

THE PROMPT SEAM. The unconditional inputs are built by the lmms-eval wrapper
(`m3id_lmms_plugin/models/qwen2_5_vl_m3id.py::_prepare_inputs`) and handed down via
`set_uncond_inputs()`, so this class stays a pure two-stream lockstep with no tokenizer
or chat-template knowledge.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.m3id.combine import m3id_combine, m3id_weight

# `<|image_pad|>`; the span is used only to compute the offset t0, never to mask
# anything.
IMAGE_TOKEN_ID = 151655
OFFSET_MODES = ("tokens_to_image", "none")


def find_image_span(
    input_ids: torch.Tensor, image_token_id: int = IMAGE_TOKEN_ID
) -> tuple[int, int] | None:
    """``(img_lo, img_hi)``: the half-open ``<|image_pad|>`` span, or ``None`` if there
    is none.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"M3ID expects a single unpadded sequence [1, L], "
            f"got {tuple(input_ids.shape)}."
        )
    positions = (input_ids[0] == image_token_id).nonzero().flatten()
    if positions.numel() == 0:
        return None
    img_lo = int(positions[0])
    img_hi = int(positions[-1]) + 1
    if img_hi - img_lo != positions.numel():
        raise ValueError(
            "M3ID found a non-contiguous image-token span (multi-image prompt?); "
            "t0 would be ambiguous."
        )
    return img_lo, img_hi


class Qwen2_5_VLForConditionalGenerationM3ID(Qwen2_5_VLForConditionalGeneration):
    """`Qwen2_5_VLForConditionalGeneration` + M3ID, configured by attributes set
    post-construction.
    """

    # Neutral defaults: lambda 0 => weight expm1(0) == 0 => the combine is the identity.
    m3id_lambda: float = 0.0
    m3id_alpha: float = 0.3
    m3id_offset_mode: str = "tokens_to_image"
    m3id_t0_override: int = -1
    m3id_max_weight: float = 0.0
    # Bypasses the lambda==0 short-circuit so the neutral test exercises the real
    # machinery.
    _force_compute: bool = False

    def __init__(self, config):
        super().__init__(config)
        self._m3id_uncond_inputs = None
        self._m3id_uncond_set = False
        self._m3id_cache = None
        self._m3id_len = 0
        self._m3id_rope_deltas = None
        self._m3id_span = None
        self._m3id_t = 0
        self._m3id_t0 = 0
        # Public bookkeeping the effect smoke test reads back.
        self.m3id_last_t_eff = None
        self.m3id_last_weight = None
        self.m3id_last_gate_fired = None
        self.m3id_n_gate_fired = 0
        self.m3id_n_steps = 0
        # Every step's `max_k p_cond`. The confidence gate fires only where this is
        # BELOW alpha, so when a run reports zero corrections this is the distribution
        # that says whether the gate is correctly closed (a confident model) or wrongly
        # closed (a bug).
        self.m3id_max_p: list = []

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Drop the previous document's unconditional bundle and stream state.

        `chat/qwen2_5_vl.py::generate_until` calls this on `self.model` once per chunk.
        Clearing `_m3id_uncond_set` is a TRIPWIRE, not housekeeping: without it, a
        wiring bug that skipped `set_uncond_inputs` would silently contrast this
        document's image against the PREVIOUS document's question, which no metric would
        reveal.
        """
        self._m3id_uncond_inputs = None
        self._m3id_uncond_set = False
        self._m3id_cache = None
        self._m3id_len = 0
        self._m3id_rope_deltas = None
        self._m3id_span = None
        self._m3id_t = 0
        self._m3id_t0 = 0

    def set_uncond_inputs(self, uncond_inputs) -> None:
        """Hand down this document's image-free inputs.

        Args:
            uncond_inputs: a processor output (mapping with `input_ids` and
                `attention_mask`, and deliberately no `pixel_values`) for the same
                prompt with every image and video dropped, already on the model's
                device.
        """
        self._m3id_uncond_inputs = uncond_inputs
        self._m3id_uncond_set = True

    def _uncond_logits(
        self, next_input_ids: torch.Tensor, is_prefill: bool
    ) -> torch.Tensor:
        """One lockstep step of the text-only stream; returns its last-position logits
        `[B,1,V]`.

        At prefill this runs the image-free prompt (shorter than the main stream's). At
        every decode step it is fed `next_input_ids` -- the token the combined logits
        produced, i.e. the same token the main stream just consumed -- so the two
        streams share a token history while keeping separate lengths and positions.
        """
        uncond = self._m3id_uncond_inputs

        # mRoPE state is one attribute on the shared inner model, owned by the MAIN
        # stream. Borrow it, always give it back.
        saved_rope_deltas = self.model.rope_deltas
        try:
            if is_prefill:
                self._m3id_cache = DynamicCache(config=self.config)
                self._m3id_len = 0
                input_ids = uncond["input_ids"]
                attention_mask = uncond.get("attention_mask", None)
                seq_len = int(input_ids.shape[1])
                cache_position = torch.arange(seq_len, device=input_ids.device)
                # `rope_deltas = None` + `cache_position[0] == 0` sends
                # `Qwen2_5_VLModel.forward` down its own `get_rope_index` prefill
                # branch. With no image this is the text-only branch, giving plain
                # 0..L-1 positions and a delta of 0 -- but it is computed by upstream
                # transformers rather than assumed here.
                self.model.rope_deltas = None
            else:
                input_ids = next_input_ids
                seq_len = int(input_ids.shape[1])
                cache_position = torch.arange(
                    self._m3id_len, self._m3id_len + seq_len, device=input_ids.device
                )
                attention_mask = torch.ones(
                    (input_ids.shape[0], self._m3id_len + seq_len),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                # Restore THIS stream's delta so the decode branch
                # (`cache_position[0] + self.rope_deltas`) uses the unconditional
                # offset.
                self.model.rope_deltas = self._m3id_rope_deltas

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=self._m3id_cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                pixel_values=None,
                image_grid_thw=None,
                cache_position=cache_position,
                return_dict=True,
            )
            if is_prefill:
                self._m3id_rope_deltas = self.model.rope_deltas
        finally:
            self.model.rope_deltas = saved_rope_deltas

        self._m3id_cache = out.past_key_values
        self._m3id_len += seq_len
        # Only the last position is ever combined, so running lm_head over just that row
        # (rather than over the whole prefill) is the same value for far less compute.
        return self.lm_head(out.last_hidden_state[:, -1:, :])

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

        # EARNED short-circuit (see the module docstring): only legitimate because the
        # unguarded arithmetic has been shown bit-identical. `_force_compute` disables
        # it so that claim stays testable rather than becoming folklore.
        if self.m3id_lambda == 0.0 and not self._force_compute:
            return main_out

        if not self._m3id_uncond_set:
            raise RuntimeError(
                "M3ID is enabled but no unconditional inputs were handed down for "
                "this sample. `Qwen2_5_VL_M3ID._prepare_inputs` must call "
                "`set_uncond_inputs(...)` after every `reset_sample_tracking(...)`; "
                "a bare `model.generate(...)` outside that wrapper cannot use M3ID."
            )

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            if input_ids is None:
                raise RuntimeError(
                    "M3ID needs input_ids at prefill to locate the image span for t0."
                )
            self._m3id_t = 0
            self._m3id_span = find_image_span(input_ids, self.config.image_token_id)
            self._m3id_t0 = self._resolve_t0(input_ids)

        # Text-only prompt: `l_u` would be `l_c`, so the correction is identically zero
        # and the method is undefined. Skipped rather than run, matching SID's
        # precedent.
        if self._m3id_span is None:
            return main_out

        if input_ids is None or input_ids.shape[0] != 1:
            raise NotImplementedError(
                f"M3ID requires batch_size 1 and num_beams 1 (got a batch of "
                f"{None if input_ids is None else input_ids.shape[0]}): HF expands "
                f"the batch to num_beams, and one unconditional cache cannot serve "
                f"several main-stream rows."
            )
        if main_out.logits.shape[1] != 1:
            raise NotImplementedError(
                f"M3ID combines the NEXT-token distribution only, but this forward "
                f"produced {main_out.logits.shape[1]} logit positions. Use it through "
                f"`generate()`; teacher-forced scoring is not supported."
            )

        uncond_logits = self._uncond_logits(input_ids, is_prefill)

        t_eff = self._m3id_t0 + self._m3id_t
        self._m3id_t += 1
        self.m3id_last_t_eff = t_eff
        self.m3id_last_weight = m3id_weight(t_eff, self.m3id_lambda)
        self.m3id_n_steps += 1

        # Recorded before combining: this is the quantity the gate thresholds, and it is
        # the only way to tell "the model was confident" from "the gate is broken".
        self.m3id_max_p.append(
            float(main_out.logits[:, -1, :].float().softmax(-1).max())
        )

        combined = m3id_combine(
            main_out.logits,
            uncond_logits,
            t=t_eff,
            lam=self.m3id_lambda,
            alpha=self.m3id_alpha,
            max_weight=self.m3id_max_weight,
        )
        self.m3id_last_gate_fired = bool((combined != main_out.logits).any())
        self.m3id_n_gate_fired += int(self.m3id_last_gate_fired)
        main_out.logits = combined
        return main_out

    def _resolve_t0(self, input_ids: torch.Tensor) -> int:
        """The offset added to the decode step index, per the paper's Sec. 5.2.

        M3ID's weight is `expm1(lambda*t)`, which is EXACTLY ZERO at `t = 0` -- so on a
        single-token answer the method is a mathematical no-op. The paper hits this on
        POPE and fixes it by offsetting: "we take an offset into account when using M3ID
        on POPE, and select t = t0 where t0 is the number of tokens in between the
        output and the image." Every benchmark in this comparison is short-answer VQA of
        exactly that shape, so `"tokens_to_image"` is the default; `"none"` reproduces
        the paper's captioning regime and exists as the ablation that makes the
        degeneracy visible rather than hidden.

        t0 EXCLUDES the image tokens -- it is measured from `img_hi`, the END of the
        image span. This is not a nit. With `lambda = 0.02` the weight is 1.72 at t=50
        and 6.39 at t=100, but 1.9e11 at t=1300, and Qwen2.5-VL's dynamic resolution
        routinely emits 1000-3000 image tokens. Measuring from `img_lo` instead would
        inflate the weight by ~11 orders of magnitude AND make it vary with image size,
        giving a different method that still "runs".
        """
        if self.m3id_t0_override >= 0:
            return int(self.m3id_t0_override)
        if self.m3id_offset_mode == "none" or self._m3id_span is None:
            return 0
        if self.m3id_offset_mode != "tokens_to_image":
            raise ValueError(
                f"m3id_offset_mode must be one of {OFFSET_MODES}, "
                f"got {self.m3id_offset_mode!r}."
            )
        return int(input_ids.shape[1]) - int(self._m3id_span[1])
