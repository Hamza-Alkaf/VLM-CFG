"""M3ID (Multi-Modal Mutual-Information Decoding) on Qwen3-VL.

Ported from `qwen_m3id_model.py`. The method, the offset policy (`t0` measured from the
END of the image span) and the combine are imported unchanged; only the backbone
plumbing moves.

WHY THIS IS NOT A SIMPLIFICATION OVER THE Qwen2.5-VL PORT. The InternVL3 port could
delete M3ID's nastiest section outright, because InternVL has no mRoPE and no
`rope_deltas`. Qwen3-VL does: `Qwen3VLModel.rope_deltas` exists at the same attribute
path (modeling_qwen3_vl.py:899, read back at :1200-1212). It is ONE mutable attribute on
the inner model shared by both streams, and the unconditional stream has a DIFFERENT
length and therefore a different delta. So the borrow/restore block survives verbatim:

    saved = self.model.rope_deltas          # owned by the main stream
    ...   # set None at uncond prefill so get_rope_index recomputes; restore the uncond
          # delta at every uncond decode step
    finally: self.model.rope_deltas = saved # always given back

Dropping it would leave the main stream decoding with the text-only prompt's offset --
which runs, produces fluent text, and is wrong.

SIGNATURE CHANGES. `Qwen3VLForConditionalGeneration.forward` no longer names
`use_cache`, `output_attentions`, `output_hidden_states`, `rope_deltas` or
`second_per_grid_ts`, and `Qwen3VLModel.forward` names none of them either. The two
model calls below therefore pass only what those signatures accept; extras ride in
`**kwargs`.
"""

from __future__ import annotations

import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.m3id.combine import m3id_combine, m3id_weight
from vlmcfg.baselines.m3id.qwen2_5_vl import OFFSET_MODES, find_image_span


class Qwen3VLForConditionalGenerationM3ID(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with M3ID's text-only-stream mutual-information correction."""

    m3id_lambda: float = 0.0
    m3id_alpha: float = 0.3
    m3id_offset_mode: str = "tokens_to_image"
    m3id_t0_override: int = -1
    m3id_max_weight: float = 0.0
    # EARNED-GUARD BYPASS -- see qwen3vl_vcd_model.py.
    _force_compute: bool = False

    def __init__(self, config):
        super().__init__(config)
        self._m3id_uncond_inputs = None
        self._m3id_uncond_set = False
        self._m3id_cache = None
        self._m3id_len = 0
        self._m3id_rope_deltas = None
        self._m3id_t = 0
        self._m3id_t0 = 0
        self._m3id_span = None
        self.m3id_last_t_eff = None
        self.m3id_last_weight = None
        self.m3id_last_gate_fired = False
        self.m3id_n_gate_fired = 0
        self.m3id_n_steps = 0
        self.m3id_max_p: list = []

    def reset_sample_tracking(self, doc_id=None):
        """Clear per-sample state. `_m3id_uncond_set` is a TRIPWIRE, not housekeeping: a
        wiring bug that skipped `set_uncond_inputs` would otherwise silently reuse the
        PREVIOUS document's image-free prompt.
        """
        self._m3id_uncond_inputs = None
        self._m3id_uncond_set = False
        self._m3id_cache = None
        self._m3id_len = 0
        self._m3id_rope_deltas = None
        self._m3id_t = 0
        self._m3id_t0 = 0
        self._m3id_span = None

    def set_uncond_inputs(self, uncond_inputs) -> None:
        """Hand down this document's image-free inputs (no `pixel_values`)."""
        self._m3id_uncond_inputs = uncond_inputs
        self._m3id_uncond_set = True

    def _resolve_t0(self, input_ids: torch.Tensor) -> int:
        """Offset added to the decode step index (paper Sec. 5.2), measured from
        `img_hi`.

        t0 EXCLUDES image tokens. With `lambda=0.02` the weight is 1.72 at t=50 but
        1.9e11 at t=1300, and Qwen3-VL's dynamic resolution routinely emits thousands of
        image tokens -- measuring from `img_lo` would inflate it by ~11 orders of
        magnitude and make it vary with image size, giving a different method that still
        "runs".
        """
        if self.m3id_t0_override >= 0:
            return int(self.m3id_t0_override)
        if self.m3id_offset_mode == "none" or self._m3id_span is None:
            return 0
        if self.m3id_offset_mode not in OFFSET_MODES:
            raise ValueError(
                f"m3id_offset_mode must be one of {OFFSET_MODES}, "
                f"got {self.m3id_offset_mode!r}."
            )
        return int(input_ids.shape[1]) - int(self._m3id_span[1])

    def _uncond_logits(
        self, next_input_ids: torch.Tensor, is_prefill: bool
    ) -> torch.Tensor:
        """One lockstep step of the text-only stream; returns last-position logits
        `[B,1,V]`.
        """
        uncond = self._m3id_uncond_inputs

        saved_rope_deltas = self.model.rope_deltas  # owned by the MAIN stream
        try:
            if is_prefill:
                self._m3id_cache = DynamicCache(config=self.config)
                self._m3id_len = 0
                input_ids = uncond["input_ids"]
                attention_mask = uncond.get("attention_mask", None)
                seq_len = int(input_ids.shape[1])
                cache_position = torch.arange(seq_len, device=input_ids.device)
                # None + cache_position[0]==0 sends Qwen3VLModel down its own
                # get_rope_index prefill branch; with no image that is the text-only
                # branch. Computed by transformers rather than assumed here.
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
                self.model.rope_deltas = self._m3id_rope_deltas

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=self._m3id_cache,
                pixel_values=None,
                image_grid_thw=None,
                cache_position=cache_position,
                use_cache=True,
            )
            if is_prefill:
                self._m3id_rope_deltas = self.model.rope_deltas
        finally:
            self.model.rope_deltas = saved_rope_deltas

        self._m3id_cache = out.past_key_values
        self._m3id_len += seq_len
        # Only the last position is ever combined.
        return self.lm_head(out.last_hidden_state[:, -1:, :])

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

        # EARNED short-circuit: `expm1(0) == 0.0` exactly, so the combine is a no-op at
        # lambda 0. `_force_compute` disables it so that claim stays testable.
        if self.m3id_lambda == 0.0 and not self._force_compute:
            return main_out

        if not self._m3id_uncond_set:
            raise RuntimeError(
                "M3ID is enabled but no unconditional inputs were handed down for "
                "this sample. The wrapper's `_prepare_inputs` must call "
                "`set_uncond_inputs(...)` after every `reset_sample_tracking(...)`; "
                "a bare `model.generate(...)` cannot use M3ID."
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

        # Text-only prompt: l_u would equal l_c, so the correction is identically zero
        # and the method is undefined. Skipped rather than run.
        if self._m3id_span is None:
            return main_out

        if input_ids is None or input_ids.shape[0] != 1:
            raise NotImplementedError(
                "M3ID requires batch_size 1 and num_beams 1: one unconditional cache "
                "cannot serve several main-stream rows."
            )
        if main_out.logits.shape[1] != 1:
            raise NotImplementedError(
                f"M3ID combines the NEXT-token distribution only, but this forward "
                f"produced {main_out.logits.shape[1]} logit positions. "
                f"Use it through `generate()`."
            )

        uncond_logits = self._uncond_logits(input_ids, is_prefill)

        t_eff = self._m3id_t0 + self._m3id_t
        self._m3id_t += 1
        self.m3id_last_t_eff = t_eff
        self.m3id_last_weight = m3id_weight(t_eff, self.m3id_lambda)
        self.m3id_n_steps += 1
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
