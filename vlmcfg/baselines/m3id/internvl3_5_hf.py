"""InternVL3 with M3ID (Favero et al., CVPR 2024) -- image-free second stream, no DPO.

The operator and its combine (`m3id_combine.py`) are shared verbatim with the Qwen2.5-VL
port; only the second stream's plumbing changes. `m3id_combine` is pure tensor math and
is imported, not reimplemented, so the two backbones cannot drift apart in the formula.

THE HARDEST PART OF THE QWEN PORT DOES NOT EXIST HERE. On Qwen2.5-VL the image-free
stream has to borrow and restore `model.rope_deltas` inside a `try/finally`, because
mRoPE state is a single attribute on the shared inner model owned by the MAIN stream;
dropping the restore silently corrupts the conditional stream's positions from its
second token onward and produces fluent, wrong text with no error anywhere. InternVL3's
text tower is Qwen3, which uses plain 1-D RoPE: there is no `get_rope_index`, no
`rope_deltas` attribute and no mRoPE section
(`grep -n "rope_deltas\\|get_rope_index\\|mrope" modeling_internvl.py` returns nothing).
Positions are `cache_position.unsqueeze(0)`, derived per call. So the second stream
needs only its own cache, its own length counter and its own attention mask -- three
locals, no shared mutable state and nothing to restore.

WHAT STILL MATTERS: `t0` EXCLUDES THE IMAGE TOKENS. M3ID's weight is
`expm1(lambda * t)`, and InternVL emits `image_seq_length * n_tiles` = 256 x up to 13 =
up to 3328 image tokens per prompt. Measuring `t0` from the START of the image span
rather than its end would push `0.02 * t` from ~1 to ~66, i.e. inflate the weight by ~28
orders of magnitude, and make it vary with the tile count (that is, with image aspect
ratio). Unit test M6 pins `t0 == prompt_len - img_hi`.
"""

import torch
from transformers import InternVLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.common.backbone import InternVL3Adapter
from vlmcfg.baselines.m3id.combine import m3id_combine, m3id_weight

OFFSET_MODES = ("tokens_to_image", "none")


class InternVLForConditionalGenerationM3ID(InternVLForConditionalGeneration):
    """`InternVLForConditionalGeneration` + M3ID, configured post-construction."""

    m3id_lambda: float = 0.0
    m3id_alpha: float = 0.3
    m3id_offset_mode: str = "tokens_to_image"
    m3id_t0_override: int = -1
    m3id_max_weight: float = 0.0
    _force_compute: bool = False

    def __init__(self, config):
        super().__init__(config)
        self._adapter = InternVL3Adapter()
        self.reset_sample_tracking()
        self.m3id_n_gate_fired = 0
        self.m3id_n_steps = 0
        self.m3id_last_t_eff = 0
        self.m3id_last_weight = 0.0
        self.m3id_last_gate_fired = False
        self.m3id_max_p: list = []

    def reset_sample_tracking(self, doc_id=None) -> None:
        """Drop the previous document's unconditional bundle and stream state.

        Clearing `_m3id_uncond_set` is a TRIPWIRE, not housekeeping: without it a wiring
        bug that skipped `set_uncond_inputs` would silently contrast this document's
        image against the PREVIOUS document's question, which no metric would reveal.
        """
        self._m3id_uncond_inputs = None
        self._m3id_uncond_set = False
        self._m3id_cache = None
        self._m3id_len = 0
        self._m3id_span: tuple[int, int] | None = None
        self._m3id_t = 0
        self._m3id_t0 = 0

    def set_uncond_inputs(self, uncond_inputs) -> None:
        """Hand down this document's image-free inputs (no `pixel_values`), already on
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
        if is_prefill:
            self._m3id_cache = DynamicCache(config=self.config)
            self._m3id_len = 0
            input_ids = uncond["input_ids"]
            attention_mask = uncond.get("attention_mask", None)
            seq_len = int(input_ids.shape[1])
            cache_position = torch.arange(seq_len, device=input_ids.device)
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

        out = self.model(
            input_ids=input_ids,
            pixel_values=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=self._m3id_cache,
            cache_position=cache_position,
            use_cache=True,
        )
        self._m3id_cache = out.past_key_values
        self._m3id_len += seq_len
        # Only the last position is ever combined, so running lm_head over just that row
        # (rather than over the whole prefill) is the same value for far less compute.
        return self.lm_head(out.last_hidden_state[:, -1:, :])

    def _resolve_t0(self, input_ids: torch.Tensor) -> int:
        """The offset added to the decode step index, per the paper's Sec. 5.2.

        The weight `expm1(lambda*t)` is EXACTLY ZERO at `t = 0`, so on a single-token
        answer M3ID is a mathematical no-op. The paper hits this on POPE and fixes it by
        offsetting: "we take an offset into account when using M3ID on POPE, and select
        t = t0 where t0 is the number of tokens in between the output and the image."
        Every benchmark here is short-answer VQA of that shape, so `tokens_to_image` is
        the default; `none` reproduces the captioning regime and exists as the ablation
        that makes the degeneracy visible rather than hidden.
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

        # EARNED short-circuit: legitimate only because the unguarded arithmetic has
        # been shown bit-identical. `_force_compute` disables it so that claim stays
        # testable.
        if self.m3id_lambda == 0.0 and not self._force_compute:
            return main_out

        if not self._m3id_uncond_set:
            raise RuntimeError(
                "M3ID is enabled but no unconditional inputs were handed down for "
                "this sample. `InternVL3_M3ID._prepare_inputs` must call "
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
            self._m3id_span = self._adapter.image_span(
                input_ids, int(self.config.image_token_id)
            )
            self._m3id_t0 = self._resolve_t0(input_ids)

        # Text-only prompt: `l_u` would be `l_c`, so the correction is identically zero
        # and the method is undefined. Skipped rather than run.
        if self._m3id_span is None:
            return main_out

        if input_ids is None or input_ids.shape[0] != 1:
            raise NotImplementedError(
                f"M3ID requires batch_size 1 and num_beams 1 (got a batch of "
                f"{None if input_ids is None else input_ids.shape[0]}): HF expands "
                "the batch to num_beams, and one unconditional cache cannot serve "
                "several main-stream rows."
            )
        if main_out.logits.shape[1] != 1:
            raise NotImplementedError(
                f"M3ID combines the NEXT-token distribution only, but this forward "
                f"produced {main_out.logits.shape[1]} logit positions. Use it through "
                "`generate()`; teacher-forced scoring is not supported."
            )

        uncond_logits = self._uncond_logits(input_ids, is_prefill)

        t_eff = self._m3id_t0 + self._m3id_t
        self._m3id_t += 1
        self.m3id_last_t_eff = t_eff
        self.m3id_last_weight = m3id_weight(t_eff, self.m3id_lambda)
        self.m3id_n_steps += 1
        # Recorded before combining: this is the quantity the gate thresholds, and the
        # only way to tell "the model was confident" from "the gate is broken".
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
