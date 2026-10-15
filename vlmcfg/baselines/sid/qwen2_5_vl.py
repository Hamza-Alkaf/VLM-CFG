"""SID (Self-Introspective Decoding, Huo et al. 2024, arXiv:2408.02032) ported to
Qwen2.5-VL.

WHAT SID DOES
-------------
SID is a contrastive-decoding method whose "amateur" stream is not a noised image (VCD)
or a shallow exit (LayerCD) but a *self-introspected* one: the model itself is asked
which image tokens it currently cares LEAST about, and the amateur stream is then forced
to look only at those. Upstream calls this CT2S (Context and Text-aware Token
Selection). Concretely, for every forward pass:

  1. run the normal ("main") pass;
  2. run a second ("amateur") pass in which, at decoder layer K (`fast_v_agg_layer`,
     upstream default 2):
       * take layer K-1's attention, averaged over heads, LAST QUERY ROW only,
         restricted to the image-token columns;
       * select the `fast_v_attention_rank` (upstream default 100) image tokens with the
         LOWEST attention -- upstream literally writes
         `...topk(ATTENTION_RANK, largest=False)` (see
         `transformers/src/transformers/models/llama/modeling_llama.py` line ~792). The
         least-attended tokens are the point of the method: they amplify hallucination,
         which is what makes the amateur stream worth contrasting away;
       * for every layer >= K, mask out every OTHER image token in the attention mask.
         Non-image tokens (system prompt, instruction, generated tokens) are always
         kept;
  3. combine the two logit streams with the standard contrastive-decoding formula.

The selection is recomputed INSIDE EVERY FORWARD PASS, not fixed once at prefill. That
is what "context and text aware" means: as generation proceeds, the last query row is a
different token and the least-attended set adapts to it. This port preserves that (the
selection state is reset at the top of each amateur forward, and layer K recomputes it).

THE COMBINATION FORMULA IS VCD'S, LITERALLY
-------------------------------------------
SID's reference `vcd_sample.py` (lines 175-185) computes

    cutoff = log(cd_beta) + next_token_logits.max(-1, keepdim=True).values
    diffs  = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
    result = diffs.masked_fill(next_token_logits < cutoff, -inf)

which is character for character what VCD does. So this module reuses
`vlmcfg.baselines.vcd.combine.contrastive_combine` rather than growing another copy of
it (LayerCD's port does the same).

WHY `output_attentions=True` IS NOT USED (the eager trap)
--------------------------------------------------------
Step 2 needs layer K-1's last-row attention. Upstream gets it from `layer_outputs[1]`,
i.e. by running the whole model with `output_attentions=True`. On this codebase that
would be a silent correctness bug, not just a memory one: `output_attentions=True`
forces `attn_implementation="eager"`, and this repo has MEASURED that eager materially
changes Qwen2.5-VL's outputs -- `max|dlogit| ~ 1.22` against sdpa/flash on a fixed
prompt, worth ~6 points of textvqa accuracy (see
`vlmcfg/baselines/common/lastrow_attn.py`). An SID arm computed under eager would not be
comparable to the sdpa baseline it is meant to be compared against.

Instead this port reuses the repo's own `LastRowAttentionCapture`, which computes
exactly the ONE attention row SID reads (project q, apply mRoPE, read the
already-RoPE'd, already-cache-accumulated K straight out of the `Cache` object) while
the real forward stays on the model's configured fused kernel. Softmax is
row-independent, so this is an exact refactor of the row upstream reads, not an
approximation. It is installed on layer K-1 ONLY, and only for the duration of the
amateur forward.

WHY `fast_v_inplace=True` IS NOT IMPLEMENTED
--------------------------------------------
Upstream ships two CT2S implementations behind `fast_v_inplace`. The `True` one
physically prunes hidden states and rebuilds `position_ids`; its own comment says
"KV-Cache not supported", and dropping tokens mid-sequence would break Qwen2.5-VL's
mRoPE, whose per-token (t, h, w) position triples are derived from the image grid and
cannot be re-indexed by a `keep_indexs` gather without changing what every surviving
token thinks its spatial position is. The `False` variant -- upstream's own DEFAULT, and
the one their `pope_eval.py` runs -- keeps the sequence intact and masks the
non-selected image tokens out of the attention mask. That is the one implemented here,
so nothing is lost relative to how SID is actually run.

`fast_v_attention_rank_add` IS NOT IMPLEMENTED, because it is dead in this path: in
`modeling_llama.py` it is read only inside the `vad == True` branch (the "vision add /
enhancement" experiment), and `vcd_sample.py` passes `vad=False` on both the main and
the contrastive forward, with the only `vad=True` call site commented out. Adding the
knob would be adding a control that does nothing.

TOKEN COUNT: THE ONE PLACE THE PAPER'S NUMBER MAY NOT TRANSFER
---------------------------------------------------------------
`fast_v_attention_rank=100` was tuned on LLaVA-1.5, whose vision tower emits a FIXED 576
tokens per image -- so 100 is always ~17% of the image. Qwen2.5-VL is
dynamic-resolution: the image-token count is a function of the image's own size and
`max_pixels`, and routinely ranges from a couple of hundred to several thousand within a
single benchmark. A fixed 100 is therefore a different fraction of the image on every
sample. The paper-faithful absolute value is kept as the DEFAULT here (these runs exist
to compare against SID as published); `sid_attention_rank_ratio` exposes the
fraction-of-image alternative explicitly and is off (`None`) by default. If the
requested count exceeds the actual image-token count, it is clamped to keep all of them
rather than crashing.
"""

import torch
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.common.lastrow_attn import LastRowAttentionCapture
from vlmcfg.baselines.common.tokens import (
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
)
from vlmcfg.baselines.vcd.combine import contrastive_combine

# Additive pre-softmax score meaning "this key column is unreachable". Same constant,
# and same reasoning, as `vlmcfg/negatives/attention.py`: NOT -inf, because a row whose
# every column is -inf produces NaN out of softmax, whereas a large finite sentinel
# degrades to a ~0 probability. That cannot happen here anyway (only image columns are
# ever killed, and every row still sees the system-prompt tokens), but -1e30 fails safe.
# |A| is single digits for real attention logits, so exp() underflows to exactly 0.
NEG_INF = -1e30


def neg_inf(dtype):
    """Representable additive-mask fill; see common.backbone.neg_inf."""
    return torch.finfo(dtype).min / 2


def find_image_span(input_ids: torch.Tensor) -> tuple | None:
    """Locate the image-token span `[img_lo, img_hi)` from
    `<|vision_start|>`/`<|vision_end|>`.

    Same convention as `model_utils/attn_intervene_context.py` (`img_lo = start + 1`,
    `img_hi = end`), and the token ids are imported from `att_grounding` rather than
    hardcoded. Vectorised rather than reusing `find_pattern_starts`, which scans
    position by position with a `torch.equal` per index -- one GPU sync per token, on
    prompts that are routinely thousands of tokens long.

    Returns `None` for a text-only prompt (no vision markers). Raises for a multi-image
    prompt, which CT2S has no single well-defined image span for.
    """
    ids = input_ids[0]
    starts = (ids == VISION_START_TOKEN_ID).nonzero(as_tuple=True)[0]
    ends = (ids == VISION_END_TOKEN_ID).nonzero(as_tuple=True)[0]
    if starts.numel() == 0 or ends.numel() == 0:
        return None
    if starts.numel() != 1 or ends.numel() != 1:
        raise ValueError(
            "qwen2_5_vl_sid supports a single image per prompt; found "
            f"{int(starts.numel())} <|vision_start|> / "
            f"{int(ends.numel())} <|vision_end|> markers."
        )
    return int(starts[0]) + 1, int(ends[0])


class Ct2sImageMask:
    """Mask every image token except the least-attended ones, for layers >= `agg_layer`.

    Wraps `self_attn.forward` per layer -- the technique `vlmcfg/negatives/attention.py`
    uses -- rather than passing a 2D padding mask into the model, because a 2D mask
    would apply to EVERY layer, whereas CT2S only masks from layer K onwards (layers < K
    must see the whole image, which is what makes layer K-1's attention row meaningful
    in the first place).

    The selection itself is computed lazily, at the first wrapped layer of each forward
    pass, from the row `capture` harvested at layer K-1 during that same pass -- layers
    run in order, so by the time layer K's wrapper executes, layer K-1's row is already
    there. It is then reused for layers > K within that pass, matching upstream's `else:
    new_attention_mask = gen_attention_mask`.
    """

    def __init__(
        self,
        img_lo: int,
        img_hi: int,
        rank: int,
        agg_layer: int,
        capture: LastRowAttentionCapture,
    ) -> None:
        self.img_lo = img_lo
        self.img_hi = img_hi
        self.rank = rank
        self.agg_layer = agg_layer
        self.capture = capture
        self._patched: dict[nn.Module, object] = {}
        self._kill_cols: torch.Tensor | None = None
        # Per-run bookkeeping the smoke tests assert on: without it, "the selection ran
        # and kept the intended number of tokens" is not observable from outside.
        self.n_selected: int = 0
        self.n_image_tokens: int = img_hi - img_lo

    def reset_pass(self) -> None:
        """Drop the previous forward's selection AND the row it was derived from.

        Both halves matter. Keeping a stale `_kill_cols` would freeze the selection at
        prefill, turning SID into a static token-drop and destroying the "text aware"
        half of CT2S. Keeping a stale captured row would let a forward in which layer
        K-1 somehow did not fire silently reuse the previous step's attention.
        """
        self._kill_cols = None
        self.capture.rows.clear()

    def _select(self, device: torch.device) -> torch.Tensor:
        """Absolute key-column indices to mask, from layer K-1's head-averaged last
        row."""
        row = self.capture.rows.get(self.agg_layer - 1)
        if row is None:
            raise RuntimeError(
                f"SID: layer {self.agg_layer - 1}'s attention row was not captured "
                f"before layer {self.agg_layer} ran; CT2S cannot select tokens."
            )
        n_img = row.shape[-1]
        # Clamp rather than crash: Qwen2.5-VL's image-token count is per-sample, so a
        # small image can legitimately have fewer tokens than the requested rank.
        # Keeping all of them is the graceful degradation (the mask then kills nothing).
        k = min(self.rank, n_img)
        # `largest=False` -- the LEAST-attended tokens. This is SID, not FastV: upstream
        # deliberately inverted FastV's top-k to build a hallucination-amplified stream.
        keep = torch.topk(row, k, largest=False).indices
        kill = torch.ones(n_img, dtype=torch.bool)
        kill[keep] = False
        self.n_selected = int(k)
        self.n_image_tokens = int(n_img)
        return (kill.nonzero(as_tuple=True)[0] + self.img_lo).to(device)

    def install(self, layers: nn.ModuleList) -> None:
        for layer in layers:
            attn = layer.self_attn
            layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None or layer_idx < self.agg_layer:
                continue
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn.forward)

    def _make_wrapper(self, original):
        def wrapper(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            cache_position = kwargs.get("cache_position")
            if hidden_states is None or cache_position is None:
                # Nothing to index against; leave the layer exactly as it was rather
                # than guessing. (Qwen2_5_VLDecoderLayer always passes both as
                # keywords.)
                return original(*args, **kwargs)

            if self._kill_cols is None:
                self._kill_cols = self._select(hidden_states.device)

            kwargs["attention_mask"] = self._biased_mask(
                kwargs.get("attention_mask"), hidden_states, cache_position
            )
            return original(*args, **kwargs)

        return wrapper

    def _biased_mask(
        self,
        mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """Return a 4D additive mask with the non-selected image columns pushed to -inf.

        `mask` arrives as `None` on the sdpa path whenever the prompt is unpadded,
        because `create_causal_mask` elides it and lets
        `F.scaled_dot_product_attention`'s own `is_causal=True` shortcut supply
        causality. Handing sdpa a non-None mask turns that shortcut off, so the causal
        structure has to be rebuilt here -- from `cache_position` (the query rows'
        ABSOLUTE positions), which is exactly what transformers' own mask builder uses.

        Every query row is masked, not just the last: upstream builds one full decoder
        mask at layer K and reuses it for all rows and all deeper layers.
        """
        q_pos = cache_position.view(-1, 1)
        kv_len = int(cache_position[-1]) + 1
        if mask is None:
            k_idx = torch.arange(kv_len, device=cache_position.device).view(1, -1)
            blocked = k_idx > q_pos
            mask = torch.zeros(
                blocked.shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            mask = mask.masked_fill(blocked, neg_inf(mask.dtype)).view(
                1, 1, *blocked.shape
            )
            mask = mask.expand(hidden_states.shape[0], 1, *blocked.shape).clone()
        else:
            # Never mutate the model's own mask tensor: it is shared by every layer in
            # the stack (`causal_mask_mapping[...]` is built once per forward), so an
            # in-place edit here would leak CT2S's masking back into layers < K.
            mask = mask[:, :, :, :kv_len].clone()
        mask[:, :, :, self._kill_cols] += neg_inf(mask.dtype)
        return mask

    def remove(self) -> None:
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()


class Qwen2_5_VLForConditionalGenerationSID(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with SID's self-introspective contrastive decoding.

    The knobs are plain instance attributes (defaulted to SID's neutral point below),
    set by the lmms-eval wrapper model (`sid_lmms_plugin/models/qwen2_5_vl_sid.py`)
    right after `from_pretrained` loads this class -- the same "configure
    post-construction" pattern this project already uses for `qwen2_5_vl_vcd` /
    `qwen2_5_vl_vti` / `attn_intervene_*`, rather than threading extra kwargs through
    `from_pretrained`.
    """

    # Upstream names in comments; the paper's own defaults are 1 / 0.1 / 2 / 100, but
    # the first two are defaulted to 0 HERE so that the NEUTRAL point is what you get
    # with no flags -- the wrapper's docstring explains the choice.
    # EARNED-GUARD BYPASS. The neutral guards below are an optimisation, not the proof
    # that a method is neutral: with the guard in place, `max|dlogit| == 0` only
    # demonstrates that the BRANCH works, and an implementation whose arithmetic is
    # wrong passes it untouched. Setting `_force_compute = True` runs the whole method
    # with its strength at zero, so the neutral test measures the real arithmetic. Kept
    # permanently so that check stays a repeatable regression test rather than something
    # done once during development.
    _force_compute: bool = False
    sid_cd_alpha: float = 0.0  # upstream `cd_alpha`   (paper 1)
    sid_cd_beta: float = 0.0  # upstream `cd_beta`     (paper 0.1)
    sid_agg_layer: int = 2  # upstream `fast_v_agg_layer`      (paper's K)
    sid_attention_rank: int = (
        100  # upstream `fast_v_attention_rank` (paper's absolute count)
    )
    sid_attention_rank_ratio: float | None = (
        None  # this port's dynamic-resolution alternative; None = paper-faithful
    )

    def __init__(self, config):
        super().__init__(config)
        self._aux_cache = None
        self._sid_span = None
        # Last amateur pass's CT2S bookkeeping, for the effect smoke test and for job
        # logs.
        self.sid_last_n_selected: int | None = None
        self.sid_last_n_image_tokens: int | None = None

    def reset_sample_tracking(self, doc_id=None):
        """No-op: `chat/qwen2_5_vl.py::generate_until` calls this unconditionally on
        every `model_fn` at the start of each chunk (regardless of `cfg_mode`), because
        the CFG model class uses it to flush per-sample bookkeeping. This wrapper needs
        none: both `_aux_cache` and `_sid_span` reset themselves inside `forward()`
        whenever the main stream is at its own prefill step (`cache_position[0] == 0`).
        """

    def _resolve_rank(self, n_image_tokens: int) -> int:
        """The number of image tokens the amateur stream is allowed to keep.

        Paper-faithful by default: the absolute `sid_attention_rank`, clamped down to
        `n_image_tokens` when the image is small. `sid_attention_rank_ratio`, when set,
        overrides it with `round(ratio * n_image_tokens)` -- see this module's docstring
        for why that alternative exists and why it is not the default. Never returns 0:
        masking the entire image would make the amateur stream image-blind, which is
        VCD/LayerCD's degradation, not SID's.
        """
        if self.sid_attention_rank_ratio is not None:
            return max(
                1,
                min(
                    n_image_tokens,
                    int(round(float(self.sid_attention_rank_ratio) * n_image_tokens)),
                ),
            )
        return max(1, min(n_image_tokens, int(self.sid_attention_rank)))

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

        # Neutral point: skip the amateur forward entirely, install no wrappers, capture
        # no attention -- so the returned logits are bit-for-bit the base model's, not
        # merely numerically close to them.
        if (
            self.sid_cd_alpha == 0.0
            and self.sid_cd_beta == 0.0
            and not self._force_compute
        ):
            return main_out

        is_prefill = cache_position is None or int(cache_position[0]) == 0
        if is_prefill:
            self._aux_cache = DynamicCache(config=self.config)
            # The image span is a property of the PROMPT, so it is resolved once, at
            # prefill, while `input_ids` still holds the whole sequence. On decode steps
            # `input_ids` is a single new token and the markers are long gone.
            self._sid_span = None if input_ids is None else find_image_span(input_ids)

        if self._sid_span is None:
            # Text-only prompt: CT2S has no image tokens to select from, so there is no
            # amateur stream to contrast against. Returning the main logits unchanged is
            # the only faithful thing -- SID is undefined without an image.
            return main_out

        img_lo, img_hi = self._sid_span
        rank = self._resolve_rank(img_hi - img_lo)

        layers = self.model.language_model.layers
        if not (0 < self.sid_agg_layer < len(layers)):
            raise ValueError(
                f"sid_agg_layer must be in [1, {len(layers) - 1}] (upstream asserts "
                "K > 0, and layer K-1's attention must exist); got "
                f"{self.sid_agg_layer!r}."
            )

        capture = LastRowAttentionCapture(img_lo, img_hi)
        ct2s = Ct2sImageMask(img_lo, img_hi, rank, self.sid_agg_layer, capture)
        ct2s.reset_pass()
        # Layer K-1 only: that is the single row CT2S reads. Installing the capture on
        # the whole stack would cost one extra row per layer for nothing.
        capture.install([layers[self.sid_agg_layer - 1]])
        ct2s.install(layers)
        try:
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
        finally:
            # Uninstall unconditionally, and BEFORE anything can raise past this point:
            # the main stream must never run with CT2S wrappers or the capture attached.
            ct2s.remove()
            capture.remove()

        self._aux_cache = aux_out.past_key_values
        self.sid_last_n_selected = ct2s.n_selected
        self.sid_last_n_image_tokens = ct2s.n_image_tokens

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        aux_logits = self.lm_head(aux_out.last_hidden_state[:, slice_indices, :])

        main_out.logits = contrastive_combine(
            main_out.logits,
            aux_logits,
            self.sid_cd_alpha,
            self.sid_cd_beta,
            force=self._force_compute,
        )
        return main_out
