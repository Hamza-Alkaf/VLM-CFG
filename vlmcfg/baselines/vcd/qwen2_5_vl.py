"""Visual Contrastive Decoding (VCD, Leng et al. 2023) ported to Qwen2.5-VL.

The original repo (`vcd_sample.py`) monkey-patches
`transformers.GenerationMixin.sample`, which only fires for `do_sample=True` and
copy-pastes an entire HF generation loop that drifts every time `transformers` changes
its internals. This port instead overrides just `forward()`: the conditional ("main")
and contrastive ("noised") streams run in LOCKSTEP inside a single `generate()` call --
every decode step, HF's loop calls this `forward()` exactly once and feeds back
whichever token the *combined* logits produced, so both streams necessarily see the
identical token history at every step. Each stream keeps its own KV cache; the noised
stream's cache is (re)built from scratch whenever the main stream is at its own prefill
step, and advanced in lockstep otherwise. This makes the noised stream a real second
forward pass on the same prefix, not the pre-computed-independent-trajectory
approximation this project's own CFG pipeline uses for its negative streams.

`pixel_values` is only present on the prefill call (`transformers` drops it afterwards,
see `Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation`), and
`position_ids`/ `attention_mask` are identical between the two streams (same image grid,
same token layout -- noising the image never changes its shape), so they are simply
reused rather than re-derived.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

from vlmcfg.baselines.vcd.combine import contrastive_combine
from vlmcfg.baselines.vcd.noise import add_diffusion_noise


def _noised_pixel_values(
    pixel_values: torch.Tensor, noise_step: int, vision_config=None
) -> torch.Tensor:
    """Apply VCD's diffusion noise to Qwen's (already patchified) pixel tensor.

    `add_diffusion_noise` is imported unmodified from `vcd_add_noise.py`; it builds its
    DDPM schedule on CPU float32 regardless of the input's device/dtype, so the input is
    cast down before the call and the result cast back. The noising math
    (`sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise`) is elementwise, but
    WHERE the independent noise draws land is not a free choice here:

    Qwen's processor does not hand us an image, it hands us a patch tensor of shape
    `(n_patches, C * temporal_patch_size * ph * pw)` whose per-row layout is
    `(C, T, ph, pw)` (see `image_processing_qwen2_vl._preprocess`'s reshape/transpose).
    For a SINGLE image `temporal_patch_size=2` and the processor DUPLICATES the frame to
    fill it, so `T=0` and `T=1` hold identical copies of the same pixels. Drawing
    independent noise per element would therefore corrupt the two copies differently --
    and since `patch_embed` is a `Conv3d` with a `(2, ph, pw)` kernel spanning exactly
    that temporal axis, the two independent draws get averaged, shrinking the noise
    actually reaching the vision transformer by ~sqrt(2) relative to the paper (which
    noises the IMAGE, so both copies necessarily share one draw).

    So the noise is drawn on a single temporal slice and broadcast across `T`, which
    reproduces "noise the image, then patchify" exactly while still giving every pixel
    `(patch, c, ph, pw)` its own independent draw.

    Falls back to the flat elementwise behaviour (with a warning) if the tensor does not
    factor into the expected layout -- e.g. a video, where the temporal frames are
    genuinely distinct and per-frame independent noise is the correct thing anyway.
    """
    orig_dtype, orig_device = pixel_values.dtype, pixel_values.device
    flat = pixel_values.detach().float().cpu()

    t_patch = getattr(vision_config, "temporal_patch_size", None)
    patch = getattr(vision_config, "patch_size", None)
    channels = getattr(vision_config, "in_channels", None)
    expected = None
    if None not in (t_patch, patch, channels):
        expected = channels * t_patch * patch * patch

    if (
        expected is not None
        and flat.ndim == 2
        and flat.shape[-1] == expected
        and t_patch > 1
    ):
        n = flat.shape[0]
        rows = flat.view(n, channels, t_patch, patch, patch)
        # Identical temporal copies must share one noise draw: noise slice T=0, then
        # expand.
        noised_slice = add_diffusion_noise(rows[:, :, :1, :, :], noise_step)
        noisy = noised_slice.expand(n, channels, t_patch, patch, patch).reshape(
            n, expected
        )
    else:
        if expected is not None and flat.ndim == 2 and flat.shape[-1] != expected:
            print(
                f"[vcd] pixel_values last dim {flat.shape[-1]} != expected {expected} "
                f"for the (C,T,ph,pw) layout; falling back to flat elementwise noise.",
                flush=True,
            )
        noisy = add_diffusion_noise(flat, noise_step)

    return noisy.to(device=orig_device, dtype=orig_dtype)


class Qwen2_5_VLForConditionalGenerationVCD(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with VCD's image-noise contrastive decoding.

    `cd_alpha`/`cd_beta`/`noise_step` are plain instance attributes (defaulted to VCD's
    neutral point below), set by the lmms-eval wrapper model
    (`vcd_lmms_plugin/models/qwen2_5_vl_vcd.py`) right after `from_pretrained` loads
    this class -- the same "configure post-construction" pattern this project already
    uses for e.g. `attn_intervene_*` (see `qwen2_5_vl_attn_intervene.py`), rather than
    threading extra kwargs through `from_pretrained`.
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
    noise_step: int = 500

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
            impaired_pixel_values = (
                None
                if pixel_values is None
                else _noised_pixel_values(
                    pixel_values,
                    self.noise_step,
                    vision_config=getattr(self.config, "vision_config", None),
                )
            )
        else:
            # `pixel_values` is already None on every non-prefill step (see
            # `prepare_inputs_for_generation`); nothing to noise.
            impaired_pixel_values = pixel_values

        aux_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self._aux_cache,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            pixel_values=impaired_pixel_values,
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
