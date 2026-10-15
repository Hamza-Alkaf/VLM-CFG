"""Edits to how text tokens attend to the image,
for decoder layers in [layer_lo, layer_hi).

Used two ways: on the negative stream, to weaken the model's access to the image instead
of deleting it (MUL, SEG, Over-blur, Soft PAG / PAG, Masked; Table "attention variants"
of the paper), and on the conditional stream, to steer attention towards an annotated
region.

Every edit keeps each layer on the model's real fused kernel and corrects only the text
rows (every query row after the image: the question, the generation prompt and each
decode step) afterwards, with the difference of two identical recomputes:

    delta    = o_proj(softmax(A + bias) @ V) - o_proj(softmax(A) @ V)
    out[rows] = real_kernel_out[rows] + delta

Both branches run the same fp32 ops on the same tensors, so the recompute noise cancels
and a zero bias is a bitwise no-op. Requires batch size 1 and a real KV cache.
"""

import math
from collections.abc import Callable

import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv

from vlmcfg.baselines.common.rope_compat import apply_qwen_rope

# Additive score for "unreachable": finite so a fully masked row cannot produce NaN.
NEG_INF = -1e30

#   mul     A * (1 + gamma)                  (MUL; gamma < 0 flattens)
#   add     A + gamma                        a uniform shift (with a region split)
#   abs     A + gamma * |A|                  the original operator
#   blur    G_sigma * A over the patch grid  (SEG)
#   sharpen A_blur + alpha (A - A_blur)      (Over-blur at alpha < 0)
#   peak    mix towards a one-hot on the row's top image column, in probability space
#           (Soft PAG; lambda = 1 is PAG)
INTERVENTION_OPS = ("mul", "add", "abs", "blur", "peak", "sharpen")
STRUCTURAL_OPS = ("blur", "peak", "sharpen")

#   global          one softmax over all columns; the image-vs-text mass may move
#   preserve_ratio  the image block is rescaled back to its unedited mass
#   cap_ratio       like global, but the image mass may not rise above img_cap_ratio
#                   (when the unedited mass is already above it, preserve_ratio applies)
RENORM_MODES = ("global", "preserve_ratio", "cap_ratio")

# Query rows per block, bounding the [H, rows, S_kv] score tensors.
ROW_CHUNK = 512


def neg_inf(dtype: torch.dtype) -> float:
    """A mask fill representable in `dtype` that stays finite when added twice."""
    return torch.finfo(dtype).min / 2


def _attend_and_project(
    attn: nn.Module, probs: torch.Tensor, value_f32: torch.Tensor, bsz: int
) -> torch.Tensor:
    """`o_proj(probs @ V)` for a block of query rows, entirely in fp32.

    Shared by both branches of the delta so that they run provably identical ops.
    """
    out = torch.matmul(probs, value_f32)  # [B, H, R, D]
    out = out.transpose(1, 2).contiguous().reshape(bsz, probs.shape[2], -1)
    return nn.functional.linear(out, attn.o_proj.weight.float(), None)


def _rescale_to_target(
    p: torch.Tensor, img_target: torch.Tensor, img_lo: int, img_hi: int
) -> torch.Tensor:
    """Rescale `p` so its image block sums to `img_target`
    and the rest to 1 - target."""
    img_current = p[:, :, :, img_lo:img_hi].sum(dim=-1, keepdim=True)
    non_target = 1.0 - img_target
    non_current = 1.0 - img_current
    tiny = torch.finfo(torch.float32).tiny
    img_scale = torch.where(
        img_current > 0,
        img_target / img_current.clamp_min(tiny),
        torch.ones_like(img_current),
    )
    non_scale = torch.where(
        non_current > 0,
        non_target / non_current.clamp_min(tiny),
        torch.ones_like(non_current),
    )
    out = p * non_scale
    out[:, :, :, img_lo:img_hi] = p[:, :, :, img_lo:img_hi] * img_scale
    return out


def _gaussian_kernel1d(sigma: float, device, dtype=torch.float32) -> torch.Tensor:
    """Normalised 1-D Gaussian, truncated at 3 sigma."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _blur_image_block(
    a_img: torch.Tensor, sigma: float, grid_h: int, grid_w: int
) -> torch.Tensor:
    """SEG: 2-D Gaussian blur of the image-column scores over their patch grid.

    Normalised convolution at the border (each output is a weighted mean over in-bounds
    neighbours), so the block's mean barely moves while its variance collapses.
    """
    B, H, R, n_img = a_img.shape
    if grid_h * grid_w != n_img:
        raise ValueError(
            f"op='blur' needs the image token grid: got {grid_h}x{grid_w} "
            f"for {n_img} tokens"
        )
    k = _gaussian_kernel1d(sigma, a_img.device, a_img.dtype)
    pad = (k.numel() - 1) // 2

    def _sep_blur(t: torch.Tensor) -> torch.Tensor:
        t = nn.functional.conv2d(
            nn.functional.pad(t, (pad, pad, 0, 0)), k.view(1, 1, 1, -1)
        )
        return nn.functional.conv2d(
            nn.functional.pad(t, (0, 0, pad, pad)), k.view(1, 1, -1, 1)
        )

    x = a_img.reshape(B * H * R, 1, grid_h, grid_w)
    ones = _sep_blur(
        torch.ones((1, 1, grid_h, grid_w), device=a_img.device, dtype=a_img.dtype)
    )
    return (_sep_blur(x) / ones).reshape(B, H, R, n_img)


def _sharpen_image_block(
    a_img: torch.Tensor, sigma: float, grid_h: int, grid_w: int, alpha: float
) -> torch.Tensor:
    """A_blur + alpha (A - A_blur); alpha < 0 extrapolates past the blur (Over-blur).

    alpha == 1 returns the input itself, so it is a bitwise no-op.
    """
    if alpha == 1.0:
        return a_img
    blurred = _blur_image_block(a_img, sigma, grid_h, grid_w)
    return blurred + alpha * (a_img - blurred)


def _peak_image_block(a_img: torch.Tensor, lam: float) -> torch.Tensor:
    """Soft PAG: mix the image attention towards a one-hot on its top column.

    The mix is done on probabilities, p_mix = (1 - lam) p + lam onehot, and mapped back
    with a log; lam = 1 (PAG) keeps the top column's score and sends the rest to
    NEG_INF.
    """
    if lam == 0.0:
        return a_img
    idx = a_img.argmax(dim=-1, keepdim=True)
    if lam >= 1.0:
        peaked = torch.full_like(a_img, NEG_INF)
        peaked.scatter_(-1, idx, a_img.gather(-1, idx))
        return peaked
    p = nn.functional.softmax(a_img, dim=-1)
    p_mix = (1.0 - lam) * p
    p_mix.scatter_add_(-1, idx, torch.full_like(idx, lam, dtype=p_mix.dtype))
    return torch.log(p_mix)


def _text_rows(q_len: int, row_offset: int, img_hi: int) -> torch.Tensor:
    """Local indices of this forward's query rows at or after the end of the image."""
    absolute = torch.arange(row_offset, row_offset + q_len, dtype=torch.long)
    return (absolute >= img_hi).nonzero().flatten()


def parse_layer_range(spec: str, num_layers: int) -> tuple[int, int]:
    """Parse "[start,end]" (half-open) into clamped bounds;
    "-" may replace the comma."""
    spec = spec.strip()
    if not (spec.startswith("[") and spec.endswith("]")):
        raise ValueError(f"layer range must be '[start,end]', got {spec!r}")
    parts = [p.strip() for p in spec[1:-1].replace("-", ",").split(",")]
    if len(parts) != 2:
        raise ValueError(f"invalid layer range {spec!r} (expected '[start,end]')")
    start = max(0, min(int(parts[0]), num_layers))
    end = max(0, min(int(parts[1]), num_layers))
    if not start < end:
        raise ValueError(f"empty layer range: {spec!r} resolved to [{start}, {end})")
    return start, end


class AttentionIntervention:
    """Install/remove an edit of the image-column attention scores on a layer range.

    `region_mask` (bool over the image tokens) splits the image into two gammas: `gamma`
    inside the region, `gamma2` outside; None applies `gamma` to the whole image.
    `ablate` instead removes every image column from the softmax (Masked).
    """

    def __init__(
        self,
        layer_lo: int,
        layer_hi: int,
        img_lo: int,
        img_hi: int,
        gamma: float,
        gamma2: float = 0.0,
        region_mask: "torch.Tensor | None" = None,
        ablate: bool = False,
        op: str = "mul",
        renorm: str = "global",
        img_cap_ratio: float = 1.0,
        blur_sigma: float = 0.0,
        attn_cfg_alpha: float = 1.0,
        peak_lambda: float = 0.0,
        img_grid: "tuple[int, int] | None" = None,
    ) -> None:
        if op not in INTERVENTION_OPS:
            raise ValueError(f"op must be one of {INTERVENTION_OPS}, got {op!r}")
        if renorm not in RENORM_MODES:
            raise ValueError(f"renorm must be one of {RENORM_MODES}, got {renorm!r}")
        if renorm == "cap_ratio" and not 0.0 < img_cap_ratio <= 1.0:
            raise ValueError("img_cap_ratio must be in (0, 1]")
        uniform = region_mask is None or gamma == gamma2
        if op == "add" and renorm == "preserve_ratio" and not ablate and uniform:
            raise ValueError(
                "a uniform op='add' under preserve_ratio is an exact no-op"
            )
        if op in STRUCTURAL_OPS and not ablate:
            if op in ("blur", "sharpen"):
                if not blur_sigma > 0.0:
                    raise ValueError(f"op={op!r} needs blur_sigma > 0")
                if img_grid is None or img_grid[0] * img_grid[1] != img_hi - img_lo:
                    raise ValueError(f"op={op!r} needs the post-merge image grid")
            if op == "peak" and not 0.0 < peak_lambda <= 1.0:
                raise ValueError("op='peak' needs peak_lambda in (0, 1]")
        if region_mask is not None and region_mask.shape != (img_hi - img_lo,):
            raise ValueError(f"region_mask must have shape ({img_hi - img_lo},)")
        self.layer_lo, self.layer_hi = layer_lo, layer_hi
        self.img_lo, self.img_hi = img_lo, img_hi
        self.gamma, self.gamma2 = gamma, gamma2
        self.region_mask = region_mask
        self.ablate = ablate
        self.op = op
        self.renorm = renorm
        self.img_cap_ratio = img_cap_ratio
        self.blur_sigma = blur_sigma
        self.attn_cfg_alpha = attn_cfg_alpha
        self.peak_lambda = peak_lambda
        self.img_grid = img_grid
        self._patched: dict[nn.Module, Callable] = {}

    def install(self, layers: nn.ModuleList) -> None:
        """Wrap the attention of every layer in [layer_lo, layer_hi)."""
        for layer in layers:
            attn = layer.self_attn
            layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None or not (self.layer_lo <= layer_idx < self.layer_hi):
                continue
            layer_types = getattr(attn.config, "layer_types", None)
            if layer_types is not None and layer_types[layer_idx] != "full_attention":
                raise ValueError(f"layer {layer_idx} is not a full-attention layer")
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def remove(self) -> None:
        """Restore every wrapped attention forward."""
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()

    def __enter__(self) -> "AttentionIntervention":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def _make_wrapper(self, attn: nn.Module, original: Callable):
        gamma = self.gamma
        img_lo, img_hi = self.img_lo, self.img_hi
        ablate, op = self.ablate, self.op
        blur_sigma, attn_cfg_alpha = self.blur_sigma, self.attn_cfg_alpha
        peak_lambda, img_grid = self.peak_lambda, self.img_grid

        # Per-column gamma in fp32 (a bf16 vector would round gamma differently from the
        # scalar path); None keeps the scalar path.
        gamma_vec = None
        region_bool = None
        if self.region_mask is not None:
            device = attn.q_proj.weight.device
            region_mask_f = self.region_mask.to(device=device, dtype=torch.float32)
            gamma_vec = region_mask_f * gamma + (1.0 - region_mask_f) * self.gamma2
            region_bool = self.region_mask.to(device=device, dtype=torch.bool)

        def apply_bias(scores_blk: torch.Tensor) -> torch.Tensor:
            """A -> A_biased for one block of rows; never mutates its input."""
            if ablate:
                out = scores_blk.clone()
                out[:, :, :, img_lo:img_hi] = neg_inf(out.dtype)
                return out
            if op in STRUCTURAL_OPS:
                a_img = scores_blk[:, :, :, img_lo:img_hi]
                if op == "blur":
                    new_img = _blur_image_block(
                        a_img, blur_sigma, img_grid[0], img_grid[1]
                    )
                elif op == "sharpen":
                    new_img = _sharpen_image_block(
                        a_img, blur_sigma, img_grid[0], img_grid[1], attn_cfg_alpha
                    )
                else:
                    new_img = _peak_image_block(a_img, peak_lambda)
                if region_bool is not None:
                    new_img = torch.where(region_bool.view(1, 1, 1, -1), new_img, a_img)
                out = scores_blk.clone()
                out[:, :, :, img_lo:img_hi] = new_img
                return out
            g = gamma_vec if gamma_vec is not None else gamma
            a_img = scores_blk[:, :, :, img_lo:img_hi]
            if op == "mul":
                new_img = a_img * (1.0 + g)
            elif op == "add":
                new_img = a_img + g
            else:
                new_img = a_img + g * a_img.abs()
            out = scores_blk.clone()
            out[:, :, :, img_lo:img_hi] = new_img
            return out

        def to_probs(
            scores_blk: torch.Tensor, plain_probs: torch.Tensor | None
        ) -> torch.Tensor:
            """Softmax honouring `renorm`;
            the plain branch is always the raw softmax."""
            p = nn.functional.softmax(scores_blk, dim=-1, dtype=torch.float32)
            if self.renorm == "global" or plain_probs is None:
                return p
            if self.renorm == "cap_ratio":
                r0 = plain_probs[:, :, :, img_lo:img_hi].sum(dim=-1, keepdim=True)
                own_current = p[:, :, :, img_lo:img_hi].sum(dim=-1, keepdim=True)
                over_cap = r0 >= self.img_cap_ratio
                cap = torch.full_like(r0, self.img_cap_ratio)
                img_target = torch.where(over_cap, r0, torch.minimum(own_current, cap))
                return _rescale_to_target(p, img_target, img_lo, img_hi)
            img_target = plain_probs[:, :, :, img_lo:img_hi].sum(dim=-1, keepdim=True)
            return _rescale_to_target(p, img_target, img_lo, img_hi)

        def wrapper(hidden_states, *args, **kwargs):
            # the real forward first, untouched: real kernel, every row, one cache write
            attn_output, attn_weights = original(hidden_states, *args, **kwargs)
            attention_mask = kwargs.get("attention_mask")
            position_embeddings = kwargs.get("position_embeddings")
            past_key_values = kwargs.get("past_key_values")
            if args:
                attention_mask = args[0] if attention_mask is None else attention_mask
            if past_key_values is None:
                raise RuntimeError("AttentionIntervention needs a real KV cache")
            if position_embeddings is None:
                raise RuntimeError("AttentionIntervention needs position_embeddings")

            with torch.no_grad():
                bsz, q_len, _ = hidden_states.size()
                assert bsz == 1, "AttentionIntervention supports batch size 1 only"
                # this step's queries, projected and roped as the real forward did; the
                # keys/values come from the cache the real forward just wrote
                q = (
                    attn.q_proj(hidden_states)
                    .view(bsz, q_len, -1, attn.head_dim)
                    .transpose(1, 2)
                )
                k_dummy = (
                    attn.k_proj(hidden_states)
                    .view(bsz, q_len, -1, attn.head_dim)
                    .transpose(1, 2)
                )
                cos, sin = position_embeddings
                q, _ = apply_qwen_rope(attn, q, k_dummy, cos, sin)
                cached = past_key_values.layers[attn.layer_idx]
                key_states = repeat_kv(cached.keys, attn.num_key_value_groups)
                value_states = repeat_kv(cached.values, attn.num_key_value_groups)
                value_f32 = value_states.float()
                s_kv = key_states.shape[-2]
                row_offset = s_kv - q_len
                rows = _text_rows(q_len, row_offset, img_hi)
                if rows.numel() == 0:
                    return attn_output, attn_weights
                rows = rows.to(hidden_states.device)
                new_output = attn_output.clone()
                for start in range(0, rows.numel(), ROW_CHUNK):
                    blk = rows[start : start + ROW_CHUNK]
                    q_blk = q[:, :, blk, :]
                    scores = (
                        torch.matmul(q_blk, key_states.transpose(2, 3)) * attn.scaling
                    )
                    if attention_mask is not None:
                        scores = scores + attention_mask[:, :, blk, :s_kv]
                    else:
                        # sdpa's is_causal path passes no mask: build the causal term
                        absolute = (blk + row_offset).view(1, 1, -1, 1)
                        cols = torch.arange(s_kv, device=scores.device).view(
                            1, 1, 1, -1
                        )
                        scores = scores.masked_fill(
                            cols > absolute, neg_inf(scores.dtype)
                        )
                    scores_f32 = scores.float()
                    scores_biased = apply_bias(scores_f32)
                    p_plain = to_probs(scores_f32, None)
                    p_biased = to_probs(scores_biased, p_plain)
                    out_plain = _attend_and_project(attn, p_plain, value_f32, bsz)
                    out_biased = _attend_and_project(attn, p_biased, value_f32, bsz)
                    delta = out_biased - out_plain
                    new_output[:, blk, :] = attn_output[:, blk, :] + delta.to(
                        attn_output.dtype
                    )
                attn_output = new_output
            return attn_output, attn_weights

        return wrapper
