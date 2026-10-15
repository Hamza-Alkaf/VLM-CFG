"""PAI's last-row image-attention amplification on Molmo2, spliced as a delta.

    A[:, :, -1, img] += alpha * |A[:, :, -1, img]|

Same operator and the same delta form as `internvl_pai_attention.py` (see `attention.py`
for why the delta: alpha=0 is bitwise neutral without a guard). Two Molmo2-specific
changes:

  * `img` is a BOOLEAN COLUMN MASK, not a `[lo, hi)` span. Molmo2's image block
    interleaves the patch tokens with column-separator and start/end marker tokens, and
    only the patch positions carry vision features (`Molmo2Adapter.image_token_mask`).
    PAI boosts attention to the image's visual tokens, so the markers are left alone.
    This is the one place the port has to choose, and it is recorded as a deviation in
    `molmo2_pai.py`.
  * The output projection is `attn.attn_out` (bias-free), not `o_proj`, so
    `_attend_and_project` is re-stated here rather than imported.
"""

from collections.abc import Callable

import torch
from torch import nn

from vlmcfg.baselines.common.backbone import Molmo2Adapter, neg_inf


def _attend_and_project(
    attn: nn.Module, probs: torch.Tensor, value_f32: torch.Tensor, bsz: int
) -> torch.Tensor:
    """`attn_out(probs @ V)` for the query row, in fp32 (attention.py's, with Molmo2's
    name)."""
    out = torch.matmul(probs, value_f32)
    out = out.transpose(1, 2).contiguous().reshape(bsz, probs.shape[2], -1)
    return nn.functional.linear(out, attn.attn_out.weight.float(), None)


class Molmo2PAIIntervention:
    """PAI's last-row image-attention amplification on Molmo2.

    `img_mask` is the `[L_prompt]` boolean patch mask of the CONDITIONAL prompt. Decode
    steps only append keys after the prompt, so the mask is padded with False to the
    cache length. `use_cfg` is PAI's own switch, raised by `CFGLogits` around the
    text-only forward.
    """

    def __init__(
        self, layer_lo: int, layer_hi: int, img_mask: torch.Tensor, alpha: float
    ) -> None:
        if not layer_lo < layer_hi:
            raise ValueError(f"Empty layer range [{layer_lo}, {layer_hi}).")
        if img_mask.dim() != 1 or not bool(img_mask.any()):
            raise ValueError(
                "img_mask must be a non-empty 1-D boolean mask over the prompt."
            )
        self.layer_lo, self.layer_hi = layer_lo, layer_hi
        self.img_mask = img_mask.bool()
        self.alpha = float(alpha)
        self.use_cfg = False
        self._patched: dict[nn.Module, Callable] = {}
        self._adapter = Molmo2Adapter()

    def install(self, layers: nn.ModuleList) -> None:
        for layer in layers:
            attn = layer.self_attn
            layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None or not (self.layer_lo <= layer_idx < self.layer_hi):
                continue
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def _columns(self, s_kv: int, device) -> torch.Tensor:
        """The patch mask, extended with False over the generated tokens."""
        mask = torch.zeros(s_kv, dtype=torch.bool, device=device)
        n = min(s_kv, self.img_mask.numel())
        mask[:n] = self.img_mask[:n].to(device)
        return mask

    def _make_wrapper(self, attn: nn.Module, original: Callable):
        adapter = self._adapter

        def wrapper(hidden_states, *args, **kwargs):
            attn_output, attn_weights = original(hidden_states, *args, **kwargs)
            if self.use_cfg:
                return attn_output, attn_weights

            attention_mask = kwargs.get("attention_mask")
            position_embeddings = kwargs.get("position_embeddings")
            past_key_values = kwargs.get("past_key_values")
            if past_key_values is None:
                raise RuntimeError("PAI requires use_cache=True with a real Cache.")
            if position_embeddings is None:
                raise RuntimeError(
                    "PAI requires position_embeddings (RoPE cos/sin) to be passed."
                )

            with torch.no_grad():
                bsz = hidden_states.shape[0]
                if bsz != 1:
                    raise RuntimeError("PAI only supports batch_size=1 (no padding).")
                q, _ = adapter.recompute_qk(attn, hidden_states, position_embeddings)
                keys, values = adapter.read_cache(past_key_values, attn.layer_idx)
                key_states = adapter.expand_kv(keys, attn)
                value_f32 = adapter.expand_kv(values, attn).float()
                s_kv = key_states.shape[-2]

                scores = (
                    torch.matmul(q[:, :, -1:, :], key_states.transpose(2, 3))
                    * attn.scaling
                )
                if attention_mask is not None:
                    mask = attention_mask
                    if mask.dtype == torch.bool:
                        mask = torch.where(mask, 0.0, neg_inf(scores.dtype)).to(
                            scores.dtype
                        )
                    scores = scores + mask[:, :, -1:, :s_kv]
                scores_f32 = scores.float()

                ### PAI's modification, on the patch columns
                cols = self._columns(s_kv, scores.device)
                scores_biased = scores_f32.clone()
                a_img = scores_f32[..., cols]
                scores_biased[..., cols] = a_img + self.alpha * a_img.abs()
                ### PAI's modification

                p_plain = nn.functional.softmax(scores_f32, dim=-1, dtype=torch.float32)
                p_biased = nn.functional.softmax(
                    scores_biased, dim=-1, dtype=torch.float32
                )
                delta = _attend_and_project(
                    attn, p_biased, value_f32, bsz
                ) - _attend_and_project(attn, p_plain, value_f32, bsz)

                new_output = attn_output.clone()
                new_output[:, -1, :] = attn_output[:, -1, :] + delta[:, 0, :].to(
                    attn_output.dtype
                )
            return new_output, attn_weights

        return wrapper

    def remove(self) -> None:
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()
