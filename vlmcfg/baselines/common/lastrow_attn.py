"""Harvest the grounding probe's attention row without materialising the full matrix.

The grounding probe needs one number per image token: how much the newest token
attends to it, averaged over heads. Obtaining that via ``output_attentions=True``
forces every layer to build a ``[B, H, seq, seq]`` fp32 tensor -- ~27 GiB for a
single layer at 16k tokens, which OOMs an 80 GB A100.

Worse than the memory cost, it forces ``attn_implementation="eager"`` on the text
tower, and eager measurably changes Qwen2.5-VL's output: on a fixed prompt it
diverges from flash-attention-2 by ``max|dlogit| = 1.22`` and produces different
greedy text, which costs ~6 points of textvqa accuracy. Grounding runs were
therefore not comparable to their flash-attention baselines.

Attention from a single query position is a matrix-vector product, so the row can
be computed directly for ~1 MiB while the real forward stays on flash attention.
Softmax is row-independent, so this is an exact refactor of the row the probe
reads, not an approximation.
"""

import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv


class LastRowAttentionCapture:
    """Capture each layer's last-row attention over image tokens.

    Install with :meth:`install`, run the probe forward, then read :attr:`rows`.
    The capture is keyed by ``module.layer_idx`` so rows can be ordered by layer
    regardless of execution order.

    Only the query's final position is used, matching the probe's ``[..., -1, ...]``
    slice, so the full attention matrix is never built.
    """

    def __init__(
        self, img_lo: int, img_hi: int, per_head: bool = False, keep_heads: bool = False
    ) -> None:
        """Store the image-token span to slice, as ``[img_lo, img_hi)``.

        The two head-related flags are INDEPENDENT and do different things -- they came
        from separate efforts and both have live callers, so neither is a rename of the
        other:

        ``per_head`` CHANGES :attr:`rows` to keep the head axis (``[H, img_tokens]``
        instead of ``[img_tokens]``) and additionally fills :attr:`head_mass` with each
        head's total attention on the image span -- what
        ``--attn_weaken_strategy heads`` ranks by (``model_utils/attn_weaken.py``).

        ``keep_heads`` leaves :attr:`rows` exactly as it was (head-averaged) and
        additionally retains the per-head rows in :attr:`head_rows`, for consumers that
        need the full ``[L, H, N_img]`` stack -- the trained ROI mask decoder
        (``mask_decoder/qwen_roi_decoder``) takes one input channel per (layer, head).
        Purely additive: a ``keep_heads=False`` capture, and every existing caller, is
        untouched.
        """
        self.img_lo = img_lo
        self.img_hi = img_hi
        self.per_head = per_head
        self.keep_heads = keep_heads
        self.rows: dict[int, torch.Tensor] = {}
        self.head_mass: dict[int, torch.Tensor] = {}
        self.head_rows: dict[int, torch.Tensor] = {}
        self._patched: dict[nn.Module, object] = {}

    def _capture(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
    ) -> None:
        """Compute and store this layer's row. Shapes follow eager_attention_forward."""
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return

        # Grouped-query attention: Qwen2.5-VL-7B has 28 query heads over 4 KV heads,
        # so the KV side must be expanded before the heads line up.
        key_states = repeat_kv(key, module.num_key_value_groups)

        # [B, H, 1, D] @ [B, H, D, S] -> [B, H, 1, S]: the newest token's row only.
        q_last = query[:, :, -1:, :]
        row = torch.matmul(q_last, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            # Slice the real last row rather than assuming it is all-zeros; with
            # left padding the masked positions matter.
            causal_mask = attention_mask[:, :, -1:, : key_states.shape[-2]]
            row = row + causal_mask

        # fp32 to match eager's softmax exactly; at [H, S] this costs ~1 MiB.
        row = nn.functional.softmax(row, dim=-1, dtype=torch.float32)

        if self.per_head:
            # [H, img_tokens]: keep the head axis, for interventions that select heads
            # rather than tokens (--attn_weaken_strategy heads). Also record each head's
            # TOTAL image share, which is what heads are ranked by.
            img = row[0, :, 0, self.img_lo : self.img_hi]
            self.rows[layer_idx] = img.detach().cpu()
            self.head_mass[layer_idx] = img.sum(dim=-1).detach().cpu()
        else:
            # [B, H, 1, S] -> mean over heads of the image-token span -> [img_tokens]
            self.rows[layer_idx] = (
                row[0, :, 0, self.img_lo : self.img_hi].mean(dim=0).detach().cpu()
            )

        # Deliberately a SEPARATE slice rather than deriving `rows` from this one by a
        # CPU `.mean(dim=0)`: the branch above averages on-device in fp32 and moving
        # that reduction to CPU would perturb its last bits, so every consumer of the
        # head-averaged map (cfg_mode='grounded') would silently stop being
        # bit-identical to a run without keep_heads. The duplicated slice is ~4 MB.
        if self.keep_heads:
            # [B, H, 1, S] -> [heads, img_tokens]
            self.head_rows[layer_idx] = (
                row[0, :, 0, self.img_lo : self.img_hi].detach().cpu()
            )

    def install(self, layers: nn.ModuleList) -> None:
        """Wrap each layer's ``self_attn.forward`` so the row is captured in place.

        A forward hook cannot be used: ``query_states``/``key_states`` are locals of
        ``Qwen2_5_VLAttention.forward`` and never appear in its inputs or outputs.
        Wrapping the projections is what makes them reachable while the attention
        itself still runs on the model's configured (flash) kernel.
        """
        for layer in layers:
            attn = layer.self_attn
            if attn in self._patched:
                continue
            self._patched[attn] = attn.forward
            attn.forward = self._make_wrapper(attn, attn.forward)

    def _make_wrapper(self, attn: nn.Module, original):
        capture = self._capture

        def wrapper(hidden_states, *args, **kwargs):
            # Run the real attention first: it writes this step's keys into the KV
            # cache, so afterwards the cache holds the whole sequence even when
            # `generate` has narrowed hidden_states to a single new token. Reading the
            # cache is what keeps the captured row indexed by absolute position,
            # matching the image-token span.
            out = original(hidden_states, *args, **kwargs)

            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                return out

            past_key_values = kwargs.get("past_key_values")
            attention_mask = kwargs.get("attention_mask")
            if args:
                attention_mask = args[0] if attention_mask is None else attention_mask

            # `apply_multimodal_rotary_pos_emb` was removed in transformers 5.x; the
            # shim dispatches on what the installed version provides.
            from vlmcfg.baselines.common.rope_compat import apply_qwen_rope

            with torch.no_grad():
                bsz, q_len, _ = hidden_states.size()
                q = (
                    attn.q_proj(hidden_states)
                    .view(bsz, q_len, -1, attn.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    attn.k_proj(hidden_states)
                    .view(bsz, q_len, -1, attn.head_dim)
                    .transpose(1, 2)
                )
                cos, sin = position_embeddings
                q, k = apply_qwen_rope(attn, q, k, cos, sin)
                # With a cache the freshly projected `k` covers only the new token, so
                # the span slice would be empty; take the full key history the layer
                # just used.
                if past_key_values is not None:
                    cached = past_key_values.layers[attn.layer_idx].keys
                    if cached is not None and cached.shape[-2] >= k.shape[-2]:
                        k = cached
                capture(attn, q, k, attention_mask, attn.scaling)

            return out

        return wrapper

    def remove(self) -> None:
        """Restore the original ``forward`` on every wrapped module."""
        for attn, original in self._patched.items():
            attn.forward = original
        self._patched.clear()

    def stacked(self) -> torch.Tensor:
        """Return the captured rows as ``[num_layers, img_tokens]``, ordered by
        layer."""
        return torch.stack([self.rows[i] for i in sorted(self.rows)], dim=0)

    def stacked_heads(self) -> torch.Tensor:
        """Return the per-head rows as ``[num_layers, num_heads, img_tokens]``.

        Layer-major ordering, matching the channel layout the ROI decoder's first conv
        was trained on (``layer * num_heads + head``; see
        ``mask_decoder/qwen_roi_decoder/decoder.py``). Requires ``keep_heads=True``.
        """
        if not self.keep_heads:
            raise RuntimeError(
                "stacked_heads() needs LastRowAttentionCapture(..., keep_heads=True); "
                "this capture only kept the head-averaged rows."
            )
        return torch.stack([self.head_rows[i] for i in sorted(self.head_rows)], dim=0)

    def __enter__(self) -> "LastRowAttentionCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
