"""Backbone-agnostic guided greedy decoding with two streams through one model.

The conditional stream sees the image; the reference (negative) stream does not. Both
are advanced by the same chosen token, so their prefixes never diverge. The paper's
reference rule is the switch: for the first `switch_k` tokens the reference is the
negative stream's step-0 state, cached (prefix-free); at token `switch_k` the negative
stream catches up on the answer one token at a time and is teacher-forced from then on.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import torch

from vlmcfg.guidance import GuidanceConfig, LmHead, apply_guard, blend, rule_blend


@dataclass
class Stream:
    """Decoding state of one stream: token ids, attention mask and KV cache.

    `extra` holds the processor outputs (pixels, grids) that only the prefill consumes;
    `state` is private bookkeeping of the backbone (e.g. Qwen's mRoPE offsets).
    """

    ids: torch.Tensor
    mask: torch.Tensor
    past: Any = None
    extra: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    def append(self, token: int) -> None:
        """Extend the stream by one token."""
        tok = torch.tensor([[token]], device=self.ids.device)
        self.ids = torch.cat([self.ids, tok], dim=1)
        self.mask = torch.cat([self.mask, torch.ones_like(tok)], dim=1)


class Backbone(Protocol):
    """What the decode loop needs from a model."""

    blend_form: str
    eos_ids: frozenset[int]

    @property
    def lm_head(self) -> LmHead: ...

    def step(self, stream: Stream, first: bool, feature: str) -> torch.Tensor:
        """Run one forward of `stream` and return this step's feature, shape [1, *]."""
        ...

    def past_length(self, stream: Stream) -> int:
        """Number of tokens already in the stream's KV cache."""
        ...


@torch.no_grad()
def guided_decode(
    backbone: Backbone,
    cond: Stream,
    neg: Stream,
    max_new_tokens: int,
    cfg: GuidanceConfig,
    neg_backbone: Backbone | None = None,
) -> list[int]:
    """Greedy-decode with guidance and return the generated token ids (no stop token).

    `neg_backbone` runs the negative stream when it is another model (a weak model).
    """
    nb = backbone if neg_backbone is None else neg_backbone
    k = cfg.switch_k if cfg.reference == "switch" else 0
    windowed = cfg.reference == "window"
    prompt_ids = neg.ids
    out: list[int] = []
    cached_ref = None
    rule_state: dict = {}
    unguided_cfg = replace(cfg, guard="none", guide_steps=0)
    for step in range(max_new_tokens):
        c = backbone.step(cond, step == 0, cfg.feature)
        if step > 0 and (cfg.reference == "cached" or (k and step < k)):
            u = cached_ref
        elif step > 0 and cfg.rule == "cap":
            u = None  # the cap reads only the cached step-0 reference
        elif k and step == k:
            u = _catch_up(nb, neg, cfg.feature)
        elif windowed and step > 0:
            u = _windowed(nb, neg, prompt_ids, out, cfg.window, cfg.feature)
        else:
            u = nb.step(neg, step == 0, cfg.feature)
            cached_ref = u if step == 0 else cached_ref
        step_cfg = cfg
        if cfg.guide_steps and step >= cfg.guide_steps:
            # past the guided steps: the conditional alone, unguarded
            guided = backbone.lm_head(c) if cfg.feature == "hidden" else c
            step_cfg = unguided_cfg
        elif cfg.rule != "plain":
            guided = rule_blend(c, u, step, rule_state, cfg, backbone.lm_head)
        else:
            assert u is not None
            guided = blend(c, u, cfg, backbone.blend_form, backbone.lm_head)
        guided = apply_guard(guided, c, step_cfg, backbone.eos_ids, backbone.lm_head)
        token = int(guided.argmax(dim=-1).item())
        if token in backbone.eos_ids:
            break
        out.append(token)
        cond.append(token)
        if not windowed:
            neg.append(token)
    return out


def _catch_up(backbone: Backbone, stream: Stream, feature: str) -> torch.Tensor:
    """Feed the tokens the cached phase skipped, one forward per token.

    One token at a time, not one k-token forward: the latter is the same math on
    different kernels, and its rounding drift flips near-ties later in long answers.
    This keeps the switch bitwise equal to teacher forcing from token k on.
    """
    full_ids, full_mask = stream.ids, stream.mask
    feat = None
    for j in range(backbone.past_length(stream) + 1, full_ids.shape[1] + 1):
        stream.ids, stream.mask = full_ids[:, :j], full_mask[:, :j]
        feat = backbone.step(stream, False, feature)
    stream.ids, stream.mask = full_ids, full_mask
    if feat is None:
        raise RuntimeError("catch-up had no tokens to feed")
    return feat


def _windowed(
    backbone: Backbone,
    stream: Stream,
    prompt_ids: torch.Tensor,
    generated: list[int],
    window: int,
    feature: str,
) -> torch.Tensor:
    """The reference on the prompt plus only the last `window` generated tokens.

    A sliding window cannot keep its KV cache (dropping the oldest token shifts every
    later position), so the cache is cropped back to the prompt and the window re-fed in
    one forward. With no window the last prompt position is re-run instead.
    """
    prompt_len = prompt_ids.shape[1]
    win = generated[-window:] if window > 0 else []
    stream.past.crop(prompt_len)
    ids = prompt_ids
    if win:
        ids = torch.cat(
            [prompt_ids, torch.tensor([win], device=prompt_ids.device)], dim=1
        )
    else:
        stream.past.crop(prompt_len - 1)
    stream.ids, stream.mask = ids, torch.ones_like(ids)
    return backbone.step(stream, False, feature)
