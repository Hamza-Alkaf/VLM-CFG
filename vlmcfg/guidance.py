"""Guidance arithmetic:
the configuration, the blends, the reference rules and the guards.

Every function here is a pure tensor operation. The order of operations is part of the
contract: guidance compares two forward passes in bf16/fp16, so an algebraically equal
rewrite can round differently and flip a near-tie token.
Do not simplify the expressions.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vlmcfg.args import split_args

FEATURES = ("hidden", "logits", "logprobs")
# Where the reference stands. "switch" is the paper rule (switch_k=0 is teacher
# forcing); "cached" keeps the prefix-free step-0 reference for the whole answer (i-1);
# "window" lets the reference read only the last `window` generated tokens.
REFERENCES = ("switch", "cached", "window")
# Rules that rescale the cached (i-1) push, all on a teacher-forced negative stream
# except "cap", which runs the negative stream at step 0 only.
RULES = ("plain", "strength_oracle", "decay", "cap")
OPERATORS = ("cfg", "apg", "zerostar", "tcfg", "epcfg")
GUARDS = ("none", "apc", "eos")
# The two algebraically equal ways the backbones write the blend, kept apart because
# they round differently: "reference" is u + a(c - u),
# "conditional" is c + (a - 1)(c - u).
BLEND_FORMS = ("reference", "conditional")

LmHead = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GuidanceConfig:
    """Settings of one guided decode. The defaults are the paper configuration.

    Attributes:
        alpha: guidance scale; 1.0 runs both streams and lets them cancel.
        feature: blend the last hidden state ("hidden"), the logits ("logits", in the
            model's dtype) or the fp32 log-probabilities ("logprobs").
        reference: where the reference stands, see REFERENCES.
        switch_k: with reference="switch", the reference stays on the cached prefix-free
            state for the first `switch_k` tokens, then is teacher-forced.
            0 teacher-forces from the start.
        window: with reference="window", how many of the last generated tokens the
            reference reads (0 re-reads the prompt alone at every step).
        rule: rescaling of the cached push (teacher-forced reference, hidden feature
            only): the strength oracle, the decay schedule or the step-0 cap.
        operator: the blend operator; "apg" uses `apg_eta`. Hidden-state feature only.
        apg_eta: APG's scale on the part of (c - u) parallel to c.
        guard: "apc" masks tokens the conditional finds implausible; "eos" takes the
            stop decision from the conditional alone; "none" leaves the blend as it is.
        apc_beta: APC keeps tokens with p_c >= apc_beta * max p_c.
        guide_steps: with the cached reference, guide only the first `guide_steps`
            tokens and decode the rest from the conditional alone, unguarded;
            0 guides them all.
    """

    alpha: float = 1.5
    feature: str = "hidden"
    reference: str = "switch"
    switch_k: int = 8
    window: int = 0
    rule: str = "plain"
    operator: str = "cfg"
    apg_eta: float = 1.0
    guard: str = "apc"
    apc_beta: float = 0.1
    guide_steps: int = 0

    def __post_init__(self) -> None:
        _check_choice("feature", self.feature, FEATURES)
        _check_choice("reference", self.reference, REFERENCES)
        _check_choice("rule", self.rule, RULES)
        _check_choice("operator", self.operator, OPERATORS)
        _check_choice("guard", self.guard, GUARDS)
        if self.switch_k < 0:
            raise ValueError(f"switch_k must be >= 0, got {self.switch_k}")
        if self.window < 0:
            raise ValueError(f"window must be >= 0, got {self.window}")
        if not 0.0 < self.apc_beta <= 1.0:
            raise ValueError(f"apc_beta must be in (0, 1], got {self.apc_beta}")
        if self.rule != "plain" and not (
            self.reference == "switch" and self.switch_k == 0
        ):
            raise ValueError("rules need the teacher-forced reference (switch_k=0)")
        if self.rule != "plain" and self.feature != "hidden":
            raise ValueError("rules need feature=hidden")
        if self.guide_steps < 0 or (self.guide_steps and self.reference != "cached"):
            raise ValueError("guide_steps needs reference=cached and must be >= 0")
        if self.operator != "cfg" and self.feature != "hidden":
            raise ValueError("operators other than cfg need feature=hidden")
        if self.operator != "cfg" and self.rule != "plain":
            raise ValueError("an operator and a rule cannot be combined")

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> tuple["GuidanceConfig", dict[str, Any]]:
        """Split command-line `args` into a config and the remaining keywords."""
        return split_args(cls, args)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def blend(
    cond: torch.Tensor,
    ref: torch.Tensor,
    cfg: GuidanceConfig,
    form: str,
    lm_head: LmHead,
) -> torch.Tensor:
    """Guided next-token scores from one step of the two streams.

    Hidden states and logits are blended in the model's own dtype (hidden states are
    then read out); log-probabilities in fp32.
    alpha=1 is evaluated as written, never short-cut.
    """
    a = cfg.alpha
    if cfg.feature == "hidden":
        if cfg.operator == "apg":
            return lm_head(
                cond + (a - 1.0) * apg_project(cond - ref, cond, cfg.apg_eta)
            )
        if cfg.operator != "cfg":
            return lm_head(_geometry_blend(cond, ref, a, cfg.operator))
        if form == "reference":
            return lm_head(ref + a * (cond - ref))
        return lm_head(cond + (a - 1.0) * (cond - ref))
    if cfg.feature == "logprobs":
        lc = torch.log_softmax(cond.float(), dim=-1)
        lu = torch.log_softmax(ref.float(), dim=-1)
        if form == "reference":
            return lu + a * (lc - lu)
        return lc + (a - 1.0) * (lc - lu)
    if ref.shape[-1] != cond.shape[-1]:
        return _blend_common_vocab(cond, ref, a)
    if form == "reference":
        return ref + a * (cond - ref)
    return cond + (a - 1.0) * (cond - ref)


def _blend_common_vocab(
    cond: torch.Tensor, ref: torch.Tensor, a: float
) -> torch.Tensor:
    """Logit blend against a weak model whose padded vocabulary is narrower: the blend
    runs on the shared width and the padding ids are filled with -1e4, never chosen."""
    v = min(cond.shape[-1], ref.shape[-1])
    delta = cond[..., :v] - ref[..., :v]
    guided = ref[..., :v] + (a * delta)
    if v == cond.shape[-1]:
        return guided
    pad_shape = (*guided.shape[:-1], cond.shape[-1] - v)
    pad = torch.full(pad_shape, -1e4, dtype=guided.dtype, device=guided.device)
    return torch.cat([guided, pad], dim=-1)


# ---------------------------------------------------------------------- operators
def apg_project(
    delta: torch.Tensor, reference: torch.Tensor, eta: float
) -> torch.Tensor:
    """APG (Sadat et al. 2024):
    scale the part of `delta` parallel to `reference` by eta."""
    if eta == 1.0:
        return delta
    delta_flat = delta.reshape(-1).float()
    ref_flat = reference.reshape(-1).float()
    parallel = torch.dot(delta_flat, ref_flat) / (torch.dot(ref_flat, ref_flat) + 1e-8)
    delta_parallel = (parallel * reference).to(delta.dtype)
    delta_orthogonal = delta - delta_parallel
    return eta * delta_parallel + delta_orthogonal


def _geometry_blend(
    c: torch.Tensor, u: torch.Tensor, a: float, op: str
) -> torch.Tensor:
    """CFG-Zero* and TCFG rewrite the reference before the blend;
    EP-CFG rescales after."""
    if op == "zerostar":
        u = zero_star_reference(u, c)
    elif op == "tcfg":
        u = tcfg_reference(u, c)
    g = u + a * (c - u)
    if op == "epcfg":
        g = energy_preserving_rescale(g, c)
    return g


def zero_star_reference(
    u: torch.Tensor, c: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """CFG-Zero* (arXiv:2503.18886): u scaled by its least-squares fit to c."""
    orig_dtype = u.dtype
    uf, cf = u.float(), c.float()
    dot = (cf * uf).sum(dim=-1, keepdim=True)
    norm_sq = (uf * uf).sum(dim=-1, keepdim=True)
    s = dot / norm_sq.clamp_min(eps)
    return (s * uf).to(orig_dtype)


def tcfg_reference(u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """TCFG (arXiv:2503.18137): u projected on the top singular direction of [c; u]."""
    orig_dtype = u.dtype
    uf, cf = u.float(), c.float()
    _, _, vh = torch.linalg.svd(torch.stack((cf, uf), dim=-2), full_matrices=False)
    vh_damped = vh.clone()
    vh_damped[..., 1, :] = 0
    proj = torch.matmul(torch.matmul(uf.unsqueeze(-2), vh.transpose(-2, -1)), vh_damped)
    return proj.squeeze(-2).to(orig_dtype)


def energy_preserving_rescale(g: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """EP-CFG (arXiv:2412.09966): rescale the guided vector onto c's energy."""
    x, cf = g.float(), c.float()
    e_c = cf.pow(2).sum(dim=-1, keepdim=True)
    e_g = x.pow(2).sum(dim=-1, keepdim=True)
    ones = torch.ones_like(e_g)
    scale = torch.where(e_g > 0, torch.sqrt(e_c / e_g.clamp_min(1e-12)), ones)
    return (x * scale).to(g.dtype)


# ---------------------------------------------------------------------- reference rules
def rule_blend(
    cond: torch.Tensor,
    ref: torch.Tensor | None,
    step: int,
    state: dict,
    cfg: GuidanceConfig,
    lm_head: LmHead,
) -> torch.Tensor:
    """The cached (i-1) push rescaled by a scalar strength g, in hidden-state space.

    `ref` is the teacher-forced reference (None for "cap" after step 0); `state` carries
    the step-0 reference and constants across the steps of one answer.
      strength_oracle  g = min(|D*_TF| / |D*_i1|, 1) at the conditional's top token
      decay            g = min(|D*_TF(t)| / |D*_TF(0)|, 1)
      cap              g = min(|c(0) - u0| / |c(t) - u0|, 1), vector norm
    """

    def to_lp(f: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(lm_head(f).float(), dim=-1)

    a = cfg.alpha
    if step == 0:
        assert ref is not None
        state["u0"] = ref.clone()
        state["u0_lp"] = to_lp(ref)
    if cfg.rule == "cap":
        return _cap_blend(cond, step, a, state, lm_head)
    assert ref is not None
    lc, ln = to_lp(cond), to_lp(ref)
    w = int(lc.argmax(dim=-1).item())
    ds_tf = float(lc[0, w] - ln[0, w])
    if cfg.rule == "decay":
        if step == 0:
            state["ds0"] = abs(ds_tf)
        g = min(abs(ds_tf) / max(state["ds0"], 1e-6), 1.0)
    else:
        ds_i1 = float(lc[0, w] - state["u0_lp"][0, w])
        g = min(abs(ds_tf) / abs(ds_i1), 1.0) if abs(ds_i1) > 1e-6 else 1.0
    return lm_head(cond + (a - 1.0) * g * (cond - state["u0"]))


def _cap_blend(
    cond: torch.Tensor, step: int, a: float, state: dict, lm_head: LmHead
) -> torch.Tensor:
    """i-1 blend with its push never larger than at step 0; g == 1 is the i-1 blend."""
    u0 = state["u0"]
    d = cond - u0
    n = float(d.float().norm())
    if step == 0:
        state["cap_n0"] = n
    g = min(state["cap_n0"] / n, 1.0) if n > 0 else 1.0
    # written in the i-1 blend's own form so that g == 1 rounds exactly as i-1 does
    a_eff = a if g == 1.0 else 1.0 + (a - 1.0) * g
    return lm_head(u0 + a_eff * d)


# ---------------------------------------------------------------------- guards
def cond_logprobs(cond: torch.Tensor, feature: str, lm_head: LmHead) -> torch.Tensor:
    """Log-probabilities of the conditional stream at this step."""
    if feature == "hidden":
        return torch.log_softmax(lm_head(cond).float(), dim=-1)
    return torch.log_softmax(cond.float(), dim=-1)


def apply_guard(
    guided: torch.Tensor,
    cond: torch.Tensor,
    cfg: GuidanceConfig,
    eos_ids: frozenset[int],
    lm_head: LmHead,
) -> torch.Tensor:
    """Constrain the guided scores before the argmax (see GUARDS)."""
    if cfg.guard == "none":
        return guided
    lc = cond_logprobs(cond, cfg.feature, lm_head)
    if cfg.guard == "apc":
        return apc_mask(guided, lc, cfg.apc_beta)
    return eos_guard(guided, lc, eos_ids)


def apc_mask(guided: torch.Tensor, lc: torch.Tensor, beta: float) -> torch.Tensor:
    """Adaptive plausibility constraint (VCD, Leng et al. 2024, eq. 4).

    Keeps only tokens with p_c(y) >= beta * max_w p_c(w); the rest get -inf.
    """
    head = lc >= (lc.max() + math.log(beta))
    return guided.masked_fill(~head, float("-inf"))


def eos_guard(
    guided: torch.Tensor, lc: torch.Tensor, eos_ids: frozenset[int]
) -> torch.Tensor:
    """Stop exactly when the conditional's top token is a stop token,
    never otherwise."""
    c_top = int(lc.argmax(dim=-1).item())
    if c_top in eos_ids:
        out = torch.full_like(guided, float("-inf"))
        out[..., c_top] = 0.0
        return out
    out = guided.clone()
    for e in eos_ids:
        out[..., e] = float("-inf")
    return out
