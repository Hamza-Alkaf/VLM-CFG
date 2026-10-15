"""M3ID's logit combination -- deliberately NOT `vcd_utils/contrastive_combine.py` or
MARINE's.

M3ID (Multi-Modal Hallucination Control by Visual Information Grounding, Favero et al.,
CVPR 2024) amplifies the image-conditioned distribution away from the image-free
language prior, by an amount that GROWS with the number of tokens generated, and only
when the conditioned model is UNSURE. Its Eq. 4::

    gamma_t = exp(-lambda * t)
    l* = l_c + 1[ max_k (l_c)_k < log(alpha) ] * ((1 - gamma_t)/gamma_t) * (l_c - l_u)

where `l_c = log p(y | y_<t, x, c)` (with the image) and `l_u = log p(y | y_<t, x)`
(text only).

WHY NOT SHARE A COMBINE. Three methods in this repo contrast two streams and all three
differ:

  * VCD / LayerCD / SID (`contrastive_combine`): RAW logits, coefficients `(1+a)` and
    `-a`, and an adaptive-plausibility MASK that sets low-`logits_main` entries to -inf.
    Its truncation is per-token and fires MORE on peaked distributions.
  * MARINE (`marine_guidance_combine`): log-softmax space, a FIXED strength, guiding
    TOWARD an enriched stream, with a final renormalisation.
  * M3ID (here): a step-dependent extrapolation AWAY FROM a degraded stream, with no
    per-token mask at all -- instead a per-row on/off GATE keyed to the conditional
    model's confidence, which fires LESS on peaked distributions. Sign, time-dependence,
    truncation and renorm all differ from both of the others.

THE NEUTRAL POINT IS IN THE ARITHMETIC, NOT IN A BRANCH. The correction term is computed
in log-softmax space exactly as the paper specifies (it is a difference of
log-probabilities) but is added to the RAW logits rather than to `l_c`. The result
differs from the paper's `l*` by `logsumexp(logits_cond)` -- a per-row additive
constant -- which changes neither `argmax`, nor `softmax(./T)` at any temperature, nor
top-p/top-k ordering. Same distribution, therefore the same method; but at `lambda = 0`
the weight is `expm1(0) == 0.0` exactly, so the function returns `logits_cond`
BIT-FOR-BIT while having executed every line. That is what lets the neutral smoke test
be a statement about this formula rather than about an `if` that skipped it.

NO RENORMALISATION. The paper's Eq. 4 ends at `argmax l*`; adding a `log_softmax` here
would be provably inert (it subtracts a per-row constant) and would depart from the
source, so it is not done. This is the opposite call from `marine_combine.py`, which
DOES renormalise -- because MARINE's upstream code does. Both are "reproduce the
source"; the sources differ.
"""

import math

import torch
import torch.nn.functional as F


def m3id_weight(t: int, lam: float) -> float:
    """M3ID's guidance weight ``(1 - gamma_t)/gamma_t`` with
    ``gamma_t = exp(-lambda*t)``.

    Computed as ``expm1(lambda*t)``, which is that expression exactly::

        (1 - e^-x)/e^-x  =  e^x - 1  =  expm1(x)

    NEVER evaluate the literal form. It divides by ``e^-x``, which underflows to ``0.0``
    in fp32 once ``x > 88`` -- giving ``inf`` and then NaN logits -- and it loses
    precision to catastrophic cancellation for small ``x``. ``expm1`` is exact at both
    ends and finite everywhere. This matters in practice: ``t`` here is offset by the
    prompt's tail length, so a mis-set offset can push ``lambda*t`` far higher than the
    paper's captioning regime ever did.
    """
    return math.expm1(lam * t)


def m3id_combine(
    logits_cond: torch.Tensor,
    logits_uncond: torch.Tensor,
    t: int,
    lam: float,
    alpha: float,
    max_weight: float = 0.0,
) -> torch.Tensor:
    """M3ID's Eq. 4 for one decode step.

    Args:
        logits_cond: raw logits from the image-conditioned stream, `[B, 1, V]`.
        logits_uncond: raw logits from the text-only stream, same shape.
        t: the EFFECTIVE step index (`t0 + tokens generated so far`); see
            `qwen_m3id_model.py` for why `t0` exists and why it excludes the image
            tokens.
        lam: M3ID's `lambda`, the fading-memory rate. Paper default 0.02. `0.0` disables
            the correction exactly, via `expm1(0) == 0.0`, without a branch.
        alpha: M3ID's confidence threshold. The correction is applied only where
            `max_k p_cond < alpha`, i.e. only where the conditioned model is UNSURE --
            the paper's "we suppress our intervention when the contextual pressure is
            high". Note this is INVERTED relative to VCD's `beta`: `alpha -> 0` never
            corrects, `alpha -> 1` always does.
        max_weight: `0.0` (default) means uncapped, which is what the paper specifies.
            Any positive value clamps the weight and is an explicitly-labelled
            DEVIATION, provided only because the weight grows without bound in `t`.

    Returns:
        The guided logits, same shape and dtype as `logits_cond`, differing from the
        paper's `l*` by a per-row constant (see the module docstring).
    """
    lc32 = logits_cond.float()
    l_c = F.log_softmax(lc32, dim=-1)
    l_u = F.log_softmax(logits_uncond.float().to(l_c.device), dim=-1)

    w = m3id_weight(t, lam)
    if max_weight > 0.0:
        w = min(w, max_weight)

    # `l_c` is a log-softmax, so `max_k (l_c)_k < log(alpha)` is exactly
    # `max_k p_cond < alpha`. Comparing in log space keeps it one op and matches the
    # paper's own phrasing.
    gate = l_c.max(dim=-1, keepdim=True).values < math.log(alpha)
    coeff = torch.where(
        gate,
        torch.full_like(gate, w, dtype=l_c.dtype),
        torch.zeros_like(gate, dtype=l_c.dtype),
    )
    return (lc32 + coeff * (l_c - l_u)).to(logits_cond.dtype)
