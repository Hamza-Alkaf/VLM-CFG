"""Shared contrastive-decoding combination formula.

Both VCD (`vcd_sample.py`) and LayerCD (`../../LayerCD/util/cd_utils.py`) combine a
"mature" logit stream against an "impaired" one with the exact same formula and adaptive
plausibility constraint (verified by reading both repos' source):

    diffs  = (1 + cd_alpha) * logits_main - cd_alpha * logits_impaired
    cutoff = log(cd_beta) + max(logits_main)
    result = diffs, with any position where logits_main < cutoff set to -inf

`cd_alpha=0` cancels the contrastive term exactly (`diffs == logits_main`). `cd_beta=0`
makes `cutoff == log(0) == -inf`, so the mask never fires (every finite logit is
>= -inf) -- at `cd_alpha=0, cd_beta=0` this function returns `logits_main` bit-for-bit.

Kept here (VCD's own tree) and imported by the LayerCD plugin too, rather than
duplicated, since the formula is identical in both repos and duplicating it would just
invite drift.
"""

import torch


def contrastive_combine(
    logits_main: torch.Tensor,
    logits_impaired: torch.Tensor,
    cd_alpha: float,
    cd_beta: float,
    force: bool = False,
) -> torch.Tensor:
    """Combine two logit streams via contrastive decoding.

    Args:
        logits_main: the "mature" stream's logits, shape `[..., vocab]`.
        logits_impaired: the "amateur"/impaired stream's logits, same shape.
        cd_alpha: contrastive strength (`0.0` disables the contrastive term).
        cd_beta: adaptive-plausibility-constraint coefficient (`0.0` disables the mask).
        force: run the formula even at the neutral parameters. The early return below is
            an optimisation; with it in place a neutral-point test proves only that the
            branch exists. `force=True` makes the arithmetic itself do the work, which
            is what the neutral test must measure.

    Returns:
        Combined logits, same shape as the inputs.
    """
    if cd_alpha == 0.0 and cd_beta == 0.0 and not force:
        return logits_main

    diffs = (1.0 + cd_alpha) * logits_main - cd_alpha * logits_impaired

    if cd_beta < 0.0:
        raise ValueError(f"cd_beta must be >= 0, got {cd_beta!r}")
    log_beta = torch.log(
        torch.as_tensor(cd_beta, dtype=logits_main.dtype, device=logits_main.device)
    )
    cutoff = log_beta + logits_main.max(dim=-1, keepdim=True).values
    diffs = diffs.masked_fill(logits_main < cutoff, float("-inf"))
    return diffs
