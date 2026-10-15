"""M3ID: Multi-Modal Mutual-Information Decoding (Favero et al., CVPR 2024).

Amplify the conditional stream away from an image-free stream, with a weight that grows
with the distance from the image; only the decoding half (Eq. 3-4) is used.
Re-implemented from the paper, as no code was released.
"""

from vlmcfg.baselines.common.messages import drop_visuals  # noqa: F401  (re-export)
from vlmcfg.baselines.m3id.qwen2_5_vl import OFFSET_MODES

# The settings used for every model in the paper.
LAMBDA, ALPHA = 0.02, 0.3


def configure(
    model,
    lam: float,
    alpha: float,
    offset_mode: str,
    t0_override: int,
    max_weight: float,
) -> None:
    """Set M3ID's hyperparameters on an M3ID model class (read in its `forward`)."""
    if float(lam) < 0.0:
        raise ValueError(f"m3id_lambda must be >= 0, got {lam}")
    if float(lam) > 0.0 and not 0.0 < float(alpha) <= 1.0:
        raise ValueError(f"m3id_alpha must be in (0, 1], got {alpha}")
    if offset_mode not in OFFSET_MODES:
        raise ValueError(
            f"m3id_offset_mode must be one of {OFFSET_MODES}, got {offset_mode!r}"
        )
    model.m3id_lambda = float(lam)
    model.m3id_alpha = float(alpha)
    model.m3id_offset_mode = str(offset_mode)
    model.m3id_t0_override = int(t0_override)
    model.m3id_max_weight = float(max_weight)
