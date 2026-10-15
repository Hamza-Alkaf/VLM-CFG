"""ACG: Attention-Calibrated Guidance (Algorithm 1), re-implemented from the paper.

Reweights each text token's attention to the image inside the selected decoder layers,
removing the component parallel to the unmodified output. No upstream code was released.
"""

from vlmcfg.baselines.acg.attention import ORTH_SCOPES

# The settings used for every model in the paper: ACG-Full (all layers), gamma 1.4.
GAMMA, LAYERS, ORTH_SCOPE, EPS = 1.4, "all", "concat", 1e-6


def parse_layers(spec: str, num_layers: int) -> tuple[int, int]:
    """`"all"`, `"[start,end]"` or `"start-end"` -> half-open `(start, end)`.

    The dash form exists because `sbatch --export` splits on commas.
    """
    s = str(spec).strip()
    if s == "all":
        return 0, num_layers
    if s.startswith("[") and s.endswith("]"):
        parts = s[1:-1].split(",")
    elif "-" in s:
        parts = s.split("-", 1)
    else:
        raise ValueError(
            f"acg_layers must be 'all', '[start,end]' or 'start-end', got {spec!r}"
        )
    if len(parts) != 2:
        raise ValueError(f"acg_layers must have exactly two bounds, got {spec!r}")
    start, end = int(parts[0]), int(parts[1])
    if not 0 <= start < end <= num_layers:
        raise ValueError(f"acg_layers {spec!r} is not within [0, {num_layers}]")
    return start, end


def configure(model, gamma: float, layers: str, orth_scope: str, eps: float) -> None:
    """Set ACG's hyperparameters on an ACG model class (read when it installs the
    hook)."""
    if orth_scope not in ORTH_SCOPES:
        raise ValueError(
            f"acg_orth_scope must be one of {ORTH_SCOPES}, got {orth_scope!r}"
        )
    if float(eps) <= 0.0:
        raise ValueError(f"acg_eps must be > 0, got {eps}")
    n_layers = len(model.model.language_model.layers)
    model.acg_gamma = float(gamma)
    model.acg_layer_range = parse_layers(layers, n_layers)
    model.acg_orth_scope = str(orth_scope)
    model.acg_eps = float(eps)
