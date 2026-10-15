"""LayerCD (Tong et al., 2024): contrast against the same image read out by a truncated
vision tower.

Port to Qwen2.5-VL, Qwen3-VL and InternVL3.5-HF of the decoding rule in the authors'
code (https://github.com/maifoundations/LayerCD, MIT). Not available for Molmo2, whose
vision features are read from fixed negative layer indices that a truncated tower would
remap.
"""

# The settings used for every model in the paper.
CD_ALPHA, CD_BETA, IMMATURE_DEPTH = 1.0, 0.1, 1


def configure(model, cd_alpha: float, cd_beta: float, immature_depth: int) -> None:
    """Set LayerCD's hyperparameters on a LayerCD model class (read in its
    `forward`)."""
    model.cd_alpha = float(cd_alpha)
    model.cd_beta = float(cd_beta)
    model.immature_depth = int(immature_depth)
