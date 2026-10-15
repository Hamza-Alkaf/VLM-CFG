"""SID: Self-Introspective Decoding (Huo et al., ICLR 2025).

Contrast against an "amateur" stream that, after decoder layer K, keeps only the image
tokens the model attends to least (CT2S). Ported to Qwen2.5-VL, Qwen3-VL, InternVL3.5-HF
and Molmo2 from the method as described in the paper.
"""

# The published settings, used for every model in the paper.
CD_ALPHA, CD_BETA, AGG_LAYER, ATTENTION_RANK = 1.0, 0.1, 2, 100


def configure(
    model, cd_alpha: float, cd_beta: float, agg_layer: int, rank: int, rank_ratio
) -> None:
    """Set SID's hyperparameters on a SID model class (read in its `forward`)."""
    if int(agg_layer) < 1:
        raise ValueError(
            f"sid_agg_layer must be >= 1 (CT2S reads layer K-1), got {agg_layer}"
        )
    if int(rank) < 1:
        raise ValueError(f"sid_attention_rank must be >= 1, got {rank}")
    model.sid_cd_alpha = float(cd_alpha)
    model.sid_cd_beta = float(cd_beta)
    model.sid_agg_layer = int(agg_layer)
    model.sid_attention_rank = int(rank)
    model.sid_attention_rank_ratio = rank_ratio
