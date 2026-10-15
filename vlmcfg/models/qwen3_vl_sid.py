"""lmms-eval model `qwen3_vl_sid`: SID (Huo et al., ICLR 2025) on Qwen3-VL."""

from vlmcfg.backbones.qwen3_vl import Qwen3VL
from vlmcfg.baselines import sid
from vlmcfg.baselines.sid.qwen3_vl import Qwen3VLForConditionalGenerationSID
from vlmcfg.models.qwen3_vl_base import Qwen3VLBase


class _Backbone(Qwen3VL):
    model_cls = Qwen3VLForConditionalGenerationSID


class Qwen3VLSID(Qwen3VLBase):
    """Contrast each step against a stream that, after layer `sid_agg_layer`, keeps only
    the `sid_attention_rank` least-attended image tokens. `sid_cd_alpha`: contrast
    strength; `sid_cd_beta`: plausibility cutoff; `sid_attention_rank_ratio` (off by
    default) keeps a fraction of the image tokens instead. Defaults are the published
    settings.
    """

    backbone_cls = _Backbone

    def __init__(
        self,
        pretrained: str | None = None,
        sid_cd_alpha: float = sid.CD_ALPHA,
        sid_cd_beta: float = sid.CD_BETA,
        sid_agg_layer: int = sid.AGG_LAYER,
        sid_attention_rank: int = sid.ATTENTION_RANK,
        sid_attention_rank_ratio: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        sid.configure(
            self.backbone.model,
            sid_cd_alpha,
            sid_cd_beta,
            sid_agg_layer,
            sid_attention_rank,
            float(sid_attention_rank_ratio),
        )
