"""lmms-eval model `qwen2_5_vl_sid`: SID (Huo et al., ICLR 2025) on Qwen2.5-VL."""

from vlmcfg.backbones.qwen2_5_vl import Qwen25VL
from vlmcfg.baselines import sid
from vlmcfg.baselines.sid.qwen2_5_vl import Qwen2_5_VLForConditionalGenerationSID
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class _Backbone(Qwen25VL):
    model_cls = Qwen2_5_VLForConditionalGenerationSID


class Qwen25VLSID(Qwen25VLBase):
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
        sid_attention_rank_ratio: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        sid.configure(
            self.backbone.model,
            sid_cd_alpha,
            sid_cd_beta,
            sid_agg_layer,
            sid_attention_rank,
            None
            if sid_attention_rank_ratio is None
            else float(sid_attention_rank_ratio),
        )
