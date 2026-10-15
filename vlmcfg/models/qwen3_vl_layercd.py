"""lmms-eval model `qwen3_vl_layercd`: LayerCD (Tong et al., 2024) on Qwen3-VL."""

from vlmcfg.backbones.qwen3_vl import Qwen3VL
from vlmcfg.baselines import layercd
from vlmcfg.baselines.layercd.qwen3_vl import Qwen3VLForConditionalGenerationLayerCD
from vlmcfg.models.qwen3_vl_base import Qwen3VLBase


class _Backbone(Qwen3VL):
    model_cls = Qwen3VLForConditionalGenerationLayerCD


class Qwen3VLLayerCD(Qwen3VLBase):
    """Contrast each step against the image read out after `layercd_immature_depth`
    vision blocks. `layercd_cd_alpha`: contrast strength; `layercd_cd_beta`:
    plausibility cutoff. Defaults are the settings the paper used.
    """

    backbone_cls = _Backbone

    def __init__(
        self,
        pretrained: str | None = None,
        layercd_cd_alpha: float = layercd.CD_ALPHA,
        layercd_cd_beta: float = layercd.CD_BETA,
        layercd_immature_depth: int = layercd.IMMATURE_DEPTH,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        layercd.configure(
            self.backbone.model,
            layercd_cd_alpha,
            layercd_cd_beta,
            layercd_immature_depth,
        )
