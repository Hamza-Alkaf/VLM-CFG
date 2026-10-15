"""lmms-eval model `qwen2_5_vl_layercd`: LayerCD (Tong et al., 2024) on Qwen2.5-VL."""

from vlmcfg.backbones.qwen2_5_vl import Qwen25VL
from vlmcfg.baselines import layercd
from vlmcfg.baselines.layercd.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGenerationLayerCD,
)
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class _Backbone(Qwen25VL):
    model_cls = Qwen2_5_VLForConditionalGenerationLayerCD


class Qwen25VLLayerCD(Qwen25VLBase):
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
