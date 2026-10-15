"""lmms-eval model `internvl3_5_hf_layercd`: LayerCD (Tong et al., 2024) on
InternVL3.5-HF."""

from vlmcfg.backbones.internvl3_5_hf import InternVL35HF
from vlmcfg.baselines import layercd
from vlmcfg.baselines.layercd.internvl3_5_hf import (
    InternVLForConditionalGenerationLayerCD,
)
from vlmcfg.models.internvl3_5_hf_base import InternVL35HFBase


class _Backbone(InternVL35HF):
    model_cls = InternVLForConditionalGenerationLayerCD


class InternVL35HFLayerCD(InternVL35HFBase):
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
