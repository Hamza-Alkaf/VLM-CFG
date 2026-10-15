"""lmms-eval model `qwen2_5_vl_acg`: ACG (attention-calibrated guidance) on
Qwen2.5-VL."""

from vlmcfg.backbones.qwen2_5_vl import Qwen25VL
from vlmcfg.baselines import acg
from vlmcfg.baselines.acg.qwen2_5_vl import Qwen2_5_VLForConditionalGenerationACG
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class _Backbone(Qwen25VL):
    model_cls = Qwen2_5_VLForConditionalGenerationACG


class Qwen25VLACG(Qwen25VLBase):
    """Recalibrate text-to-image attention in layers `acg_layers` (`all`, `[lo,hi]` or
    `lo-hi`) with strength `acg_gamma`. Defaults are the settings the paper used.
    """

    backbone_cls = _Backbone

    def __init__(
        self,
        pretrained: str | None = None,
        acg_gamma: float = acg.GAMMA,
        acg_layers: str = acg.LAYERS,
        acg_orth_scope: str = acg.ORTH_SCOPE,
        acg_eps: float = acg.EPS,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        acg.configure(
            self.backbone.model, acg_gamma, acg_layers, acg_orth_scope, acg_eps
        )
