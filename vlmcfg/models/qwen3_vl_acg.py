"""lmms-eval model `qwen3_vl_acg`: ACG (attention-calibrated guidance) on Qwen3-VL."""

from vlmcfg.backbones.qwen3_vl import Qwen3VL
from vlmcfg.baselines import acg
from vlmcfg.baselines.acg.qwen3_vl import Qwen3VLForConditionalGenerationACG
from vlmcfg.models.qwen3_vl_base import Qwen3VLBase


class _Backbone(Qwen3VL):
    model_cls = Qwen3VLForConditionalGenerationACG


class Qwen3VLACG(Qwen3VLBase):
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
