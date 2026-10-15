"""lmms-eval model `internvl3_5_hf_acg`: ACG (attention-calibrated guidance) on
InternVL3.5-HF."""

from vlmcfg.backbones.internvl3_5_hf import InternVL35HF
from vlmcfg.baselines import acg
from vlmcfg.baselines.acg.internvl3_5_hf import InternVLForConditionalGenerationACG
from vlmcfg.models.internvl3_5_hf_base import InternVL35HFBase


class _Backbone(InternVL35HF):
    model_cls = InternVLForConditionalGenerationACG


class InternVL35HFACG(InternVL35HFBase):
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
