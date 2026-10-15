"""lmms-eval model `qwen2_5_vl_vcd`: VCD (Leng et al., CVPR 2024) on Qwen2.5-VL."""

from vlmcfg.backbones.qwen2_5_vl import Qwen25VL
from vlmcfg.baselines import vcd
from vlmcfg.baselines.vcd.qwen2_5_vl import Qwen2_5_VLForConditionalGenerationVCD
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class _Backbone(Qwen25VL):
    model_cls = Qwen2_5_VLForConditionalGenerationVCD


class Qwen25VLVCD(Qwen25VLBase):
    """Contrast each step against the same prompt with a diffusion-noised image.

    `vcd_cd_alpha`: contrast strength; `vcd_cd_beta`: plausibility cutoff;
    `vcd_noise_step`: noise timestep (0-999). Defaults are the published settings.
    """

    backbone_cls = _Backbone

    def __init__(
        self,
        pretrained: str | None = None,
        vcd_cd_alpha: float = vcd.CD_ALPHA,
        vcd_cd_beta: float = vcd.CD_BETA,
        vcd_noise_step: int = vcd.NOISE_STEP,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        vcd.configure(self.backbone.model, vcd_cd_alpha, vcd_cd_beta, vcd_noise_step)
