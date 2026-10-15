"""lmms-eval model `qwen3_vl_vcd`: VCD (Leng et al., CVPR 2024) on Qwen3-VL."""

from vlmcfg.backbones.qwen3_vl import Qwen3VL
from vlmcfg.baselines import vcd
from vlmcfg.baselines.vcd.qwen3_vl import Qwen3VLForConditionalGenerationVCD
from vlmcfg.models.qwen3_vl_base import Qwen3VLBase


class _Backbone(Qwen3VL):
    model_cls = Qwen3VLForConditionalGenerationVCD


class Qwen3VLVCD(Qwen3VLBase):
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
