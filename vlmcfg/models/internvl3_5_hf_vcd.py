"""lmms-eval model `internvl3_5_hf_vcd`: VCD (Leng et al., CVPR 2024) on
InternVL3.5-HF."""

from vlmcfg.backbones.internvl3_5_hf import InternVL35HF
from vlmcfg.baselines import vcd
from vlmcfg.baselines.vcd.internvl3_5_hf import InternVLForConditionalGenerationVCD
from vlmcfg.models.internvl3_5_hf_base import InternVL35HFBase


class _Backbone(InternVL35HF):
    model_cls = InternVLForConditionalGenerationVCD


class InternVL35HFVCD(InternVL35HFBase):
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
