"""lmms-eval model `molmo2_vcd`: VCD (Leng et al., CVPR 2024) on Molmo2."""

import torch

from vlmcfg.baselines import vcd
from vlmcfg.baselines.vcd.combine import contrastive_combine
from vlmcfg.baselines.vcd.noise import add_diffusion_noise
from vlmcfg.models.molmo2_base import Molmo2Base


class Molmo2VCD(Molmo2Base):
    """Contrast each step against the same prompt with a diffusion-noised image.

    Molmo2 is remote code, so VCD runs as its own two-stream loop over the backbone's
    stream primitives instead of inside the model's `forward`.
    """

    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-4B",
        vcd_cd_alpha: float = vcd.CD_ALPHA,
        vcd_cd_beta: float = vcd.CD_BETA,
        vcd_noise_step: int = vcd.NOISE_STEP,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        if not 0 <= int(vcd_noise_step) <= 999:
            raise ValueError(
                f"vcd_noise_step must be in [0, 999], got {vcd_noise_step}"
            )
        self.cd_alpha, self.cd_beta = float(vcd_cd_alpha), float(vcd_cd_beta)
        self.noise_step = int(vcd_noise_step)

    @torch.no_grad()
    def _generate_one(self, text: str, images: list | None, max_new_tokens: int) -> str:
        bb = self.backbone
        inputs = bb.build_inputs(images, text)
        contrast = bool(images) and not (self.cd_alpha == 0.0 and self.cd_beta == 0.0)
        streams = [bb.new_stream(inputs)]
        if contrast:
            pixels = inputs["pixel_values"]
            noisy = add_diffusion_noise(pixels, self.noise_step).to(pixels.dtype)
            streams.append(bb.new_stream({**inputs, "pixel_values": noisy}))
        out: list[int] = []
        for step in range(max_new_tokens):
            logits = bb.step(streams[0], step == 0, "logits")
            if contrast:
                noisy_logits = bb.step(streams[1], step == 0, "logits")
                logits = contrastive_combine(
                    logits, noisy_logits, self.cd_alpha, self.cd_beta
                )
            token = int(logits.argmax(dim=-1).item())
            if token in bb.eos_ids:
                break
            out.append(token)
            for stream in streams:
                stream.append(token)
        return bb.decode(out)
