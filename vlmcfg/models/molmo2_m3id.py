"""lmms-eval model `molmo2_m3id`: M3ID (Favero et al., CVPR 2024) on Molmo2."""

import torch

from vlmcfg.baselines import m3id
from vlmcfg.baselines.m3id.combine import m3id_combine
from vlmcfg.baselines.m3id.molmo2 import tokens_after_image
from vlmcfg.models.molmo2_base import Molmo2Base


class Molmo2M3ID(Molmo2Base):
    """Amplify away from the image-free stream, more as the answer grows."""

    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-4B",
        m3id_lambda: float = m3id.LAMBDA,
        m3id_alpha: float = m3id.ALPHA,
        m3id_max_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        self.lam, self.alpha = float(m3id_lambda), float(m3id_alpha)
        self.max_weight = float(m3id_max_weight)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"m3id_alpha must be in (0, 1], got {self.alpha}")

    @torch.no_grad()
    def _generate_one(self, text: str, images: list | None, max_new_tokens: int) -> str:
        bb = self.backbone
        inputs = bb.build_inputs(images, text)
        t0 = tokens_after_image(inputs["input_ids"])
        # without an image the two streams coincide and the correction is zero
        contrast = t0 >= 0 and self.lam != 0.0
        streams = [bb.new_stream(inputs)]
        if contrast:
            streams.append(bb.new_stream(bb.build_inputs(None, text)))
        out: list[int] = []
        for step in range(max_new_tokens):
            logits = bb.step(streams[0], step == 0, "logits")
            if contrast:
                uncond = bb.step(streams[1], step == 0, "logits")
                logits = m3id_combine(
                    logits,
                    uncond,
                    t=t0 + step,
                    lam=self.lam,
                    alpha=self.alpha,
                    max_weight=self.max_weight,
                )
            token = int(logits.argmax(dim=-1).item())
            if token in bb.eos_ids:
                break
            out.append(token)
            for stream in streams:
                stream.append(token)
        return bb.decode(out)
