"""lmms-eval model `molmo2_pai`: PAI (Liu et al., ECCV 2024) on Molmo2."""

import contextlib

import torch
from transformers import LogitsProcessorList

from vlmcfg.baselines import pai
from vlmcfg.baselines.common.backbone import Molmo2Adapter
from vlmcfg.baselines.pai.cfg import CFGLogits
from vlmcfg.baselines.pai.molmo2_attention import Molmo2PAIIntervention
from vlmcfg.models.molmo2_base import Molmo2Base


class Molmo2PAI(Molmo2Base):
    """Amplify image attention in layers `pai_layers` by `pai_alpha`, and guide away
    from the image-free prompt with scale `pai_gamma` (1 turns that part off).
    """

    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-4B",
        pai_layers: str = pai.LAYERS,
        pai_alpha: float = pai.ALPHA,
        pai_gamma: float = pai.GAMMA,
        pai_apc: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        self.adapter = Molmo2Adapter()
        n_layers = len(self.adapter.text_layers(self.backbone.model))
        self.layer_range = pai.parse_layers(pai_layers, n_layers)
        self.alpha, self.gamma = float(pai_alpha), float(pai_gamma)
        self.apc = str(pai_apc).lower() not in ("0", "false", "no")

    @torch.no_grad()
    def _generate_one(self, text: str, images: list | None, max_new_tokens: int) -> str:
        bb = self.backbone
        inputs = bb.build_inputs(images, text)
        if not images:
            return bb.generate(inputs, max_new_tokens)
        mask = self.adapter.image_token_mask(
            inputs["input_ids"], self.adapter.image_token_id(bb.model)
        )
        handle = Molmo2PAIIntervention(*self.layer_range, mask, self.alpha)
        extra = {}
        if self.gamma != 1.0:
            uncond = bb.build_inputs(None, text)
            if "pixel_values" in uncond:
                raise RuntimeError("PAI's image-free inputs still carry pixel_values")
            extra["logits_processor"] = LogitsProcessorList(
                [
                    CFGLogits(
                        self.gamma,
                        uncond["input_ids"],
                        bb.model,
                        intervention=handle,
                        apc=self.apc,
                    )
                ]
            )
        with contextlib.ExitStack() as stack:
            handle.install(self.adapter.text_layers(bb.model))
            stack.callback(handle.remove)
            return bb.generate(inputs, max_new_tokens, **extra)
