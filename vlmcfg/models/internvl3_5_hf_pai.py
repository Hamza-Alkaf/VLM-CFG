"""lmms-eval model `internvl3_5_hf_pai`: PAI (Liu et al., ECCV 2024) on
InternVL3.5-HF."""

import contextlib

from lmms_eval.protocol import ChatMessages

from vlmcfg.backbones.internvl3_5_hf import InternVL35HF
from vlmcfg.baselines import pai
from vlmcfg.baselines.common.backbone import InternVL3Adapter
from vlmcfg.baselines.common.messages import drop_visuals
from vlmcfg.baselines.pai.cfg import CFGLogits
from vlmcfg.baselines.pai.internvl3_5_hf import InternVLForConditionalGenerationPAI
from vlmcfg.baselines.pai.internvl_attention import InternVLPAIIntervention
from vlmcfg.models.internvl3_5_hf_base import InternVL35HFBase


class _Backbone(InternVL35HF):
    model_cls = InternVLForConditionalGenerationPAI


class InternVL35HFPAI(InternVL35HFBase):
    """Amplify image attention in layers `pai_layers` by `pai_alpha`, and guide away
    from the image-free prompt with scale `pai_gamma` (1 turns that part off). Defaults
    are the settings the paper used.
    """

    backbone_cls = _Backbone

    def __init__(
        self,
        pretrained: str | None = None,
        pai_layers: str = pai.LAYERS,
        pai_alpha: float = pai.ALPHA,
        pai_gamma: float = pai.GAMMA,
        pai_apc: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        self.adapter = InternVL3Adapter()
        n_layers = len(self.adapter.text_layers(self.backbone.model))
        self.layer_range = pai.parse_layers(pai_layers, n_layers)
        self.alpha, self.gamma = float(pai_alpha), float(pai_gamma)
        self.apc = str(pai_apc).lower() not in ("0", "false", "no")

    def _generation_context(
        self, texts: list[str], inputs: dict, chat: ChatMessages, gen_kwargs: dict
    ):
        model = self.backbone.model
        span = self.adapter.image_span(
            inputs["input_ids"], int(model.config.image_token_id)
        )
        if span is None:
            return contextlib.nullcontext(), {}
        handle = InternVLPAIIntervention(*self.layer_range, *span, self.alpha)
        extra = {}
        if self.gamma != 1.0:
            _, uncond = self.backbone.build_chat_inputs(
                drop_visuals(chat.to_hf_messages()), []
            )
            if "pixel_values" in uncond:
                raise RuntimeError("PAI's image-free inputs still carry pixel_values")
            extra["logits_processor"] = [
                CFGLogits(
                    self.gamma,
                    uncond["input_ids"],
                    model,
                    intervention=handle,
                    apc=self.apc,
                )
            ]
        return self._installed(handle), extra

    @contextlib.contextmanager
    def _installed(self, handle):
        handle.install(self.adapter.text_layers(self.backbone.model))
        try:
            yield
        finally:
            handle.remove()
