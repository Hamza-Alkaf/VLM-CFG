"""lmms-eval model `qwen3_vl_pai`: PAI (Liu et al., ECCV 2024) on Qwen3-VL."""

import contextlib

from lmms_eval.protocol import ChatMessages

from vlmcfg.backbones.qwen3_vl import Qwen3VL, message_text
from vlmcfg.baselines import pai
from vlmcfg.baselines.common.backbone import Qwen3VLAdapter
from vlmcfg.baselines.pai.cfg import CFGLogits
from vlmcfg.baselines.pai.qwen3_vl import Qwen3VLForConditionalGenerationPAI
from vlmcfg.baselines.pai.qwen3vl_attention import Qwen3VLPAIIntervention
from vlmcfg.models.qwen3_vl_base import Qwen3VLBase


class _Backbone(Qwen3VL):
    model_cls = Qwen3VLForConditionalGenerationPAI


class Qwen3VLPAI(Qwen3VLBase):
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
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        self.adapter = Qwen3VLAdapter()
        n_layers = len(self.adapter.text_layers(self.backbone.model))
        self.layer_range = pai.parse_layers(pai_layers, n_layers)
        self.alpha, self.gamma = float(pai_alpha), float(pai_gamma)

    def _generation_context(
        self, texts: list[str], inputs: dict, chat: ChatMessages, gen_kwargs: dict
    ):
        model = self.backbone.model
        span = self.adapter.image_span(
            inputs["input_ids"], int(model.config.image_token_id)
        )
        if span is None:
            return contextlib.nullcontext(), {}
        handle = Qwen3VLPAIIntervention(*self.layer_range, *span, self.alpha)
        extra = {}
        if self.gamma != 1.0:
            text = message_text(chat.to_hf_messages())
            if not text:
                return self._installed(handle), {}
            uncond = self.backbone.build_text_only_inputs(text)
            extra["logits_processor"] = [
                CFGLogits(self.gamma, uncond["input_ids"], model, intervention=handle)
            ]
        return self._installed(handle), extra

    @contextlib.contextmanager
    def _installed(self, handle):
        handle.install(self.adapter.text_layers(self.backbone.model))
        try:
            yield
        finally:
            handle.remove()
