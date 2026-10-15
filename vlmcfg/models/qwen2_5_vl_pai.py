"""lmms-eval model `qwen2_5_vl_pai`: PAI (Liu et al., ECCV 2024) on Qwen2.5-VL."""

import contextlib

from lmms_eval.protocol import ChatMessages
from transformers import LogitsProcessorList

from vlmcfg.baselines import pai
from vlmcfg.baselines.pai.attention import find_image_span, qwen_modify
from vlmcfg.baselines.pai.cfg import CFGLogits, build_negative_prompt
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class Qwen25VLPAI(Qwen25VLBase):
    """Amplify image attention in layers `pai_layers` by `pai_alpha`, and guide away
    from the image-free prompt with scale `pai_gamma` (`<= 0` turns that part off).
    Defaults are the settings the paper used.
    """

    def __init__(
        self,
        pretrained: str | None = None,
        pai_layers: str = pai.LAYERS,
        pai_alpha: float = pai.ALPHA,
        pai_gamma: float = pai.GAMMA,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        n_layers = self.backbone.model.config.text_config.num_hidden_layers
        self.layer_range = pai.parse_layers(pai_layers, n_layers)
        self.alpha, self.gamma = float(pai_alpha), float(pai_gamma)

    def _generation_context(
        self, texts: list[str], inputs: dict, chat: ChatMessages, gen_kwargs: dict
    ):
        model = self.backbone.model
        img_lo, img_hi = find_image_span(
            inputs["input_ids"], model.config.image_token_id
        )
        handle = qwen_modify(
            model, *self.layer_range, True, self.alpha, False, img_lo, img_hi
        )
        extra = {}
        if self.gamma > 0:
            negative = build_negative_prompt(texts[0])
            neg_ids = self.backbone.tokenizer(
                negative, return_tensors="pt", add_special_tokens=False
            ).input_ids
            neg_ids = neg_ids.repeat(gen_kwargs["num_beams"], 1).to(model.device)
            processor = CFGLogits(self.gamma, neg_ids, model, intervention=handle)
            extra["logits_processor"] = LogitsProcessorList([processor])

        @contextlib.contextmanager
        def installed():
            try:
                yield
            finally:
                handle.remove()
                model._pai_intervention = None

        return installed(), extra
