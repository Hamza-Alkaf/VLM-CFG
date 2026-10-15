"""lmms-eval model `qwen3_vl_m3id`: M3ID (Favero et al., CVPR 2024) on Qwen3-VL."""

import contextlib

from lmms_eval.protocol import ChatMessages

from vlmcfg.backbones.qwen3_vl import Qwen3VL, message_text
from vlmcfg.baselines import m3id
from vlmcfg.baselines.m3id.qwen3_vl import Qwen3VLForConditionalGenerationM3ID
from vlmcfg.models.qwen3_vl_base import Qwen3VLBase


class _Backbone(Qwen3VL):
    model_cls = Qwen3VLForConditionalGenerationM3ID


class Qwen3VLM3ID(Qwen3VLBase):
    """Amplify away from the image-free prompt with a weight growing with distance from
    the image. `m3id_lambda`: growth rate; `m3id_alpha`: confidence gate. Defaults are
    the settings the paper used.
    """

    backbone_cls = _Backbone

    def __init__(
        self,
        pretrained: str | None = None,
        m3id_lambda: float = m3id.LAMBDA,
        m3id_alpha: float = m3id.ALPHA,
        m3id_offset_mode: str = "tokens_to_image",
        m3id_t0_override: int = -1,
        m3id_max_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        m3id.configure(
            self.backbone.model,
            m3id_lambda,
            m3id_alpha,
            m3id_offset_mode,
            m3id_t0_override,
            m3id_max_weight,
        )

    def _generation_context(
        self, texts: list[str], inputs: dict, chat: ChatMessages, gen_kwargs: dict
    ):
        text = message_text(chat.to_hf_messages())
        if not text:
            raise ValueError(
                "M3ID's image-free prompt would be empty: the message has no text"
            )
        self.backbone.model.set_uncond_inputs(
            self.backbone.build_text_only_inputs(text)
        )
        return contextlib.nullcontext(), {}
