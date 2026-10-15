"""lmms-eval model `qwen2_5_vl_m3id`: M3ID (Favero et al., CVPR 2024) on Qwen2.5-VL."""

from lmms_eval.protocol import ChatMessages

from vlmcfg.backbones.qwen2_5_vl import Qwen25VL
from vlmcfg.baselines import m3id
from vlmcfg.baselines.m3id.qwen2_5_vl import Qwen2_5_VLForConditionalGenerationM3ID
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class _Backbone(Qwen25VL):
    model_cls = Qwen2_5_VLForConditionalGenerationM3ID


class Qwen25VLM3ID(Qwen25VLBase):
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

    def _chat_inputs(self, chat: ChatMessages) -> tuple[list[str], dict]:
        messages = chat.to_hf_messages()
        texts, inputs = self.backbone.build_chat_inputs(messages)
        _, uncond = self.backbone.build_chat_inputs(m3id.drop_visuals(messages))
        self.backbone.model.set_uncond_inputs(uncond)
        return texts, inputs
