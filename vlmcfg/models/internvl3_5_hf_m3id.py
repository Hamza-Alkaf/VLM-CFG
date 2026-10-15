"""lmms-eval model `internvl3_5_hf_m3id`: M3ID (Favero et al., CVPR 2024) on
InternVL3.5-HF."""

from lmms_eval.protocol import ChatMessages

from vlmcfg.backbones.internvl3_5_hf import InternVL35HF
from vlmcfg.baselines import m3id
from vlmcfg.baselines.m3id.internvl3_5_hf import InternVLForConditionalGenerationM3ID
from vlmcfg.models.internvl3_5_hf_base import InternVL35HFBase, _to_rgb


class _Backbone(InternVL35HF):
    model_cls = InternVLForConditionalGenerationM3ID


class InternVL35HFM3ID(InternVL35HFBase):
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
        images, _, _ = chat.extract_media()
        messages = chat.to_hf_messages()
        texts, inputs = self.backbone.build_chat_inputs(
            messages, [_to_rgb(i) for i in images]
        )
        _, uncond = self.backbone.build_chat_inputs(m3id.drop_visuals(messages), [])
        if "pixel_values" in uncond:
            raise RuntimeError("M3ID's image-free inputs still carry pixel_values")
        self.backbone.model.set_uncond_inputs(uncond)
        return texts, inputs
