"""lmms-eval model `internvl3_5_hf_base`: unguided InternVL3.5-HF."""

from lmms_eval.protocol import ChatMessages
from PIL import Image

from vlmcfg.backbones.internvl3_5_hf import InternVL35HF
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class InternVL35HFBase(Qwen25VLBase):
    backbone_cls = InternVL35HF
    default_pretrained = "OpenGVLab/InternVL3_5-8B-HF"
    default_backbone_kwargs = {"attn_implementation": "sdpa", "device_map": "auto"}

    def _chat_inputs(self, chat: ChatMessages) -> tuple[list[str], dict]:
        images, _, _ = chat.extract_media()
        images = [_to_rgb(im) for im in images]
        return self.backbone.build_chat_inputs(chat.to_hf_messages(), images)


def _to_rgb(image) -> Image.Image:
    """The processor stacks tiles into one tensor, so every image must be RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")
