"""lmms-eval model `qwen3_vl_base`: unguided Qwen3-VL (fp16, 1.6 MP cap)."""

from vlmcfg.backbones.qwen3_vl import Qwen3VL
from vlmcfg.models.qwen2_5_vl_base import Qwen25VLBase


class Qwen3VLBase(Qwen25VLBase):
    backbone_cls = Qwen3VL
    default_pretrained = "Qwen/Qwen3-VL-8B-Instruct"
    default_backbone_kwargs = {}
