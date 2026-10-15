"""lmms-eval model `qwen3_vl_vlmcfg`: Qwen3-VL decoded with VLM-CFG (fp16, 1.6 MP
cap)."""

from vlmcfg.backbones.qwen3_vl import Qwen3VL
from vlmcfg.models.qwen2_5_vl_vlmcfg import Qwen25VLVLMCFG


class Qwen3VLVLMCFG(Qwen25VLVLMCFG):
    backbone_cls = Qwen3VL
    default_pretrained = "Qwen/Qwen3-VL-8B-Instruct"
    default_max_new_tokens = 50
