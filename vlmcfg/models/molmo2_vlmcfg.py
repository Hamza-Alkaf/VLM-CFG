"""lmms-eval model `molmo2_vlmcfg`: Molmo2 decoded with VLM-CFG (fp16)."""

from typing import Any

from vlmcfg.backbones.molmo2 import Molmo2
from vlmcfg.models._common import cut_at_stop_strings
from vlmcfg.models.qwen2_5_vl_vlmcfg import Qwen25VLVLMCFG


class Molmo2VLMCFG(Qwen25VLVLMCFG):
    backbone_cls = Molmo2
    default_pretrained = "allenai/Molmo2-4B"
    default_max_new_tokens = 64

    def _visual_input(self, visuals: list) -> Any:
        return visuals or None

    def _finalize(self, answer: str, gen_kwargs: dict) -> str:
        return cut_at_stop_strings(answer, gen_kwargs)
