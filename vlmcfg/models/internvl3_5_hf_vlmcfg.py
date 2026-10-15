"""lmms-eval model `internvl3_5_hf_vlmcfg`: InternVL3.5-HF decoded with VLM-CFG."""

from typing import Any

from vlmcfg.backbones.internvl3_5_hf import InternVL35HF
from vlmcfg.models.qwen2_5_vl_vlmcfg import Qwen25VLVLMCFG


class InternVL35HFVLMCFG(Qwen25VLVLMCFG):
    backbone_cls = InternVL35HF
    default_pretrained = "OpenGVLab/InternVL3_5-8B-HF"
    default_max_new_tokens = 64

    def _visual_input(self, visuals: list) -> Any:
        """Every image of the document (InternVL tiles each one)."""
        return visuals or None
