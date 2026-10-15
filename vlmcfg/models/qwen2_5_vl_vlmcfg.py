"""lmms-eval model `qwen2_5_vl_vlmcfg`: Qwen2.5-VL decoded with VLM-CFG."""

from typing import Any

from lmms_eval.api.instance import Instance
from tqdm import tqdm

from vlmcfg.backbones.qwen2_5_vl import Qwen25VL
from vlmcfg.decoding import guided_decode
from vlmcfg.guidance import GuidanceConfig
from vlmcfg.models._common import GenerateOnly, flatten
from vlmcfg.negatives import (
    NegativeConfig,
    RegionConfig,
    conditional_context,
    negative_stream,
)


class Qwen25VLVLMCFG(GenerateOnly):
    """Qwen2.5-VL with classifier-free guidance on the image.

    With no `--model_args` this runs the paper configuration: Qwen2.5-VL-7B, image
    deletion, hidden-state blend, alpha 1.5, switch at K=8, APC beta 0.1. Keywords named
    after a `GuidanceConfig` field (alpha, reference, rule, guard, ...) set the
    guidance, those named after a `NegativeConfig` field (negative, attn_op, exit_layer,
    ...) the negative stream and those of a `RegionConfig` (region, cond_attn_op, ...)
    the annotated region; any other goes to the backbone (e.g. `max_pixels`).
    """

    is_simple = True
    backbone_cls: Any = Qwen25VL
    default_pretrained = "Qwen/Qwen2.5-VL-7B-Instruct"
    default_max_new_tokens = 64

    def __init__(
        self, pretrained: str | None = None, batch_size: int = 1, **kwargs: Any
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError(
                "guided decoding runs one document at a time (batch_size=1)"
            )
        self.cfg, kwargs = GuidanceConfig.from_args(kwargs)
        self.neg_cfg, kwargs = NegativeConfig.from_args(kwargs)
        self.region, backbone_kwargs = RegionConfig.from_args(kwargs)
        edited = self.neg_cfg.negative != "drop" or self.region.cond_attn_op != "none"
        if edited and self.backbone_cls is not Qwen25VL:
            raise NotImplementedError(
                "weakened negatives and attention edits are Qwen2.5-VL only"
            )
        self.backbone = self.backbone_cls(
            pretrained or self.default_pretrained, **backbone_kwargs
        )
        self.weak = None
        if self.neg_cfg.negative == "weak_model":
            if self.cfg.feature == "hidden":
                raise ValueError(
                    "a weak model differs in hidden size: use feature=logits"
                )
            self.weak = self.backbone_cls(self.neg_cfg.weak_model, **backbone_kwargs)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        answers = []
        for ctx, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm(
            [r.args for r in requests], disable=self.rank != 0, desc="VLM-CFG"
        ):
            doc = self.task_dict[task][split][doc_id]
            image = self._visual_input(flatten([doc_to_visual(doc)]))
            max_new = gen_kwargs.get("max_new_tokens") or self.default_max_new_tokens
            region = (self.region, doc, doc_id)
            neg = None
            if (
                self.neg_cfg.negative == "region"
            ):  # its own inputs, built first as in the paper runs
                neg = negative_stream(
                    self.backbone, None, ctx, self.neg_cfg, image, *region
                )
            cond_inputs = self.backbone.build_inputs(image, ctx)
            cond = self.backbone.new_stream(cond_inputs)
            edit = conditional_context(self.backbone, cond_inputs, *region, image)
            if edit is not None:
                cond.state["context"] = edit
            if neg is None:
                neg = negative_stream(self.backbone, cond_inputs, ctx, self.neg_cfg)
            ids = guided_decode(
                self.backbone, cond, neg, int(max_new), self.cfg, neg_backbone=self.weak
            )
            answers.append(self._finalize(self.backbone.decode(ids), gen_kwargs))
        return answers

    def _visual_input(self, visuals: list) -> Any:
        """What the backbone takes as the image: the first one, as the paper runs
        did."""
        return visuals[0] if visuals else None

    def _finalize(self, answer: str, gen_kwargs: dict) -> str:
        """Post-process a decoded answer; the paper's Qwen and InternVL runs used it as
        is."""
        return answer
