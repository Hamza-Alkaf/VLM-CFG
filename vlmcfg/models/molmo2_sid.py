"""lmms-eval model `molmo2_sid`: SID (Huo et al., ICLR 2025) on Molmo2."""

import torch

from vlmcfg.baselines import sid
from vlmcfg.baselines.sid.molmo2 import ADAPTER, amateur_step, image_columns
from vlmcfg.baselines.vcd.combine import contrastive_combine
from vlmcfg.models.molmo2_base import Molmo2Base


class Molmo2SID(Molmo2Base):
    """Contrast each step against a stream that keeps only the least-attended image
    patches."""

    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-4B",
        sid_cd_alpha: float = sid.CD_ALPHA,
        sid_cd_beta: float = sid.CD_BETA,
        sid_agg_layer: int = sid.AGG_LAYER,
        sid_attention_rank: int = sid.ATTENTION_RANK,
        sid_attention_rank_ratio: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        self.cd_alpha, self.cd_beta = float(sid_cd_alpha), float(sid_cd_beta)
        self.agg_layer = int(sid_agg_layer)
        self.attention_rank, self.rank_ratio = (
            int(sid_attention_rank),
            float(sid_attention_rank_ratio),
        )
        n_layers = len(ADAPTER.text_layers(self.backbone.model))
        if not 0 < self.agg_layer < n_layers:
            raise ValueError(
                f"sid_agg_layer must be in [1, {n_layers - 1}], got {self.agg_layer}"
            )

    def _resolve_rank(self, n_img: int) -> int:
        if self.rank_ratio > 0.0:
            return max(1, int(round(self.rank_ratio * n_img)))
        return self.attention_rank

    @torch.no_grad()
    def _generate_one(self, text: str, images: list | None, max_new_tokens: int) -> str:
        bb = self.backbone
        inputs = bb.build_inputs(images, text)
        cols = image_columns(inputs["input_ids"])
        contrast = cols.numel() > 0 and not (
            self.cd_alpha == 0.0 and self.cd_beta == 0.0
        )
        streams = [bb.new_stream(inputs)]
        if contrast:
            streams.append(bb.new_stream(inputs))
        rank = self._resolve_rank(int(cols.numel()))
        out: list[int] = []
        for step in range(max_new_tokens):
            logits = bb.step(streams[0], step == 0, "logits")
            if contrast:
                amateur = amateur_step(
                    bb, streams[1], step == 0, cols, rank, self.agg_layer
                )
                logits = contrastive_combine(
                    logits, amateur, self.cd_alpha, self.cd_beta
                )
            token = int(logits.argmax(dim=-1).item())
            if token in bb.eos_ids:
                break
            out.append(token)
            for stream in streams:
                stream.append(token)
        return bb.decode(out)
