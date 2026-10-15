"""SID on Molmo2: CT2S over Molmo2's (non-contiguous) image-patch columns.

Molmo2 is remote code, so the amateur stream is one backbone step with the CT2S mask
installed for exactly that forward.
"""

import torch

from vlmcfg.backbones.molmo2 import Molmo2
from vlmcfg.baselines.common.backbone import Molmo2Adapter
from vlmcfg.baselines.sid.internvl3_5_hf import (
    InternVLCt2sImageMask,
    InternVLLastRowCapture,
)
from vlmcfg.decoding import Stream

ADAPTER = Molmo2Adapter()


class Molmo2Ct2sImageMask(InternVLCt2sImageMask):
    """CT2S over an explicit (non-contiguous) set of image-patch columns."""

    def __init__(
        self,
        cols: torch.Tensor,
        rank: int,
        agg_layer: int,
        capture: InternVLLastRowCapture,
    ) -> None:
        super().__init__(0, int(cols.numel()), rank, agg_layer, capture)
        self.cols = cols

    def _select(self, device) -> torch.Tensor:
        full = self.capture.rows.get(self.agg_layer - 1)
        if full is None:
            raise RuntimeError(
                f"SID: layer {self.agg_layer - 1}'s attention row was not captured "
                f"before layer {self.agg_layer} ran; CT2S cannot select tokens."
            )
        row = full[self.cols.cpu()]
        k = min(self.rank, row.numel())
        keep = torch.topk(
            row, k, largest=False
        ).indices  # LEAST attended: SID, not FastV
        kill = torch.ones(row.numel(), dtype=torch.bool)
        kill[keep] = False
        self.n_selected, self.n_image_tokens = int(k), int(row.numel())
        return self.cols[kill.to(self.cols.device)].to(device)


def image_columns(input_ids: torch.Tensor) -> torch.Tensor:
    """Positions of the image-patch tokens in a prompt."""
    return ADAPTER.image_token_mask(input_ids).nonzero(as_tuple=True)[0]


def amateur_step(
    backbone: Molmo2,
    stream: Stream,
    first: bool,
    cols: torch.Tensor,
    rank: int,
    agg_layer: int,
) -> torch.Tensor:
    """One forward of the amateur stream, with CT2S installed for exactly that
    forward."""
    layers = ADAPTER.text_layers(backbone.model)
    capture = InternVLLastRowCapture(0, None, adapter=ADAPTER)  # the whole row
    ct2s = Molmo2Ct2sImageMask(cols, rank, agg_layer, capture)
    ct2s.reset_pass()
    capture.install([layers[agg_layer - 1]])
    ct2s.install(layers)
    try:
        return backbone.step(stream, first, "logits")
    finally:
        ct2s.remove()
        capture.remove()
