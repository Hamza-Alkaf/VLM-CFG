"""What the two streams see.

The negative stream deletes the image by default (the paper's method). It can instead
weaken the model -- edit its attention to the image, exit early, or swap in a smaller
model, all on the conditional's own inputs -- or black out an annotated region of the
image (the localized negative). Separately, the conditional stream's attention can be
steered towards (or away from) the annotated region.
"""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vlmcfg.args import split_args
from vlmcfg.baselines.common.tokens import (
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    find_pattern_starts,
)
from vlmcfg.decoding import Stream
from vlmcfg.negatives.attention import AttentionIntervention, parse_layer_range
from vlmcfg.negatives.regions import (
    black_out_region,
    random_region,
    region_pixel_mask,
    region_token_mask,
)

NEGATIVES = ("drop", "attention", "early_exit", "weak_model", "region")
# "masked" removes the image columns from the softmax;
# the rest are AttentionIntervention ops
ATTENTION_OPS = ("mul", "blur", "sharpen", "peak", "masked")
REGION_KINDS = ("", "bbox", "seg")
COND_ATTENTION_OPS = ("none", "mul", "add", "abs")

Context = Callable[[], contextlib.AbstractContextManager]


@dataclass(frozen=True)
class NegativeConfig:
    """How the negative stream is built.

    Attributes:
        negative: "drop" deletes the image; "attention" edits the attention to it;
            "early_exit" reads the model out after `exit_layer` layers; "weak_model"
            runs `weak_model` (a smaller checkpoint of the same family) on the inputs;
            "region" blacks out the annotated region (see RegionConfig).
        attn_op: the attention edit, see ATTENTION_OPS.
        attn_gamma: MUL's scale on the image scores, (1 + gamma).
        attn_layers: "[lo-hi]" half-open layer range; "" is every layer but the first.
        attn_renorm: "global", or "preserve_ratio" to hold the image's attention mass.
        attn_blur_sigma: SEG/Over-blur kernel width in patches.
        attn_blur_sigma_frac: the same as a fraction of the grid's short side
            (per image).
        attn_alpha: Over-blur's a in p_SEG + a (p - p_SEG).
        attn_peak_lambda: Soft PAG's lambda; 1 is PAG.
        exit_layer: early exit after this many decoder layers.
        weak_model: checkpoint of the weak model.
    """

    negative: str = "drop"
    attn_op: str = "mul"
    attn_gamma: float = -0.5
    attn_layers: str = ""
    attn_renorm: str = "global"
    attn_blur_sigma: float = 0.0
    attn_blur_sigma_frac: float = 0.0
    attn_alpha: float = 1.0
    attn_peak_lambda: float = 0.0
    exit_layer: int = 15
    weak_model: str = ""

    def __post_init__(self) -> None:
        if self.negative not in NEGATIVES:
            raise ValueError(
                f"negative must be one of {NEGATIVES}, got {self.negative!r}"
            )
        if self.attn_op not in ATTENTION_OPS:
            raise ValueError(
                f"attn_op must be one of {ATTENTION_OPS}, got {self.attn_op!r}"
            )
        if self.negative == "weak_model" and not self.weak_model:
            raise ValueError("negative=weak_model needs weak_model=<checkpoint>")

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> tuple["NegativeConfig", dict[str, Any]]:
        """Split command-line `args` into a config and the remaining keywords."""
        return split_args(cls, args)


@dataclass(frozen=True)
class RegionConfig:
    """An annotated region, and the conditional stream's attention edit on it.

    Attributes:
        region: "bbox" or "seg", the kind of annotation in `region_field`; "" for none.
        region_field: the task column holding it.
        random_region: replace it with a random rectangle (10-50% of each side).
        region_seed: seed of the random rectangle (keyed per document).
        cond_attn_op: "mul" scales (A (1 + g)), "add" shifts (A + g) and "abs" applies
            A + g |A| to the conditional's image scores: g = `cond_attn_gamma` inside
            the region (or on the whole image without one),
            `cond_attn_gamma2` outside it.
        cond_attn_layers: "[lo-hi]" half-open layer range;
            "" is every layer but the first.
        cond_attn_renorm: "global"; "preserve_ratio" to hold the image's attention mass;
            "cap_ratio" to let it rise no higher than `cond_attn_cap`.
        cond_attn_cap: the image-mass ceiling of cap_ratio.
    """

    region: str = ""
    region_field: str = ""
    random_region: bool = False
    region_seed: int = 0
    cond_attn_op: str = "none"
    cond_attn_gamma: float = 0.0
    cond_attn_gamma2: float = 0.0
    cond_attn_layers: str = ""
    cond_attn_renorm: str = "global"
    cond_attn_cap: float = 1.0

    def __post_init__(self) -> None:
        if self.region not in REGION_KINDS:
            raise ValueError(
                f"region must be one of {REGION_KINDS}, got {self.region!r}"
            )
        if self.cond_attn_op not in COND_ATTENTION_OPS:
            raise ValueError(f"cond_attn_op must be one of {COND_ATTENTION_OPS}")
        if self.region and not (self.region_field or self.random_region):
            raise ValueError("region needs region_field (or random_region)")
        if self.random_region and not self.region:
            raise ValueError("random_region needs region=bbox|seg")

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> tuple["RegionConfig", dict[str, Any]]:
        """Split command-line `args` into a config and the remaining keywords."""
        return split_args(cls, args)

    def value(self, doc: dict, doc_id: Any) -> Any:
        """This document's region annotation (or its random replacement)."""
        if self.random_region:
            return random_region(self.region, self.region_seed, doc_id)
        if self.region_field not in doc:
            raise ValueError(f"the task has no {self.region_field!r} column")
        return doc[self.region_field]


def negative_stream(
    backbone: Any,
    cond_inputs: dict | None,
    text: str,
    cfg: NegativeConfig,
    image: Any = None,
    region: RegionConfig | None = None,
    doc: dict | None = None,
    doc_id: Any = None,
) -> Stream:
    """The negative stream for one document.

    The weakened negatives reuse `cond_inputs`; the localized negative needs `image` and
    the document's region (it builds its own inputs, so `cond_inputs` may be None).
    """
    if cfg.negative == "drop":
        return backbone.new_stream(backbone.build_inputs(None, text))
    if cfg.negative == "region":
        assert region is not None and region.region and doc is not None
        mask = region_pixel_mask(
            region.region, region.value(doc, doc_id), image.size, image.size
        )
        return backbone.new_stream(
            backbone.build_inputs(black_out_region(image, mask), text)
        )
    assert cond_inputs is not None
    stream = backbone.new_stream(cond_inputs)
    if cfg.negative == "early_exit":
        stream.state["exit_layer"] = cfg.exit_layer
    elif cfg.negative == "attention":
        masked = cfg.attn_op == "masked"
        stream.state["context"] = attention_context(
            backbone,
            cond_inputs,
            op="mul" if masked else cfg.attn_op,
            ablate=masked,
            gamma=cfg.attn_gamma,
            layers=cfg.attn_layers,
            renorm=cfg.attn_renorm,
            blur_sigma=cfg.attn_blur_sigma,
            blur_sigma_frac=cfg.attn_blur_sigma_frac,
            alpha=cfg.attn_alpha,
            peak_lambda=cfg.attn_peak_lambda,
        )
    return stream


def conditional_context(
    backbone: Any,
    inputs: dict,
    region: RegionConfig,
    doc: dict,
    doc_id: Any,
    image: Any,
) -> Context | None:
    """The conditional stream's attention edit on the region, or None when off."""
    if region.cond_attn_op == "none":
        return None
    mask = None
    if region.region:
        grid = inputs["image_grid_thw"][0]
        mask = region_token_mask(
            region.region, region.value(doc, doc_id), grid, image.size
        )
    return attention_context(
        backbone,
        inputs,
        op=region.cond_attn_op,
        gamma=region.cond_attn_gamma,
        gamma2=region.cond_attn_gamma2,
        region_mask=mask,
        layers=region.cond_attn_layers,
        renorm=region.cond_attn_renorm,
        img_cap_ratio=region.cond_attn_cap,
    )


def attention_context(
    backbone: Any,
    inputs: dict,
    op: str,
    gamma: float,
    layers: str = "",
    renorm: str = "global",
    img_cap_ratio: float = 1.0,
    ablate: bool = False,
    gamma2: float = 0.0,
    region_mask: torch.Tensor | None = None,
    blur_sigma: float = 0.0,
    blur_sigma_frac: float = 0.0,
    alpha: float = 1.0,
    peak_lambda: float = 0.0,
) -> Context:
    """A factory of contexts installing one attention edit around a forward."""
    model_layers = backbone.model.language_model.layers
    lo, hi = parse_layer_range(layers or f"[1-{len(model_layers)}]", len(model_layers))
    starts = find_pattern_starts(inputs["input_ids"], [VISION_START_TOKEN_ID])[0]
    ends = find_pattern_starts(inputs["input_ids"], [VISION_END_TOKEN_ID])[0]
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError("an attention edit needs exactly one image in the prompt")
    img_grid, sigma = None, blur_sigma
    if op in ("blur", "sharpen"):
        g = inputs["image_grid_thw"][0]
        m = backbone.model.config.vision_config.spatial_merge_size
        img_grid = (int(g[1]) // m, int(g[2]) // m)
        if blur_sigma_frac > 0:
            # floored so that a tiny image cannot resolve to sigma 0, an exact no-op
            sigma = max(0.05, blur_sigma_frac * min(img_grid))
    edit = AttentionIntervention(
        lo,
        hi,
        starts[0] + 1,
        ends[0],
        gamma,
        gamma2=gamma2,
        region_mask=region_mask,
        ablate=ablate,
        op=op,
        renorm=renorm,
        img_cap_ratio=img_cap_ratio,
        attn_cfg_alpha=alpha,
        blur_sigma=sigma,
        peak_lambda=peak_lambda,
        img_grid=img_grid,
    )

    @contextlib.contextmanager
    def installed():
        edit.install(model_layers)
        try:
            yield
        finally:
            edit.remove()

    return installed
