"""Scoring for `reasonseg_grounding` and `gseval_grounding`.

Both are segmentation benchmarks recast as box grounding (see
`tools/build_reasonseg_gseval_grounding.py` for how the shards are built and which Sa2VA
conventions they follow). Scoring is the same IoU family the refcoco grounding tasks
use, so a model's numbers are comparable across all of them.

Two deliberate choices:

* ReasonSeg's `ignore` region is carried on the doc but NOT scored. Sa2VA excludes
  ignore pixels from both terms of cIoU/gIoU, which is well defined for masks; there is
  no agreed box-level equivalent, and inventing one would make these numbers
  incomparable to everyone else's. The column is there for a later experiment.
* ReasonSeg reports per-group breakdowns (short vs. long query), because the long
  "reasoning" queries are the point of the benchmark and averaging them with the short
  phrases hides the effect. GSEval groups by its four class ids.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlmcfg.frames import get_visual_input_frame

eval_logger = logging.getLogger("lmms-eval")

GROUNDING_PROMPT = (
    "Bounding box coordinates are specified in the format (top-left x, top-left y, "
    "bottom-right x, bottom-right y). All values are floating point numbers bounded "
    "between 0 and 1. Please provide the bounding box coordinate of the region this "
    "sentence describes: "
)

REC_METRICS = ("IoU", "IoU@0.5", "IoU@0.25", "IoU@0.75", "IoU@0.9", "Center_ACC")


def doc_to_visual(doc: dict[str, Any]) -> list:
    return [doc["image"].convert("RGB")]


def doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{GROUNDING_PROMPT}{doc['sentence']}{post_prompt}"


def doc_to_target(doc: dict[str, Any]) -> list[float]:
    return doc["bbox_norm"]


_NUM = r"-?\d+(?:\.\d+)?"
# Ordered by specificity: a keyed box wins over a bare bracket, so the `label` string in
# Qwen's JSON or a stray list earlier in the response cannot hijack the match.
_BOX_PATTERNS = (
    rf'"(?:bbox_2d|bbox|box)"\s*:\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]',
    rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]",
    rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*,\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)",
    # A single parenthesised 4-tuple, as prose answers write it: "the coordinates for
    # the brim are approximately (130, 220, 532, 364)". The model localises correctly on
    # these and only the wrapper differs, so without this the box is dropped and the doc
    # scores IoU 0 -- which reads as a localisation failure rather than a formatting
    # one. Last in the order, so it only fires when no keyed, bracketed or point-pair
    # form matched.
    rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\)",
)


def parse_bbox(text: str) -> list[float]:
    """First `[x0, y0, x1, y1]` box in the response;
    the zero sentinel when none parses."""
    for pattern in _BOX_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
    return [0.0, 0.0, 0.0, 0.0]


def normalize_bbox_to_unit(box: list[float], doc: dict[str, Any]) -> list[float]:
    """Rescale a pixel-space prediction into the [0, 1] frame the targets live in.

    Qwen-VL answers in the pixel frame of the processor-resized image regardless of the
    prompt asking for [0, 1]; `smart_resize` rounds each side to a multiple of 28, so
    dividing by the original size mis-scales x and y by different factors.
    """
    if all(0.0 <= v <= 1.0 for v in box):
        return list(box)

    source_size = (doc.get("image_width"), doc.get("image_height"))
    if not all(source_size):
        eval_logger.warning(
            "Cannot normalise a pixel-space prediction without "
            "`image_width`/`image_height`; scoring it as-is."
        )
        return list(box)

    width, height = get_visual_input_frame(source_size) or source_size
    scales = (width, height, width, height)
    return [min(max(v / s, 0.0), 1.0) for v, s in zip(box, scales)]


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def compute_center_accuracy(gt_box: list[float], pred_box: list[float]) -> bool:
    center_x = (pred_box[0] + pred_box[2]) / 2
    center_y = (pred_box[1] + pred_box[3]) / 2
    return gt_box[0] <= center_x <= gt_box[2] and gt_box[1] <= center_y <= gt_box[3]


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict]:
    pred = normalize_bbox_to_unit(parse_bbox(results[0] if results else ""), doc)
    data = {
        "question_id": doc["question_id"],
        "sentence": doc["sentence"],
        "pred": pred,
        "bbox_norm": doc["bbox_norm"],
        "group": doc.get("group") or "all",
    }
    return {metric: data for metric in REC_METRICS}


_SCORERS = {
    "IoU": compute_iou,
    "IoU@0.25": lambda gt, pred: float(compute_iou(gt, pred) >= 0.25),
    "IoU@0.5": lambda gt, pred: float(compute_iou(gt, pred) >= 0.5),
    "IoU@0.75": lambda gt, pred: float(compute_iou(gt, pred) >= 0.75),
    "IoU@0.9": lambda gt, pred: float(compute_iou(gt, pred) >= 0.9),
    "Center_ACC": lambda gt, pred: float(compute_center_accuracy(gt, pred)),
}


# Where the per-group breakdown is written. lmms-eval's metric API returns ONE float, so
# the groups (ReasonSeg short/long, GSEval's four class ids) have nowhere to go in the
# results JSON -- and `eval_logger.info` does not survive to the job logs under the
# launcher these tasks run through, so logging them loses them silently. A file next to
# the run is the only place they reliably land. Override with REASONSEG_GROUP_REPORT.
GROUP_REPORT_PATH = os.environ.get("REASONSEG_GROUP_REPORT", "")


def _write_group_report(metric: str, by_group: dict[str, list[float]]) -> None:
    """Append one metric's per-group means to the report file, if one is configured."""
    if not GROUP_REPORT_PATH:
        return
    line = json.dumps(
        {
            "metric": metric,
            "groups": {
                g: {"mean": sum(v) / len(v), "n": len(v)}
                for g, v in sorted(by_group.items())
            },
        }
    )
    try:
        path = Path(GROUP_REPORT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        eval_logger.warning(
            f"Could not write the group report to {GROUP_REPORT_PATH!r}: {exc}"
        )


def _aggregate(results: list[dict], metric: str) -> float:
    """Overall mean, with the per-group breakdown written to `GROUP_REPORT_PATH`.

    The groups are the interesting axis on these benchmarks -- ReasonSeg's long
    "reasoning" queries are the whole point of it, and averaging them with the short
    phrases hides the effect -- but a metric aggregation can only return one number, so
    they go to a file.
    """
    if not results:
        return 0.0
    scorer = _SCORERS[metric]
    scores = [scorer(r["bbox_norm"], r["pred"]) for r in results]

    by_group: dict[str, list[float]] = defaultdict(list)
    for r, s in zip(results, scores):
        by_group[r["group"]].append(s)
    if len(by_group) > 1:
        parts = ", ".join(
            f"{g}={sum(v) / len(v):.4f} (n={len(v)})"
            for g, v in sorted(by_group.items())
        )
        eval_logger.info(f"{metric} by group: {parts}")
        print(f"[{metric}] by group: {parts}", flush=True)
        _write_group_report(metric, by_group)

    return sum(scores) / len(scores)


def iou(results: list[dict]) -> float:
    return _aggregate(results, "IoU")


def iou_25(results: list[dict]) -> float:
    return _aggregate(results, "IoU@0.25")


def iou_50(results: list[dict]) -> float:
    return _aggregate(results, "IoU@0.5")


def iou_75(results: list[dict]) -> float:
    return _aggregate(results, "IoU@0.75")


def iou_90(results: list[dict]) -> float:
    return _aggregate(results, "IoU@0.9")


def center_acc(results: list[dict]) -> float:
    return _aggregate(results, "Center_ACC")
