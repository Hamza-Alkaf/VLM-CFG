"""Build `reasonseg_gseval_grounding/reasonseg_grounding_{val,test}.parquet`.

ReasonSeg (LISA) recast as box grounding: the mask stays on each row (COCO RLE) and the
box is derived from it. The conventions are Sa2VA's/LISA's, so numbers stay comparable:
  * `text[0].strip()` is the query; rows with no text, no target or an empty mask are
    dropped;
  * shapes labelled "target" form the mask, "ignore" a separate (unscored) one, "flag"
    none; polygons with fewer than 3 points are skipped;
  * polygons are filled in the image's raw decode frame (no EXIF transpose, as the
    reference);
  * the box spans the set pixels, maxima + 1 (pixel i covers [i, i+1));
  * `is_sentence` gives the short/long group.

    python data/prepare_reasonseg_grounding.py
"""

import json
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from _common import hub_file, output, sources
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

REPO = "Myszz/ReasonSeg"  # the ReasonSeg release archives
OUT = "reasonseg_gseval_grounding/reasonseg_grounding_{split}.parquet"

SCHEMA = pa.schema(
    [
        ("question_id", pa.string()),
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("sentence", pa.string()),
        ("bbox", pa.list_(pa.float64())),
        ("bbox_norm", pa.list_(pa.float64())),
        ("segmentation", pa.string()),
        ("ignore_bbox", pa.list_(pa.float64())),
        ("ignore_segmentation", pa.string()),
        ("image_width", pa.int64()),
        ("image_height", pa.int64()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("group", pa.string()),
    ],
    metadata={
        "huggingface": json.dumps({"info": {"features": {"image": {"_type": "Image"}}}})
    },
)


def polygons_to_mask(shapes: list[dict], width: int, height: int) -> np.ndarray:
    """Rasterise LabelMe polygons, merged by maximum (the LISA reference)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        layer = Image.new("L", (width, height), 0)
        ImageDraw.Draw(layer).polygon([c for xy in points for c in xy], fill=1)
        mask = np.maximum(mask, np.array(layer, dtype=np.uint8))
    return mask


def mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    """Tight [x0, y0, x1, y1] around the set pixels, or None when the mask is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def encode_rle(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def normalize(box: list[float], width: int, height: int) -> list[float]:
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def split_dir(split: str) -> Path:
    """The extracted `<split>/` directory of the release archive."""
    root = sources() / "reasonseg"
    if not (root / split).is_dir():
        with zipfile.ZipFile(hub_file(REPO, f"{split}.zip")) as z:
            z.extractall(root)
    return root / split


def load_reasonseg(split: str) -> Iterator[dict[str, Any]]:
    """One row per ReasonSeg image."""
    for json_path in sorted(split_dir(split).glob("*.json")):
        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            continue
        doc = json.loads(json_path.read_text())
        texts = doc.get("text") or []
        sentence = texts[0].strip() if texts else ""
        if not sentence:
            continue
        shapes = doc.get("shapes", [])
        target = [s for s in shapes if s.get("label") == "target"]
        ignore = [s for s in shapes if s.get("label") == "ignore"]
        if not target:
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        mask = polygons_to_mask(target, width, height)
        box = mask_to_bbox(mask)
        if box is None:
            continue
        ignore_box, ignore_rle = None, None
        if ignore:
            ignore_mask = polygons_to_mask(ignore, width, height)
            if ignore_mask.any():
                ignore_box = mask_to_bbox(ignore_mask)
                ignore_rle = encode_rle(ignore_mask)
        yield {
            "question_id": f"reasonseg_{split}_{json_path.stem}",
            "image": {
                "bytes": image_path.read_bytes(),
                "path": os.path.basename(image_path),
            },
            "sentence": sentence,
            "bbox": box,
            "bbox_norm": normalize(box, width, height),
            "segmentation": encode_rle(mask),
            "ignore_bbox": ignore_box,
            "ignore_segmentation": ignore_rle,
            "image_width": width,
            "image_height": height,
            "dataset": "reasonseg",
            "split": split,
            "group": "long" if bool(doc.get("is_sentence", False)) else "short",
        }


def main() -> None:
    for split in ("val", "test"):
        rows = list(load_reasonseg(split))
        out = output(OUT.format(split=split))
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out)
        print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
