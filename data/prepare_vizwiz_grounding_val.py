"""Build `vizwiz_grounding_val/vizwiz_grounding_val.parquet`:
VizWiz answer grounding, val.

The 1,131 validation questions of VizWiz-VQA-Grounding (vizwiz.org), in the order of
`val_grounding.json`, with the original image and binary answer mask. `bbox` is the
mask's extent, [x0, y0, x1, y1] with exclusive x1/y1. The image archive is the full
VizWiz val set (3.5 GB); only the grounded images are kept.

    python data/prepare_vizwiz_grounding_val.py
"""

import io
import json
import zipfile

import datasets
import numpy as np
from _common import output, url_file
from PIL import Image

BASE = "https://vizwiz.cs.colorado.edu/VizWiz_final"
OUT = "vizwiz_grounding_val/vizwiz_grounding_val.parquet"
_STR = {"dtype": "string", "_type": "Value"}
_F64 = {"dtype": "float64", "_type": "Value"}
FEATURES = datasets.Features.from_dict(
    {
        "question_id": _STR,
        "file_name": _STR,
        "image": {"_type": "Image"},
        "mask": {"_type": "Image"},
        "question": _STR,
        "answers": {"feature": _STR, "_type": "List"},
        "most_common_answer": _STR,
        "answer_grounding": {"feature": {"x": _F64, "y": _F64}, "_type": "List"},
        "bbox": {"feature": _F64, "_type": "List"},
        "width": {"dtype": "int32", "_type": "Value"},
        "height": {"dtype": "int32", "_type": "Value"},
    }
)


def mask_bbox(png: bytes) -> list[float] | None:
    """[x0, y0, x1, y1] of the mask's foreground, x1/y1 exclusive."""
    ys, xs = np.nonzero(np.array(Image.open(io.BytesIO(png))) > 0)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def main() -> None:
    with zipfile.ZipFile(
        url_file(f"{BASE}/VizWiz_grounding/annotations.zip", "vizwiz_annotations.zip")
    ) as z:
        grounding = json.loads(z.read("val_grounding.json"))
    masks = zipfile.ZipFile(
        url_file(
            f"{BASE}/VizWiz_grounding/binary_masks_png.zip",
            "vizwiz_binary_masks_png.zip",
        )
    )
    images = zipfile.ZipFile(url_file(f"{BASE}/images/val.zip", "vizwiz_val.zip"))
    image_names = {n.rsplit("/", 1)[-1]: n for n in images.namelist()}
    rows = []
    for file_name, ann in grounding.items():
        qid = file_name.rsplit(".", 1)[0]
        mask = masks.read(f"val/{qid}.png")
        rows.append(
            {
                "question_id": qid,
                "file_name": file_name,
                "image": {
                    "bytes": images.read(image_names[file_name]),
                    "path": file_name,
                },
                "mask": {"bytes": mask, "path": f"{qid}.png"},
                "question": ann["question"],
                "answers": ann["answers"],
                "most_common_answer": ann["most_common_answer"],
                "answer_grounding": ann["answer_grounding"],
                "bbox": mask_bbox(mask),
                "width": int(ann["width"]),
                "height": int(ann["height"]),
            }
        )
    datasets.Dataset.from_list(rows, features=FEATURES).to_parquet(str(output(OUT)))
    print(f"wrote {len(rows)} rows -> {output(OUT)}")


if __name__ == "__main__":
    main()
