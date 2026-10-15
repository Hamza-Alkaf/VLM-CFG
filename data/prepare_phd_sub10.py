"""Build `phd_sub10/phd_sub10.parquet`, the PhD hallucination benchmark subset.

PhD: the seed-42 10% subsample of the base/iac/icc records and every
counter-common-sense (ccs) record, two rows per record (the yes and the no question).
The records are the `data.json` of the PhD GitHub repository (pinned); the images come
from its Hugging Face mirror (COCO 2014 and the CCS images), one file at a time.

    python data/prepare_phd_sub10.py
"""

import io
import json
import random
from collections.abc import Iterator
from typing import Any

import datasets
from _common import hub_snapshot, output, url_file
from PIL import Image

REPO = "AIMClab-RUC/PhD"
# The Hub's data.json is a later revision (other contexts); the paper used this one.
DATA_JSON = (
    "https://raw.githubusercontent.com/jiazhen-code/PhD/"
    "c2c59d5087ca1b146b53b93a19280a3276b6f9f0/data.json"
)
OUT = "phd_sub10/phd_sub10.parquet"

# leading and trailing spaces are load-bearing: context and question are
# concatenated raw
INSTRUCTION = (
    " In case there is an inconsistency between the context and the image "
    "content, you should follow the image. "
)
CONTEXT_MODES = ("base", "iac", "icc")
ALL_MODES = ("base", "iac", "icc", "ccs")
SUBSAMPLE_SEED = 42
SUBSAMPLE_FRACTION = 0.1
MAX_IMAGE_DIM = 2000
EXPECTED_ROWS = {"base": 3368, "iac": 3368, "icc": 3368, "ccs": 1506}

FEATURES = datasets.Features(
    {
        "question_id": datasets.Value("string"),
        "image": datasets.Image(),
        "question": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "mode": datasets.Value("string"),
        "task": datasets.Value("string"),
        "image_id": datasets.Value("string"),
        "image_path": datasets.Value("string"),
    }
)


def coco_candidates(image_id: str) -> list[str]:
    """Repository paths of a COCO image: train2014 first, then val2014."""
    name = image_id.zfill(12)
    return [
        f"images/train2014/COCO_train2014_{name}.jpg",
        f"images/val2014/COCO_val2014_{name}.jpg",
    ]


def build_records(data: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """One mode's records; `image_path` lists the candidate repository paths."""
    if mode == "ccs":
        return [
            {
                "image_id": s["image_id"],
                "yes_question": s["yes_question"],
                "no_question": s["no_question"],
                "task": s["task"],
                "image_path": [f"images/CCS_images/{s['image_id']}.png"],
            }
            for s in data
            if "ccs_description" in s
        ]
    records = []
    for s in data:
        if "ccs_description" in s:
            continue
        if mode == "base":
            yes_q, no_q = s["yes_question"], s["no_question"]
        else:
            context = s["context"][mode]
            yes_q = context + INSTRUCTION + s["yes_question"]
            no_q = context + INSTRUCTION + s["no_question"]
        records.append(
            {
                "image_id": s["image_id"],
                "yes_question": yes_q,
                "no_question": no_q,
                "task": s["task"],
                "image_path": coco_candidates(s["image_id"]),
            }
        )
    return records


def subsample(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """The fixed seed-42 10% subset, in `random.sample` order (it numbers the ids)."""
    if mode == "ccs":
        return records
    random.seed(SUBSAMPLE_SEED)
    return random.sample(records, int(SUBSAMPLE_FRACTION * len(records)))


def image_payload(path) -> dict[str, Any]:
    """The original bytes, re-encoded only if a side exceeds MAX_IMAGE_DIM."""
    with Image.open(path) as img:
        source_format = img.format or "PNG"
        oversized = max(img.size) > MAX_IMAGE_DIM
    if not oversized:
        return {"bytes": path.read_bytes(), "path": path.name}
    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = MAX_IMAGE_DIM / max(img.size)
        size = (int(round(img.width * scale)), int(round(img.height * scale)))
        resized = img.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format=source_format)
    return {"bytes": buffer.getvalue(), "path": path.name}


def generate_rows(
    subsets: dict[str, list[dict[str, Any]]], root: str
) -> Iterator[dict]:
    """Two rows per record, the yes question then the no question."""
    from pathlib import Path

    for mode in ALL_MODES:
        for idx, record in enumerate(subsets[mode]):
            rel = next(p for p in record["image_path"] if (Path(root) / p).exists())
            payload = image_payload(Path(root) / rel)
            for polarity, question in (
                ("Yes", record["yes_question"]),
                ("No", record["no_question"]),
            ):
                yield {
                    "question_id": f"{mode}_{record['image_id']}_{polarity}_{idx}",
                    "image": payload,
                    "question": question,
                    "answer": polarity.lower(),
                    "mode": mode,
                    "task": record["task"],
                    "image_id": record["image_id"],
                    "image_path": rel,
                }


def main() -> None:
    data = json.loads(url_file(DATA_JSON, "phd_data.json").read_text(encoding="utf-8"))
    subsets = {mode: subsample(build_records(data, mode), mode) for mode in ALL_MODES}
    orders = [[r["image_id"] for r in subsets[m]] for m in CONTEXT_MODES]
    assert orders[0] == orders[1] == orders[2], (
        "context modes selected different records"
    )
    wanted = sorted({p for m in ALL_MODES for r in subsets[m] for p in r["image_path"]})
    root = hub_snapshot(REPO, allow_patterns=wanted)
    missing = [
        r["image_id"]
        for m in ALL_MODES
        for r in subsets[m]
        if not any((root / p).exists() for p in r["image_path"])
    ]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    dataset = datasets.Dataset.from_generator(
        generate_rows,
        features=FEATURES,
        gen_kwargs={"subsets": subsets, "root": str(root)},
        cache_dir=str(root / ".hf_build_cache"),
    )
    counts = {m: sum(1 for x in dataset["mode"] if x == m) for m in ALL_MODES}
    assert counts == EXPECTED_ROWS, f"expected {EXPECTED_ROWS}, got {counts}"
    dataset.to_parquet(str(output(OUT)))
    print(f"wrote {len(dataset)} rows -> {output(OUT)}")


if __name__ == "__main__":
    main()
