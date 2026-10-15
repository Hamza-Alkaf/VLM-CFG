"""Build `openimages/openimages.parquet`:
958 grounded Open Images questions from Visual-CoT.

The Visual-CoT training records (deepcs233/Visual-CoT,
metadata/openimages_cot_train.jsonl) whose image id falls in a fixed range -- the slice
the paper's runs used: 662 images, 958 questions, in file order. Images are the Open
Images originals (CVDF mirror, at most 1024 px), fetched one at a time.

    python data/prepare_openimages.py
"""

import json
from concurrent.futures import ThreadPoolExecutor

import datasets
from _common import hub_file, output, url_file

REPO = "deepcs233/Visual-CoT"
IMAGES = "https://s3.amazonaws.com/open-images-dataset/train"
FIRST, LAST = "0b238719806105a6.jpg", "0d502f5e83f4ab41.jpg"
OUT = "openimages/openimages.parquet"
_STR = {"dtype": "string", "_type": "Value"}
FEATURES = datasets.Features.from_dict(
    {
        "question_id": _STR,
        "image": {"_type": "Image"},
        "image_relpath": _STR,
        "question": _STR,
        "answer": _STR,
        "answers": {"feature": _STR, "_type": "List"},
        "bboxs": {
            "feature": {
                "feature": {"dtype": "float64", "_type": "Value"},
                "_type": "List",
            },
            "_type": "List",
        },
        "width": {"dtype": "int32", "_type": "Value"},
        "height": {"dtype": "int32", "_type": "Value"},
        "split": _STR,
    }
)


def main() -> None:
    lines = (
        hub_file(REPO, "metadata/openimages_cot_train.jsonl").read_text().splitlines()
    )
    records = [r for r in map(json.loads, lines) if FIRST <= r["image"] <= LAST]
    names = sorted({r["image"] for r in records})
    with ThreadPoolExecutor(8) as pool:
        paths = dict(
            zip(
                names,
                pool.map(lambda n: url_file(f"{IMAGES}/{n}", f"openimages/{n}"), names),
            )
        )
    rows = [
        {
            "question_id": f"openimages_{i:05d}",
            "image": {"bytes": paths[r["image"]].read_bytes(), "path": None},
            "image_relpath": r["image"],
            "question": r["question"],
            "answer": r["answer"],
            "answers": [r["answer"]],
            "bboxs": [[float(v) for v in box] for box in r["bboxs"]],
            "width": int(r["width"]),
            "height": int(r["height"]),
            "split": r["split"],
        }
        for i, r in enumerate(records)
    ]
    datasets.Dataset.from_list(rows, features=FEATURES).to_parquet(str(output(OUT)))
    print(f"wrote {len(rows)} rows ({len(names)} images) -> {output(OUT)}")


if __name__ == "__main__":
    main()
