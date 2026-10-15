"""Build `amber/amber_generative.parquet`, AMBER's generative split (1,004 images).

The queries and annotations come from the AMBER GitHub repository (pinned); the images
from its Hugging Face mirror's AMBER.zip. The scorer's word lists (`relation.json`,
`safe_words.txt`) are written next to the parquet, where the task reads them.

    python data/prepare_amber.py
"""

import json
import shutil
import zipfile

from _common import hub_file, output, url_file
from datasets import Dataset, Features, Image, Sequence, Value

GITHUB = (
    "https://raw.githubusercontent.com/junyangwang0410/AMBER/"
    "534babf6bbfcce2e735c26289dedfb21cef3c939/data/"
)
IMAGES_REPO = "stzhao/amber"
IMAGES_REVISION = "f75c8afc54bdfaa2ecad4f1c04a23801113ec9e3"
OUT = "amber/amber_generative.parquet"
SCORER_FILES = ("relation.json", "safe_words.txt")
FEATURES = Features(
    {
        "doc_id": Value("int32"),
        "amber_id": Value("int32"),
        "image_file": Value("string"),
        "image": Image(),
        "query": Value("string"),
        "truth": Sequence(Value("string")),
        "hallu": Sequence(Value("string")),
    }
)


def fetch(name: str):
    """One file of the pinned AMBER `data/` directory."""
    return url_file(GITHUB + name, f"amber/{name}")


def main() -> None:
    query = json.loads(fetch("query/query_generative.json").read_text())
    annotations = json.loads(fetch("annotations.json").read_text())
    ann = {a["id"]: a for a in annotations if a["type"] == "generative"}

    rows = {k: [] for k in FEATURES}
    with zipfile.ZipFile(
        hub_file(IMAGES_REPO, "AMBER.zip", revision=IMAGES_REVISION)
    ) as z:
        for i, q in enumerate(query):
            a = ann.get(q["id"])
            if a is None:
                print(f"[warn] id {q['id']} has no generative annotation -- skipped")
                continue
            rows["doc_id"].append(i)
            rows["amber_id"].append(q["id"])
            rows["image_file"].append(q["image"])
            rows["image"].append({"bytes": z.read(f"image/{q['image']}"), "path": None})
            rows["query"].append(q["query"])
            rows["truth"].append(a["truth"])
            rows["hallu"].append(a["hallu"])

    ds = Dataset.from_dict(rows, features=FEATURES)
    ds.to_parquet(str(output(OUT)))
    for name in SCORER_FILES:
        shutil.copyfile(fetch(name), output(f"amber/{name}"))
    print(f"wrote {len(ds)} rows -> {output(OUT)}")


if __name__ == "__main__":
    main()
