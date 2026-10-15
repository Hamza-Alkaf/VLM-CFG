"""Build `treebench/treebench.parquet` from the published TreeBench TSV (405 questions).

The TSV carries each image base64-encoded; it is decoded to its original bytes and named
`<index>.jpg`. Columns are renamed to identifiers (`multi-choice options` ->
`multi_choice_options`), `target_instances` is parsed into boxes, and empty options are
null.

    python data/prepare_treebench.py
"""

import ast
import base64

import datasets
import pandas as pd
from _common import hub_file, output

REPO = "HaochenWang/TreeBench"
OUT = "treebench/treebench.parquet"
OPTIONS = list("ABCDEFGHIJK")
_STR = {"dtype": "string", "_type": "Value"}
FEATURES = datasets.Features.from_dict(
    {
        "question_id": {"dtype": "int64", "_type": "Value"},
        "image": {"_type": "Image"},
        "question": _STR,
        "answer": _STR,
        "multi_choice_options": _STR,
        "category": _STR,
        "l2_category": _STR,
        "target_instances": {
            "feature": {
                "feature": {"dtype": "float64", "_type": "Value"},
                "_type": "List",
            },
            "_type": "List",
        },
        **{k: _STR for k in OPTIONS},
    }
)


def _text(value):
    return None if pd.isna(value) else str(value)


def main() -> None:
    df = pd.read_csv(hub_file(REPO, "TreeBench.tsv"), sep="\t")
    rows = [
        {
            "question_id": int(r["index"]),
            "image": {
                "bytes": base64.b64decode(r["image"]),
                "path": f"{r['index']}.jpg",
            },
            "question": r["question"],
            "answer": r["answer"],
            "multi_choice_options": r["multi-choice options"],
            "category": r["category"],
            "l2_category": r["l2-category"],
            "target_instances": [
                [float(v) for v in box]
                for box in ast.literal_eval(r["target_instances"])
            ],
            **{k: _text(r[k]) for k in OPTIONS},
        }
        for _, r in df.iterrows()
    ]
    datasets.Dataset.from_list(rows, features=FEATURES).to_parquet(str(output(OUT)))
    print(f"wrote {len(rows)} rows -> {output(OUT)}")


if __name__ == "__main__":
    main()
