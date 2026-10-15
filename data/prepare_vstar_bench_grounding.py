"""Build `vstar_bench_grounding/vstar_bench_grounding.parquet`: V* Bench with its boxes.

The 191 questions of lmms-lab/vstar-bench (image, prompt, answer letter), with the
question and options split out of the prompt, joined by image name to the original V*
annotations (craigwu/vstar_bench), which give the target objects and their boxes
([x, y, w, h] there, [x0, y0, x1, y1] here).

    python data/prepare_vstar_bench_grounding.py
"""

import json
import re

import datasets
import pyarrow.parquet as pq
from _common import hub_file, hub_snapshot, output

QUESTIONS = "lmms-lab/vstar-bench"
ANNOTATIONS = "craigwu/vstar_bench"
OUT = "vstar_bench_grounding/vstar_bench_grounding.parquet"
_STR = {"dtype": "string", "_type": "Value"}
FEATURES = datasets.Features.from_dict(
    {
        "question_id": _STR,
        "image": {"_type": "Image"},
        "question": _STR,
        "answer": _STR,
        "category": _STR,
        "target_objects": {"feature": _STR, "_type": "List"},
        "bboxes": {
            "feature": {
                "feature": {"dtype": "float64", "_type": "Value"},
                "_type": "List",
            },
            "_type": "List",
        },
        **{k: _STR for k in "ABCD"},
    }
)
_OPTION = re.compile(r"^\(([A-D])\) (.*)$")


def split_prompt(text: str) -> tuple[str, dict[str, str]]:
    """The question (first line) and the lettered options of a V* prompt."""
    lines = text.split("\n")
    options = {m[1]: m[2] for m in map(_OPTION.match, lines[1:]) if m}
    return lines[0], options


def main() -> None:
    questions = pq.read_table(hub_file(QUESTIONS, "data/test-00000-of-00001.parquet"))
    root = hub_snapshot(
        ANNOTATIONS,
        allow_patterns=["direct_attributes/*.json", "relative_position/*.json"],
    )
    rows = []
    for r in questions.to_pylist():
        name = r["image"]["path"].rsplit(".", 1)[0]
        ann = json.loads((root / r["category"] / f"{name}.json").read_text())
        question, options = split_prompt(r["text"])
        rows.append(
            {
                "question_id": r["question_id"],
                "image": {"bytes": r["image"]["bytes"], "path": None},
                "question": question,
                "answer": r["label"],
                "category": r["category"],
                "target_objects": ann["target_object"],
                "bboxes": [
                    [float(x), float(y), float(x + w), float(y + h)]
                    for x, y, w, h in ann["bbox"]
                ],
                **{k: options.get(k) for k in "ABCD"},
            }
        )
    datasets.Dataset.from_list(rows, features=FEATURES).to_parquet(str(output(OUT)))
    print(f"wrote {len(rows)} rows -> {output(OUT)}")


if __name__ == "__main__":
    main()
