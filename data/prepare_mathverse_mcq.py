"""Build `mathverse/mathverse_mcq_<version>.parquet`:
MathVerse testmini, multiple choice.

The multiple-choice problems of MathVerse testmini (AI4Math/MathVerse) whose answer is
one clean letter in every version, so that the six versions hold the same problems
(426 each), sorted by problem index.

    python data/prepare_mathverse_mcq.py
"""

import pandas as pd
from _common import hub_file, output
from datasets import Dataset, Features, Image, Value

REPO = "AI4Math/MathVerse"
VERSIONS = [
    "Text Dominant",
    "Text Lite",
    "Text Only",
    "Vision Intensive",
    "Vision Dominant",
    "Vision Only",
]
FEATURES = Features(
    {
        "doc_id": Value("int32"),
        "problem_index": Value("string"),
        "sample_index": Value("string"),
        "problem_version": Value("string"),
        "question": Value("string"),
        "image": Image(),
        "answer": Value("string"),
        "question_type": Value("string"),
        "query_wo": Value("string"),
        "query_cot": Value("string"),
        "question_for_eval": Value("string"),
        "subject": Value("string"),
        "subfield": Value("string"),
    }
)


def main() -> None:
    img_df = pd.read_parquet(hub_file(REPO, "testmini.parquet"))
    txt_df = pd.read_parquet(hub_file(REPO, "testmini_text_only.parquet"))
    allv = pd.concat([img_df, txt_df], ignore_index=True)
    mcq = allv[allv.question_type == "multi-choice"]
    clean = mcq.answer.str.fullmatch(r"[A-Za-z]")
    dirty = set(mcq.loc[~clean, "problem_index"])
    keep = sorted(set(mcq.problem_index) - dirty, key=int)
    sub = mcq[mcq.problem_index.isin(keep)]
    per_version = sub.groupby("problem_version").size().to_dict()
    assert len(set(per_version.values())) == 1, (
        f"versions are not balanced: {per_version}"
    )
    for ver in VERSIONS:
        v = (
            sub[sub.problem_version == ver]
            .sort_values("problem_index", key=lambda s: s.astype(int))
            .reset_index(drop=True)
        )
        assert len(v) == len(keep), f"{ver}: {len(v)} rows for {len(keep)} problems"
        images = [
            {"bytes": cell["bytes"], "path": None}
            if isinstance(cell, dict) and cell.get("bytes")
            else None
            for cell in v["image"]
        ]
        md = pd.json_normalize(v.metadata)
        ds = Dataset.from_dict(
            {
                "doc_id": list(range(len(v))),
                "problem_index": v.problem_index.tolist(),
                "sample_index": v.sample_index.tolist(),
                "problem_version": v.problem_version.tolist(),
                "question": v.question.tolist(),
                "image": images,
                "answer": v.answer.str.upper().tolist(),
                "question_type": v.question_type.tolist(),
                "query_wo": v.query_wo.tolist(),
                "query_cot": v.query_cot.tolist(),
                "question_for_eval": v.question_for_eval.tolist(),
                "subject": md.get("subject", pd.Series([""] * len(v))).tolist(),
                "subfield": md.get("subfield", pd.Series([""] * len(v))).tolist(),
            },
            features=FEATURES,
        )
        out = output(f"mathverse/mathverse_mcq_{ver.lower().replace(' ', '_')}.parquet")
        ds.to_parquet(str(out))
        print(f"wrote {len(ds)} rows -> {out}")


if __name__ == "__main__":
    main()
