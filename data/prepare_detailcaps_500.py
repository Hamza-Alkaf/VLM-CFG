"""Build `detailcaps/detailcaps_500.parquet`:
a frozen 500-image subset of DetailCaps-4870.

500 of the 4,870 images, stratified across the seven image sources: largest-remainder
quotas that sum to exactly 500, a seed-0 sample within each source, sorted by
(source, image). Every arm must see the same documents in the same order, so the subset
is frozen on disk instead of resampled at load time.

    python data/prepare_detailcaps_500.py
"""

import pandas as pd
from _common import hub_file, output
from datasets import Dataset, Features, Image, Value

REPO = "foundation-multimodal-models/DetailCaps-4870"
REVISION = "b9f3aa0e154f7b3f5d7ccf83358258f01cbad39d"
OUT = "detailcaps/detailcaps_500.parquet"
N = 500
SEED = 0
FEATURES = Features(
    {
        "doc_id": Value("int32"),
        "source": Value("string"),
        "image_id": Value("string"),
        "image": Image(),
        "GT_Caption_GPT4O": Value("string"),
        "GT_Caption_GPT4V": Value("string"),
        "GT_Caption_Gemini15Pro": Value("string"),
    }
)


def main() -> None:
    df = pd.read_parquet(hub_file(REPO, "DetailCaps-4870.parquet", revision=REVISION))
    # Largest-remainder allocation over sources, so the quotas sum to exactly N and the
    # split does not depend on dict ordering.
    counts = df.source.value_counts().sort_index()
    exact = counts / counts.sum() * N
    quota = exact.astype(int)
    for s in (
        exact.sub(quota).sort_values(ascending=False).index[: N - int(quota.sum())]
    ):
        quota[s] += 1
    assert quota.sum() == N, quota

    parts = [
        g.sample(int(quota[s]), random_state=SEED)
        for s, g in df.groupby("source", sort=True)
    ]
    pick = pd.concat(parts).sort_values(["source", "image"]).reset_index(drop=True)
    assert len(pick) == N and pick.image.nunique() == N, (
        "subset is not N distinct images"
    )

    ds = Dataset.from_dict(
        {
            "doc_id": list(range(len(pick))),
            "source": pick.source.tolist(),
            "image_id": pick.image.tolist(),
            "image": [{"bytes": b, "path": None} for b in pick.binary],
            "GT_Caption_GPT4O": pick.GT_Caption_GPT4O.tolist(),
            "GT_Caption_GPT4V": pick.GT_Caption_GPT4V.tolist(),
            "GT_Caption_Gemini15Pro": pick.GT_Caption_Gemini15Pro.tolist(),
        },
        features=FEATURES,
    )
    ds.to_parquet(str(output(OUT)))
    print(f"wrote {len(ds)} rows -> {output(OUT)}")


if __name__ == "__main__":
    main()
