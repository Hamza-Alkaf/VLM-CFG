"""Build the frozen 2K stratified subset of OCRBench v2.

WHY A FROZEN SUBSET: the full test split is 10,000 items, and under CFG every decoded
token costs two forward passes, so a full alpha sweep is unaffordable. Sampling once
into a versioned artifact (rather than passing --limit) means every alpha in the sweep
sees the IDENTICAL items, which is what makes the paired alpha-vs-alpha comparison
tight; it also makes the subset releasable alongside the paper.

SAMPLING (seed 1234, fixed):
  * PROPORTIONAL, not equal-n, per `type`. OCRBench v2's per-type sizes are already
    deliberately balanced by its authors (most types are exactly 200/300/400), so an
    equal-n draw would silently RE-WEIGHT the benchmark and make the score
    incomparable to published numbers. A flat ~20% preserves their weighting.
  * Stratified within each type by EVAL PROTOCOL, so the mix of scoring paths
    (free-form / multiple choice / regression / exact match / case sensitive) survives
    inside each type rather than being left to chance.
  * Largest-remainder allocation, so the total lands on exactly 2000 rather than
    drifting with per-type rounding.

LENGTH CAP (MAX_GT_WORDS): items whose ground truth exceeds the cap are dropped BEFORE
sampling. At 500 words this removes 2.2% of the pool (9777/10000 survive) and
eliminates NO type -- only `full-page OCR en` (57.5% kept) and `document parsing en`
(70.5%) are affected materially. This bounds generation cost: the cap is what lets
max_new_tokens sit at a level that does not truncate a kept answer.

The manifest (ocrbench_v2_2k_manifest.json) records the seed, cap, and the id of every
selected item, so the exact subset is reconstructible.

    python data/prepare_ocrbench_v2_2k.py

Writes the `datasets` directory `ocrbench_v2_2k/` (and its manifest) under the data dir.
"""

import collections
import json
import os
import random

import datasets
from _common import output

SEED = 1234
TARGET = 2000
MAX_GT_WORDS = 500
SRC = "ling99/OCRBench_v2"


def _gt_words(ex):
    a = ex["answers"]
    if isinstance(a, list):
        a = a[0] if a else ""
    return len(str(a).split())


def main():
    out = str(output("ocrbench_v2_2k/dataset_dict.json").parent)
    ds = datasets.load_dataset(SRC, split="test")

    # Index rows by (type, eval) without materialising images -- the parquet decode of
    # 10k images is the slow part and none of it is needed to CHOOSE the subset.
    lens = ds.map(
        lambda ex: {"_w": _gt_words(ex)},
        remove_columns=[c for c in ds.column_names if c not in ("type", "eval")],
        desc="measuring gt length",
    )
    eligible = collections.defaultdict(list)
    n_dropped = 0
    for i, row in enumerate(lens):
        if row["_w"] > MAX_GT_WORDS:
            n_dropped += 1
            continue
        eligible[(row["type"], str(row["eval"]))].append(i)

    pool = sum(len(v) for v in eligible.values())
    print(
        f"eligible after {MAX_GT_WORDS}-word cap: {pool}/{len(ds)} "
        f"(dropped {n_dropped})"
    )

    # Largest-remainder allocation over strata so the total is exactly TARGET.
    exact = {k: len(v) * TARGET / pool for k, v in eligible.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    short = TARGET - sum(alloc.values())
    for k in sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True)[:short]:
        alloc[k] += 1

    rng = random.Random(SEED)
    picked = []
    for k in sorted(eligible):
        idxs = sorted(eligible[k])
        take = min(alloc[k], len(idxs))
        picked.extend(rng.sample(idxs, take))
    picked.sort()
    print(f"selected {len(picked)} items across {len(eligible)} (type, eval) strata")

    # INTERLEAVE BY TYPE before saving. OCRBench v2's aggregate averages 13 fixed
    # category slots and counts an EMPTY slot as 0, so a contiguous `--limit` prefix --
    # which in source order is a single type -- reports a meaningless near-zero score
    # (a limit=40 pilot hit only `APP agent en` and scored 0.06 despite 0.93 on the
    # items it actually ran). Round-robin ordering makes every prefix span all 30 types,
    # so small pilot runs stay interpretable. It does not change the 2000-item SET.
    by_type = collections.defaultdict(list)
    for pos, src_idx in enumerate(picked):
        by_type[lens[src_idx]["type"]].append(pos)
    order, keys = [], sorted(by_type)
    while any(by_type[k] for k in keys):
        for k in keys:
            if by_type[k]:
                order.append(by_type[k].pop(0))

    sub = ds.select(picked).select(order)
    # Wrap in a DatasetDict: lmms-eval indexes `self.dataset[test_split]`, so a bare
    # Dataset saved to disk would fail to resolve a split name.
    datasets.DatasetDict({"test": sub}).save_to_disk(out)

    manifest = {
        "source": SRC,
        "seed": SEED,
        "target": TARGET,
        "max_gt_words": MAX_GT_WORDS,
        "n_selected": len(picked),
        "source_indices": picked,
        "interleaved_order": order,
        "per_type": dict(collections.Counter(sub["type"])),
        "per_eval": dict(collections.Counter(str(e) for e in sub["eval"])),
    }
    mpath = os.path.join(os.path.dirname(out), "ocrbench_v2_2k_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"wrote {out}\nmanifest {mpath}")


if __name__ == "__main__":
    main()
