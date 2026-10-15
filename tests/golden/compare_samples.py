"""Compare two lmms-eval samples files response by response.

    python tests/golden/compare_samples.py NEW.jsonl REFERENCE.jsonl
        [MORE_SHARDS.jsonl ...]

Documents are matched by `doc_id`; only documents present in NEW are checked, so a
`--limit` run can be compared against a full reference run. Exits 1 on any difference.
"""

import argparse
import json
import sys
from pathlib import Path


def load_responses(path: Path) -> dict[int, list]:
    """doc_id -> model responses, from an lmms-eval `*_samples_*.jsonl` file."""
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    return {row["doc_id"]: row.get("filtered_resps", row.get("resps")) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", type=Path)
    parser.add_argument(
        "reference", type=Path, nargs="+", help="one file, or all shards"
    )
    args = parser.parse_args()

    new = load_responses(args.new)
    ref = {k: v for path in args.reference for k, v in load_responses(path).items()}
    missing = sorted(set(new) - set(ref))
    differing = sorted(d for d in new if d in ref and new[d] != ref[d])
    print(
        f"{len(new)} docs checked, {len(differing)} differ, "
        f"{len(missing)} not in reference"
    )
    for doc_id in differing[:5]:
        print(f"  doc {doc_id}\n    new: {new[doc_id]!r}\n    ref: {ref[doc_id]!r}")
    return 1 if differing or missing else 0


if __name__ == "__main__":
    sys.exit(main())
