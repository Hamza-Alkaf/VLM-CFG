"""Build `refchartqa/refchartqa_val.parquet`:
the RefChartQA validation split, as published.

python data/prepare_refchartqa.py
"""

import shutil

from _common import hub_file, output

REPO = "omoured/RefChartQA"
OUT = "refchartqa/refchartqa_val.parquet"


def main() -> None:
    src = hub_file(REPO, "data/validation-00000-of-00001.parquet")
    shutil.copyfile(src, output(OUT))
    print(f"wrote {output(OUT)}")


if __name__ == "__main__":
    main()
