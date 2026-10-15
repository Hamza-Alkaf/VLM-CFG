"""Build `hallusionbench_visual/hallusionbench_visual_vs.parquet`: HallusionBench VS.

The Visual Supplement questions that come with an image (360), in file order; text-only
records are dropped, since guidance has no image to condition on. All figure_ids are
kept. Questions and images come from the authors' Hugging Face copy of the
HallusionBench repository (pinned), whose `HallusionBench.json` is the GitHub one.
Images with a side above 2000 px are downscaled to 2000 (LANCZOS).

    python data/prepare_hallusionbench_visual.py
"""

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import datasets
from _common import hub_snapshot, output
from PIL import Image

REPO = "rayguan/HallusionBench"
REVISION = "2067dfdce5abc7efdbb0927f4721ae282f0b6c51"
OUT = "hallusionbench_visual/hallusionbench_visual_vs.parquet"
MAX_IMAGE_DIM = 2000
EXPECTED_ROWS = 360

FEATURES = datasets.Features(
    {
        "question_id": datasets.Value("string"),
        "image": datasets.Image(),
        "question": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "category": datasets.Value("string"),
        "subcategory": datasets.Value("string"),
        "set_id": datasets.Value("string"),
        "figure_id": datasets.Value("string"),
        "visual_input": datasets.Value("string"),
        "gt_answer_details": datasets.Value("string"),
        "image_path": datasets.Value("string"),
    }
)


def image_payload(path: Path, name: str) -> dict[str, Any]:
    """The original bytes, re-encoded only if a side exceeds MAX_IMAGE_DIM."""
    with Image.open(path) as img:
        source_format = img.format or "PNG"
        oversized = max(img.size) > MAX_IMAGE_DIM
    if not oversized:
        return {"bytes": path.read_bytes(), "path": name}
    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = MAX_IMAGE_DIM / max(img.size)
        size = (int(round(img.width * scale)), int(round(img.height * scale)))
        resized = img.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format=source_format)
    return {"bytes": buffer.getvalue(), "path": name}


def relative_path(record: dict[str, Any]) -> str:
    """`./VS/chart/0_1.png` -> `VS/chart/0_1.png` (the JSON's spelling)."""
    return record["filename"].lstrip("./")


def generate_rows(
    records: list[dict[str, Any]], files: dict[str, str]
) -> Iterator[dict]:
    """One row per visual record."""
    for record in records:
        rel = relative_path(record)
        yield {
            "question_id": "_".join(
                [
                    record["category"],
                    record["subcategory"],
                    record["set_id"],
                    record["figure_id"],
                    record["question_id"],
                ]
            ),
            "image": image_payload(Path(files[rel.lower()]), Path(rel).name),
            "question": record["question"],
            "answer": "yes" if record["gt_answer"] == "1" else "no",
            "category": record["category"],
            "subcategory": record["subcategory"],
            "set_id": record["set_id"],
            "figure_id": record["figure_id"],
            "visual_input": record["visual_input"],
            "gt_answer_details": record["gt_answer_details"],
            "image_path": rel,
        }


def main() -> None:
    root = hub_snapshot(REPO, ["HallusionBench.json", "data/**"], revision=REVISION)
    data = json.loads((root / "HallusionBench.json").read_text(encoding="utf-8"))
    records = [
        r for r in data if str(r["visual_input"]) != "0" and r["category"] == "VS"
    ]
    assert len(records) == EXPECTED_ROWS, (
        f"expected {EXPECTED_ROWS}, got {len(records)}"
    )
    # The mirror spells some extensions in upper case
    # (10_1.PNG for the JSON's 10_1.png).
    files = {
        str(p.relative_to(root / "data")).lower(): str(p)
        for p in (root / "data").rglob("*")
        if p.is_file()
    }
    missing = [
        relative_path(r) for r in records if relative_path(r).lower() not in files
    ]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"

    dataset = datasets.Dataset.from_generator(
        generate_rows,
        features=FEATURES,
        gen_kwargs={"records": records, "files": files},
        cache_dir=str(root / ".hf_build_cache"),
    )
    assert len(set(dataset["question_id"])) == len(dataset), "duplicate question_id"
    dataset.to_parquet(str(output(OUT)))
    print(f"wrote {len(dataset)} rows -> {output(OUT)}")


if __name__ == "__main__":
    main()
