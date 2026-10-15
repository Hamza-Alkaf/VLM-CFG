"""OCRBench v2 -- frozen 2K stratified subset.

Scoring is DELEGATED to the parent `ocrbench_v2` task: this module re-exports its
process_results and aggregation functions unchanged, so the subset is scored by exactly
the same 30-way metric dispatch (TEDS for tables/charts, IoU for grounding/spotting,
BLEU/METEOR/edit-distance for page OCR, ANLS/F1 elsewhere) as the full benchmark. Only
the ITEM SET differs; nothing about the metric does.

See prepare_ocrbench_v2_2k.py for how the subset is drawn and frozen.
"""

from vlmcfg.tasks.ocrbench_v2.utils import (
    ocrbench_v2_aggregate_accuracy,
    ocrbench_v2_doc_to_text,
    ocrbench_v2_doc_to_visual,
    ocrbench_v2_process_results,
)

__all__ = [
    "ocrbench_v2_aggregate_accuracy",
    "ocrbench_v2_doc_to_text",
    "ocrbench_v2_doc_to_visual",
    "ocrbench_v2_process_results",
]
