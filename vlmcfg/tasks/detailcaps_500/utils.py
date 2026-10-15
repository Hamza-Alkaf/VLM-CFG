"""DetailCaps-500: CAPTURE over a frozen 500-image subset of DetailCaps-4870.

CAPTURE is a corpus-level scorer, so per-document scores only exist at aggregation time
and never reach the samples jsonl. The aggregation therefore writes a SIDECAR file,
`<task>_per_doc_scores_CAPTURE.json`, mapping doc_id -> score, for paired per-item
comparisons.
"""

import json
import logging

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

eval_logger = logging.getLogger("lmms-eval")

TASK = "detailcaps_500"


def detailcaps_500_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def detailcaps_500_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return (lmms_eval_specific_kwargs or {}).get(
        "prompt", "Describe this image in detail."
    )


def detailcaps_500_doc_to_target(doc):
    """Three independently written dense references per image."""
    return [
        doc["GT_Caption_GPT4O"],
        doc["GT_Caption_GPT4V"],
        doc["GT_Caption_Gemini15Pro"],
    ]


def detailcaps_500_process_result(doc, result):
    pred = result[0]
    payload = {
        "doc_id": int(doc["doc_id"]),
        "image_id": doc["image_id"],
        "source": doc["source"],
        "answer": detailcaps_500_doc_to_target(doc),
        "pred": pred,
    }
    return {
        "detailcaps_CAPTURE": payload,
        "gen_words": float(len(pred.split())),
    }


def _write_sidecar(metric, per_doc, args):
    """doc_id -> score,
    so the paired analysis can see what the corpus aggregate hides."""
    path = generate_submission_file(f"{TASK}_per_doc_scores_{metric}.json", args)
    with open(path, "w") as f:
        json.dump(
            {"task": TASK, "metric": f"detailcaps_{metric}", "scores": per_doc},
            f,
            indent=2,
        )
    eval_logger.info(f"[{TASK}] per-document {metric} scores -> {path}")


def detailcaps_500_capture(results, args=None):
    # CAPTURE consumes raw strings and does its own parsing; PTB-tokenising first would
    # corrupt the scene-graph parse.
    from capture_metric.capture import CAPTURE

    doc_ids = [r["doc_id"] for r in results]
    gts = {i: list(r["answer"]) for i, r in zip(doc_ids, results)}
    res = {i: [r["pred"]] for i, r in zip(doc_ids, results)}
    score, scores = CAPTURE().compute_score(gts, res)
    per_doc = {str(i): float(s) for i, s in zip(doc_ids, list(scores))}
    _write_sidecar("CAPTURE", per_doc, args)
    return float(score)
