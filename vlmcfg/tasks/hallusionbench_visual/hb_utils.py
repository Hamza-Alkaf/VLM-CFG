"""Task hooks for the visual-only HallusionBench benchmark.

Scoring is a strict exact match on the yes/no answer, not the upstream GPT-4 judge used
by lmms-eval's ``hallusion_bench_image`` task.
"""

from __future__ import annotations

from typing import Any


def hb_doc_to_visual(doc: dict[str, Any]) -> list[Any]:
    """Return the single image this question is asked about."""
    return [doc["image"].convert("RGB")]


def hb_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None
) -> str:
    """Render the prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def hb_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    """Score one prediction by whole-string exact match, ignoring case.

    Whole-string and unforgiving otherwise: "yes." and "yes, it is" both score
    zero. Case is the one relaxation, and it is not cosmetic -- the baselines
    capitalise differently on some backbones (LayerCD returned 'Yes'/'No'
    where VLM-CFG returned lowercase), so a case-sensitive compare would report
    a capitalisation difference as a hallucination result.
    """
    pred = str(results[0]).strip().lower()
    gt = str(doc["answer"]).strip().lower()
    return {
        "hb_accuracy": {
            "question_id": doc["question_id"],
            "category": doc["category"],
            "subcategory": doc["subcategory"],
            "gt": gt,
            "prediction": pred,
            "score": 1.0 if pred == gt else 0.0,
        }
    }


def hb_aggregate_accuracy(results: list[dict[str, Any]]) -> float:
    """Exact-match accuracy over every question."""
    return sum(r["score"] for r in results) / len(results)
