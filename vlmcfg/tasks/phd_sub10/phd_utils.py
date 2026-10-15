"""Task hooks for the PhD 10%-subset benchmark.

The parser and the prompt suffix are this project's own, not the PhD paper's; the
paper's PhD numbers were produced with them.
"""

from __future__ import annotations

import re
from typing import Any

import datasets

# The rationale arm asks for a one-sentence "Reasoning:" line and then
# "Answer: <yes or no>". An explicit answer DECLARATION, last match wins -- the
# reasoning sentence routinely opens with "Yes, ..." or "No, ..." before the model
# settles its answer on the final line, which is exactly what parse_answer's first-token
# rule would misread.
#   "Answer: no"  "**Answer:** Yes"  "the answer is yes"  "Answer: [no]"  "\boxed{yes}"
RATIONALE_STRICT_RE = re.compile(
    r"\banswer\s*(?:is|=|:|-|\*)+\s*[\[\(\"'<]?\s*(yes|no)\b"
    r"|\\boxed\{\s*(yes|no)\s*\}",
    re.IGNORECASE,
)
# A response with no declaration falls back to parse_answer only when it is short enough
# to BE an answer. Anything longer either ran into max_new_tokens or ignored the format,
# and its first word is reasoning prose, not a verdict. Mirrors mathverse_mcq/utils.py.
RATIONALE_LOOSE_MAX_WORDS = 50

YES_ALIASES = {"yes", "y", "yeah", "yep", "true"}
NO_ALIASES = {"no", "n", "nope", "false"}

METRICS = (
    "phd_index",
    "phd_yes_recall",
    "phd_no_recall",
    "phd_accuracy",
    "phd_invalid_rate",
)


def parse_answer(ans: Any) -> str | None:
    """Return 'yes', 'no', or None (invalid). Never raises."""
    if ans is None:
        return None
    s = str(ans).strip().lower()
    if not s:
        return None

    # take first token; strip common punctuation
    tok = s.split()[0].strip(".,!?;:\"'()[]{}")
    if tok in YES_ALIASES:
        return "yes"
    if tok in NO_ALIASES:
        return "no"
    return None


def phd_doc_to_visual(doc: dict[str, Any]) -> list[Any]:
    """Return the single image this question is asked about."""
    return [doc["image"].convert("RGB")]


def phd_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None
) -> str:
    """Render the prompt.

    The question is not stripped: iac/icc questions begin with context prose and
    are concatenated raw upstream, so trimming here would silently change the
    prompt the paper's runs used.
    """
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def phd_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    """Score one prediction; an unparseable answer counts as wrong."""
    parsed = parse_answer(results[0])
    gt = doc["answer"]
    payload = {
        "question_id": doc["question_id"],
        "mode": doc["mode"],
        "task": doc["task"],
        "gt": gt,
        "parsed": parsed,
        "score": 1.0 if parsed == gt else 0.0,
    }
    return {metric: payload for metric in METRICS}


def parse_rationale_answer(ans: Any) -> tuple[str | None, str]:
    """-> ('yes' | 'no' | None, 'strict' | 'loose' | 'no_answer')."""
    s = str(ans or "").strip()
    matches = list(RATIONALE_STRICT_RE.finditer(s))
    if matches:
        last = matches[-1]
        return (last.group(1) or last.group(2)).lower(), "strict"
    if s and len(s.split()) <= RATIONALE_LOOSE_MAX_WORDS:
        parsed = parse_answer(s)
        if parsed is not None:
            return parsed, "loose"
    return None, "no_answer"


def phd_process_results_rationale(
    doc: dict[str, Any], results: list[str]
) -> dict[str, Any]:
    """Score the rationale arm on the DECLARED answer.

    Same payload as phd_process_results, so every phd_* aggregation is shared, plus the
    three covariates that make an accuracy delta readable: how often the answer came
    from an explicit declaration, how often there was none, and output length.
    """
    raw = str(results[0])
    parsed, how = parse_rationale_answer(raw)
    gt = doc["answer"]
    payload = {
        "question_id": doc["question_id"],
        "mode": doc["mode"],
        "task": doc["task"],
        "gt": gt,
        "parsed": parsed,
        "score": 1.0 if parsed == gt else 0.0,
    }
    out: dict[str, Any] = {metric: payload for metric in METRICS}
    out["parse_rate_strict"] = 1.0 if how == "strict" else 0.0
    out["gen_words"] = float(len(raw.split()))
    return out


def _recall(results: list[dict[str, Any]], polarity: str) -> float:
    """Recall over the questions whose ground truth is ``polarity``."""
    total = sum(1 for r in results if r["gt"] == polarity)
    if not total:
        return 0.0
    correct = sum(1 for r in results if r["gt"] == polarity and r["parsed"] == polarity)
    return correct / total


def phd_aggregate_yes_recall(results: list[dict[str, Any]]) -> float:
    """Fraction of yes-questions answered yes."""
    return _recall(results, "yes")


def phd_aggregate_no_recall(results: list[dict[str, Any]]) -> float:
    """Fraction of no-questions answered no."""
    return _recall(results, "no")


def phd_aggregate_index(results: list[dict[str, Any]]) -> float:
    """PhD Index: the harmonic mean of yes-recall and no-recall."""
    yes_recall = _recall(results, "yes")
    no_recall = _recall(results, "no")
    if yes_recall + no_recall <= 0:
        return 0.0
    return 2 * yes_recall * no_recall / (yes_recall + no_recall)


def phd_aggregate_accuracy(results: list[dict[str, Any]]) -> float:
    """Plain accuracy over every question."""
    return sum(r["score"] for r in results) / len(results)


def phd_aggregate_invalid_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of predictions the parser could not resolve to yes/no."""
    return sum(1 for r in results if r["parsed"] is None) / len(results)


def filter_base(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep the plain-question mode."""
    return dataset.filter(lambda doc: doc["mode"] == "base")


def filter_iac(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep the inaccurate-context mode."""
    return dataset.filter(lambda doc: doc["mode"] == "iac")


def filter_icc(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep the incorrect-context mode."""
    return dataset.filter(lambda doc: doc["mode"] == "icc")


def filter_ccs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep the counter-common-sense mode."""
    return dataset.filter(lambda doc: doc["mode"] == "ccs")
