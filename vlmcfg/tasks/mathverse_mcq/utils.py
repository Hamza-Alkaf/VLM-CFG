"""MathVerse MCQ -- the multiple-choice subset of MathVerse testmini,
scored WITHOUT a judge.

Why this task exists alongside `tasks/mathverse`:

1. **No GPT.** Upstream sets `quick_match: false`, so every document costs a
   `gpt-4o-2024-11-20` call that judges the last 30 words (`mathverse_evals.py:92`).
   That is an API dependency and a drift source between arms run weeks apart. Restricted
   to multiple-choice with a clean single-letter ground truth (426 of 436 problems), the
   answer is a letter and a regex settles it exactly.

2. **Long-form on demand.** Upstream only exposes `query_wo` ("directly answer ..."),
   which emits a handful of tokens -- the short-answer regime. `query_cot` ships in the
   same rows and asks for reasoning first. Both are exposed here as separate tasks so
   the SAME 426 problems can be run with and without reasoning.

3. **Format compliance is reported, never absorbed.** A parse failure scores 0 on `acc`,
   which silently converts an instruction-following change into an accuracy change --
   the exact failure documented for ScreenSpot in cfg_samples_20260913/README.md, where
   parse rate swung 43%->92% while grounding quality held flat. So `parse_rate_strict`,
   `parse_rate_any` and `acc_parsed` (accuracy over parsed documents only) are reported
   beside `acc`. If `acc` moves and `acc_parsed` does not, the effect is formatting.

Six versions x two arms = twelve tasks. `text_only` carries no image at all, so the
negative stream is built from exactly the same inputs as the conditional. What that
implies depends entirely on the AR reference, and getting this wrong is easy:

  * `--cfg_ar_type i-i` and the teacher-forced model: the reference is the negative
    stream at the SAME step, so u(t) = c(t), the delta is identically zero, and alpha
    cannot change the output. A non-zero delta there IS a bug.
  * `--cfg_ar_type i-1` (the project default): the reference is frozen at the negative
    stream's STEP 0 (`chat/qwen2_5_vl.py:1044`, `get_no_img_ref(..., step_idx=0)`). With
    no image u(0) = c(0), but the guidance at step t is alpha * (c(t) - c(0)) --
    comparing the current distribution against the model's own INITIAL one. That is not
    zero and it is not about the image. Measured on text_only_direct: 69/426 responses
    changed between alpha=1.0 and alpha=1.5.

So under i-1 this split is not a bug check; it is a MEASUREMENT of the non-image
component of the guidance, isolated. alpha=1.0 remains an exact no-op for every ar_type,
since u + 1*(c-u) = c whatever u is.
"""

import logging
import re

eval_logger = logging.getLogger("lmms-eval")

# An explicit answer DECLARATION -- something a model writes deliberately to state its
# final choice, as opposed to a letter that merely appears in its working. Any of these
# counts, at ANY length:
#   "Answer: B"  "**Answer:** B"  "the answer is C"  "final answer - b"
#   "the correct option is (C)"  "correct choice: B"
#   "\boxed{D}"  "answer is \boxed{D}"  "\(\boxed{D}\)"
#
# `\boxed{}` was originally only in the loose tier, behind the 50-word guard. That was a
# bug, and an arm-dependent one: under teacher-forced CFG the model concludes long,
# complete chains with "The correct answer is \boxed{D}." (39 of 58 "no-answers" on
# text_dominant_cot at alpha=1.5), so the guard scored valid answers as unanswered and
# biased that arm's accuracy down. `\boxed{}` is a final-answer convention; it does not
# appear incidentally mid-derivation.
#
# `(?:option|choice)` requires "is" / ":" / "=" before the letter, so the choices echoed
# from the question ("A:36°") and working that names an option ("option A looks right")
# do not match.
STRICT_RE = re.compile(
    r"(?:"
    r"(?:answer|option|choice)\s*(?:is|=|:|-|\*)+\s*(?:\\\(|\$)?\s*(?:\\boxed\{)?\s*\(?\s*([A-Ha-h])\s*\)?(?![A-Za-z])"
    r"|\\boxed\{\s*\(?\s*([A-Ha-h])\s*\)?\s*\}"
    r")",
    re.IGNORECASE,
)
# Fallbacks, tried in order, applied ONLY to short responses -- see parse_letter.
LOOSE_RES = [
    re.compile(r"\\boxed\{\s*\(?([A-Ha-h])\)?\s*\}"),
    re.compile(r"\(([A-Ha-h])\)"),
    re.compile(r"(?:^|[\s\*])([A-H])(?=[\s\.\,\:\)]|$)"),
]

# A response longer than this with NO answer declaration did not answer: it either ran
# into max_new_tokens mid-derivation or rambled. Measured on the 50-doc pilot, the split
# is unambiguous -- the two bare direct answers were 1 word, the two truncations 563 and
# 717, and every genuine chain in between declared its answer.
LOOSE_MAX_WORDS = 50

FORMAT_COT = (
    "\nThink step by step, then give your final answer on a new line in exactly "
    "this format:\n"
    "Reasoning: <your reasoning>\n"
    "Answer: <the option letter>"
)
FORMAT_DIRECT = (
    "\nGive your answer on a single line in exactly this format:\n"
    "Answer: <the option letter>"
)


def mathverse_mcq_doc_to_visual(doc):
    img = doc.get("image")
    if img is None:
        return []  # text_only -- no image, by design
    return [img.convert("RGB")]


def mathverse_mcq_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    k = lmms_eval_specific_kwargs or {}
    query = doc[k.get("query_type", "query_wo")]
    return f"{query}{k.get('format_prompt', '')}"


def mathverse_mcq_doc_to_target(doc):
    return doc["answer"]


def parse_letter(text):
    """-> (letter or None, 'strict' | 'loose' | 'no_answer').

    Last match wins: a reasoning chain routinely names other options on the way to its
    conclusion.

    The length guard on the loose tier is the load-bearing part. Without it, a chain
    that hit max_new_tokens mid-derivation gets a letter scraped out of its own working,
    which scores ~1-in-4 by luck. Two of fifty documents did exactly that in the pilot
    (563 and 717 words, no declaration), one scoring 1.0 and one 0.0 -- pure noise
    entering the metric while `parse_rate_any` reported 1.00. Since CFG moves
    termination, it moves the truncation rate too, so that noise would not even be
    balanced between arms. A long response with no declaration is therefore recorded as
    NOT ANSWERED rather than guessed at, and `no_answer_rate` reports how often it
    happens.
    """
    m = list(STRICT_RE.finditer(text))
    if m:
        last = m[-1]
        return (last.group(1) or last.group(2)).upper(), "strict"
    if len(text.split()) <= LOOSE_MAX_WORDS:
        for rx in LOOSE_RES:
            m = list(rx.finditer(text))
            if m:
                return m[-1].group(1).upper(), "loose"
    return None, "no_answer"


def mathverse_mcq_process_results(doc, results):
    pred = results[0].strip()
    gt = str(doc["answer"]).strip().upper()
    letter, how = parse_letter(pred)
    parsed = letter is not None
    correct = 1.0 if (parsed and letter == gt) else 0.0

    return {
        "acc": correct,
        # Two numeric fields and no "score" key, so per-item paired analyses skip this
        # as ambiguous rather than double-reporting acc. The aggregation below uses it.
        "acc_parsed": {"correct": correct, "parsed": 1.0 if parsed else 0.0},
        "parse_rate_strict": 1.0 if how == "strict" else 0.0,
        "parse_rate_any": 1.0 if parsed else 0.0,
        # Read this one before believing any acc delta: it is the rate at which the
        # model produced no answer at all, and it is the channel through which a
        # termination change becomes an apparent accuracy change.
        "no_answer_rate": 0.0 if parsed else 1.0,
        "gen_words": float(len(pred.split())),
    }


def mathverse_mcq_acc_parsed(results):
    """A FRACTION, same units as `acc` -- the two are meant to be read side by side, and
    a 0.64 next to a 64.0 invites exactly the misreading this metric exists to
    prevent."""
    n = sum(r["parsed"] for r in results)
    return sum(r["correct"] for r in results) / n if n else 0.0
