import re

_ARTICLES = {"a", "an", "the"}
_PUNCTUATION_RE = re.compile(r"[^\w\s']")


def _openimages_normalize_answer(answer):
    answer = answer.lower().strip()
    answer = _PUNCTUATION_RE.sub(" ", answer)
    words = [w for w in answer.split() if w not in _ARTICLES]
    return " ".join(words).strip()


def _openimages_levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _openimages_anls_single(prediction, reference):
    pred = prediction.strip().lower()
    ref = reference.strip().lower()
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0
    normalized_distance = _openimages_levenshtein(pred, ref) / max_len
    return 1.0 - normalized_distance if normalized_distance < 0.5 else 0.0


def openimages_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def openimages_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def openimages_doc_to_target(doc):
    return doc["answers"]


def openimages_process_results(doc, results):
    prediction = results[0]
    references = doc["answers"]

    normalized_prediction = _openimages_normalize_answer(prediction)
    exact_match = float(
        any(
            normalized_prediction == _openimages_normalize_answer(ref)
            for ref in references
        )
    )
    anls = max(
        (_openimages_anls_single(prediction, ref) for ref in references), default=0.0
    )

    return {"exact_match": exact_match, "anls": anls}


def openimages_aggregate_mean(results):
    return sum(results) / len(results) if results else 0.0
