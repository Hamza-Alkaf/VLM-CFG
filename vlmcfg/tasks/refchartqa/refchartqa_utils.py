def refchartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def refchartqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['query']}{post_prompt}"


def refchartqa_doc_to_target(doc):
    return doc["label"]


def _to_float(text):
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def _relaxed_correctness(prediction, target, max_relative_change=0.05):
    """Standard ChartQA/pix2struct relaxed accuracy: numeric answers are correct
    within `max_relative_change` relative tolerance, everything else falls back
    to a case-insensitive exact string match."""
    prediction = prediction.strip()
    target = target.strip()

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float is not None:
        if target_float == 0:
            return prediction_float == 0
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change

    return prediction.lower() == target.lower()


def refchartqa_process_results(doc, results):
    prediction = results[0]
    target = doc["label"]
    return {"relaxed_overall": float(_relaxed_correctness(prediction, target))}


def refchartqa_aggregate_mean(results):
    return sum(results) / len(results) if results else 0.0
