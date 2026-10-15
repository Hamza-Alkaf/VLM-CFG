import re

import datasets
import numpy as np
from pycocotools import mask as mask_util

_ARTICLES = {"a", "an", "the"}

# Small, common subset of the contraction normalization used by the official
# VQA/VizWiz eval scripts — keeps "don't" vs "dont" etc. from being treated
# as different answers after punctuation stripping.
_CONTRACTIONS = {
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
    "its": "it's",
    "thats": "that's",
    "theres": "there's",
    "whats": "what's",
    "im": "i'm",
    "ive": "i've",
    "youre": "you're",
    "youve": "you've",
}

_MANUAL_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_PUNCTUATION_RE = re.compile(r"[^\w\s']")


def _vizwiz_grounding_normalize_answer(answer):
    """Lowercase, strip punctuation/articles,
    normalize a few contractions and number words."""
    answer = answer.lower().strip()
    answer = _PUNCTUATION_RE.sub(" ", answer)
    words = answer.split()
    words = [_MANUAL_NUMBER_MAP.get(w, w) for w in words]
    words = [_CONTRACTIONS.get(w, w) for w in words]
    words = [w for w in words if w not in _ARTICLES]
    return " ".join(words).strip()


def _vizwiz_grounding_mask_to_rle(example):
    """Encode the raw `mask` PNG (a binary foreground/background image)
    as a COCO RLE dict.

    `--cfg_mode seg` reads a `segmentation` column expecting a polygon list or a COCO
    RLE dict (see `att_grounding.build_seg_mask`); this dataset instead ships the object
    mask as an `Image` column, which neither format accepts directly.
    """
    binary_mask = (np.array(example["mask"]) > 127).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    example["segmentation"] = rle
    return example


def vizwiz_grounding_process_docs(docs: datasets.Dataset) -> datasets.Dataset:
    return docs.map(_vizwiz_grounding_mask_to_rle)


def vizwiz_grounding_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vizwiz_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def vizwiz_grounding_doc_to_target(doc):
    return doc["answers"]


def vizwiz_grounding_process_results(doc, results):
    prediction = _vizwiz_grounding_normalize_answer(results[0])
    gt_answers = [_vizwiz_grounding_normalize_answer(a) for a in doc["answers"]]

    # standard VQA soft-accuracy: average, over each held-out reference answer,
    # of min(#matches among the other 9 / 3, 1)
    accuracies = []
    for i in range(len(gt_answers)):
        other_answers = gt_answers[:i] + gt_answers[i + 1 :]
        matches = sum(1 for a in other_answers if a == prediction)
        accuracies.append(min(matches / 3.0, 1.0))

    return {"vqa_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0}


def vizwiz_grounding_aggregate_accuracy(results):
    return sum(results) / len(results) if results else 0.0
