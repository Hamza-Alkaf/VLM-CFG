"""AMBER generative metrics -- CHAIR / Cover / Hal / Cog.

Ported from the official scorer (github.com/junyangwang0410/AMBER, `inference.py`, the
`dimension['g']` branch). The matching logic is kept step-for-step identical, including
the order in which a noun is tested against the exact-match lists before the spaCy
similarity fallback, because that order decides which bucket an ambiguous noun lands in.

Two deliberate departures, neither of which changes a number:

1. `check_synonyms_word` re-parsed both words on every call, which is ~2M spaCy parses
   over this split. Parsed Docs are cached per unique word here instead; spaCy is
   deterministic, so the similarities are identical.
2. The official script keeps only global counters. We additionally emit PER-DOCUMENT
   numerator/denominator in the samples, for paired per-item comparisons. The reported
   aggregate is still sum(num)/sum(den), exactly as upstream -- it is NOT a mean of the
   per-document ratios, which would weight a 3-object image like a 20-object one.
"""

import json
import logging
import os
from functools import lru_cache

from vlmcfg.datapath import data_dir

eval_logger = logging.getLogger("lmms-eval")

# relation.json and safe_words.txt, written next to the parquet by data/prepare_amber.py
AMBER_ROOT = data_dir() / "amber"
SIMILARITY_SCORE = 0.8

_STATE = {}


def _init():
    """Lazy: spaCy's en_core_web_lg is ~400 MB and must not load at import time."""
    if _STATE:
        return _STATE
    import nltk
    import spacy
    from nltk.stem import WordNetLemmatizer

    _STATE["nlp"] = spacy.load("en_core_web_lg")
    _STATE["lemmatizer"] = WordNetLemmatizer()
    _STATE["nltk"] = nltk

    association = json.load(
        open(os.path.join(AMBER_ROOT, "relation.json"), encoding="utf-8")
    )
    hallucination_words = []
    for w1, ws in association.items():
        hallucination_words.append(w1)
        hallucination_words.extend(ws)
    _STATE["association"] = association
    _STATE["hallucination_words"] = set(hallucination_words)
    with open(os.path.join(AMBER_ROOT, "safe_words.txt"), encoding="utf-8") as f:
        _STATE["global_safe_words"] = [ln.split("\n")[0] for ln in f]
    return _STATE


@lru_cache(maxsize=100000)
def _doc(word):
    return _init()["nlp"](word)


def _similar(w1, w2):
    return _doc(w1).similarity(_doc(w2)) > SIMILARITY_SCORE


def _extract_nouns(text):
    s = _init()
    tokens = s["nltk"].word_tokenize(text)
    tagged = s["nltk"].pos_tag(tokens)
    return [s["lemmatizer"].lemmatize(w) for w, pos in tagged if pos.startswith("NN")]


def amber_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def amber_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    k = lmms_eval_specific_kwargs or {}
    return f"{k.get('pre_prompt', '')}{doc['query']}{k.get('post_prompt', '')}"


def amber_doc_to_target(doc):
    return list(doc["truth"])


def _score_one(response, truth, hallu):
    """The official generative branch, for a single document."""
    s = _init()
    association = s["association"]

    nouns = [n for n in _extract_nouns(response) if n in s["hallucination_words"]]

    safe_words, safe_list = [], []
    for idx, word in enumerate(truth):
        safe_words += association[word]
        safe_list += [idx] * len(association[word])
    ha_words, ha_list = [], []
    for idx, word in enumerate(hallu):
        ha_words += association[word]
        ha_list += [idx] * len(association[word])

    safe_words += list(truth)
    safe_len = len(truth)
    safe_list += [0] * safe_len
    safe_flag_list = [0] * len(nouns)

    ha_words += list(hallu)
    ha_len = len(hallu)
    ha_list += [0] * ha_len

    for idx, noun in enumerate(nouns):
        if noun in s["global_safe_words"]:
            continue

        if noun in safe_words:
            for j in range(len(safe_words)):
                if noun == safe_words[j]:
                    if j < (len(safe_list) - safe_len):
                        safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                    else:
                        safe_list[j] = 1
                    break
            continue

        if noun in ha_words:
            for j in range(len(ha_words)):
                if noun == ha_words[j]:
                    if j < (len(ha_list) - ha_len):
                        ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                    else:
                        ha_list[j] = 1
                    break

        for j, check_word in enumerate(ha_words):
            if _similar(noun, check_word):
                if j < (len(ha_list) - ha_len):
                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                else:
                    ha_list[j] = 1
                break

        flag = False
        for j, check_word in enumerate(safe_words):
            if _similar(noun, check_word):
                flag = True
                if j < (len(safe_list) - safe_len):
                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                else:
                    safe_list[j] = 1
                break
        if flag:
            continue

        safe_flag_list[idx] = 1

    return {
        "chair_hit": sum(safe_flag_list),
        "chair_total": len(safe_flag_list),
        "cover_hit": sum(safe_list[-safe_len:]) if safe_len else 0,
        "cover_total": safe_len,
        "cog_hit": sum(ha_list[-ha_len:]) if ha_len else 0,
        "cog_total": ha_len,
        "hal": 1.0 if sum(safe_flag_list) > 0 else 0.0,
    }


def amber_process_results(doc, results):
    pred = results[0].strip()
    r = _score_one(pred, list(doc["truth"]), list(doc["hallu"]))

    def ratio(h, t):
        # A response mentioning no in-vocabulary noun cannot hallucinate; upstream lets
        # such a document contribute nothing to the corpus ratio, and 0.0 is its
        # per-item image.
        return float(h) / t if t else 0.0

    return {
        "CHAIR": {
            "score": ratio(r["chair_hit"], r["chair_total"]),
            "hit": r["chair_hit"],
            "total": r["chair_total"],
        },
        "Cover": {
            "score": ratio(r["cover_hit"], r["cover_total"]),
            "hit": r["cover_hit"],
            "total": r["cover_total"],
        },
        "Cog": {
            "score": ratio(r["cog_hit"], r["cog_total"]),
            "hit": r["cog_hit"],
            "total": r["cog_total"],
        },
        "Hal": r["hal"],
        "gen_words": float(len(pred.split())),
    }


def _corpus(results):
    """sum(hit)/sum(total) -- upstream's aggregate,
    not a mean of per-document ratios."""
    hit = sum(r["hit"] for r in results)
    total = sum(r["total"] for r in results)
    return 100.0 * hit / total if total else 0.0


def amber_chair(results):
    return _corpus(results)


def amber_cover(results):
    return _corpus(results)


def amber_cog(results):
    return _corpus(results)


def amber_hal(results):
    return 100.0 * sum(results) / len(results) if results else 0.0
