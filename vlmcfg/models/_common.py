"""Shared lmms-eval plumbing for the plugin's model classes."""

from typing import Any

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

# Generation defaults of the chat pipeline the baselines ran on; tasks override them.
BASE_GEN_DEFAULTS = {
    "max_new_tokens": 50,
    "temperature": 0.0,
    "top_p": None,
    "num_beams": 1,
}


def resolve_gen_kwargs(gen_kwargs: dict) -> dict:
    """Merge task kwargs over the defaults; temperature 0 means greedy."""
    merged = {**BASE_GEN_DEFAULTS, **gen_kwargs}
    if merged["temperature"] > 0:
        merged["do_sample"] = True
    else:
        merged.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return merged


def flatten(nested: list[list[Any]]) -> list[Any]:
    return [x for inner in nested for x in inner]


class GenerateOnly(lmms):
    """An lmms model that only implements `generate_until`."""

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(f"{type(self).__name__} supports generate_until only")

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError(f"{type(self).__name__} supports generate_until only")


def cut_at_stop_strings(answer: str, gen_kwargs: dict) -> str:
    """Truncate at the first of the task's `until` strings (the Molmo2 runs did
    this)."""
    for term in gen_kwargs.get("until") or []:
        if term:
            answer = answer.split(term)[0]
    return answer
