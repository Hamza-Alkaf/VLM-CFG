import re

from lmms_eval.filters.extraction import ExtendedRegexFilter


def treebench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def treebench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    # `multi_choice_options` is a pre-rendered "A. ...\nB. ...\n..." block; the
    # per-letter A-K columns carry the same text but are only populated for a subset of
    # rows (the "Perception/OCR" category leaves them null), so `multi_choice_options`
    # is the one field guaranteed non-empty for every row.
    return f"{pre_prompt}{doc['question']}\n{doc['multi_choice_options']}{post_prompt}"


def treebench_doc_to_target(doc):
    # `answer` is already the ground-truth option letter (e.g. "C"), unlike tasks that
    # store the answer text and require looking up its index among the choices.
    return doc["answer"]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use
            the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to
                          find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]),
                          where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = []
        option_letter_regex = re.compile(r"^\s*([A-Z])\.")

        for r, _doc in zip(resps, docs):
            filtered = []
            for resp in r:
                match = option_letter_regex.match(resp)
                if match:
                    filtered.append(match.group(1))
                else:
                    filtered.append(resp)
            filtered_resps.append(filtered[0])

        return filtered_resps
