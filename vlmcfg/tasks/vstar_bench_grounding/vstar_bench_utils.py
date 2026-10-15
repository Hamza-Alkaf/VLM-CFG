import re

from lmms_eval.filters.extraction import ExtendedRegexFilter

OPTION_LETTERS = list("ABCD")


def vstar_bench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vstar_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    options = [(letter, doc[letter]) for letter in OPTION_LETTERS if doc.get(letter)]
    choices_str = "\n".join(f"({letter}) {text}" for letter, text in options)
    return f"{pre_prompt}{doc['question']}\n{choices_str}{post_prompt}"


def vstar_bench_doc_to_target(doc):
    return doc["answer"]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use
            the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to
                          find these choices in the response.
                        - step 2 : We parse the choice with regex :[\\s]*([A-?]),
                          where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r in resps:
            filtered = [self.extract_option(resp) for resp in r]
            filtered_resps.append(filtered[0])

        return filtered_resps

    # Tried in order; the first match wins. The original pattern here was
    # `^\s*([A-Z])\.` alone -- a letter followed by a PERIOD -- so a model answering in
    # the parenthesised form the prompt's own choices are written in, "(A) rubber",
    # matched nothing and the raw string was compared against the target "A". Measured
    # on Qwen3-VL-4B base: 68 of 191 responses were "(X) text" and 56 of those were
    # CORRECT but scored 0.0, putting exact_match at 0.4293 where the true value is
    # 0.7225.
    OPTION_PATTERNS = (
        re.compile(r"^\s*\(\s*([A-Z])\s*\)"),  # (A) rubber
        re.compile(r"^\s*\*{1,2}\s*([A-Z])\s*\*{1,2}"),  # **A**  /  *A*
        re.compile(r"^\s*([A-Z])\s*[\).:,;]"),  # A) rubber | A. rubber | A: rubber
        re.compile(r"^\s*([A-Z])\s*$"),  # a bare letter
        re.compile(r"answer\W{0,20}?\(?([A-Z])\)?\b", re.I),  # "The answer is (A)"
    )

    def extract_option(self, resp: str) -> str:
        """Pull the option letter out of a response, or return it untouched.

        Returning the response unchanged on no match is deliberate: exact_match still
        compares it against the target, so a bare "A" scores correctly even though no
        pattern above is needed for it, and an unparseable answer stays visible in the
        samples file instead of being silently rewritten to a wrong letter.
        """
        if not isinstance(resp, str):
            return resp
        for pat in self.OPTION_PATTERNS:
            m = pat.search(resp) if pat.flags & re.I else pat.match(resp)
            if m:
                return m.group(1).upper()
        return resp
