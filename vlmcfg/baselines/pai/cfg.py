"""PAI's logits refinement (CFG), ported from LLaMA/LLaVA to Qwen2.5-VL.

Transcribed from ``PAI-master/CFG.py``. The contrastive arithmetic is untouched -- the
log-softmax difference and the adaptive-plausibility cutoff at ``log(0.1) + max logit``:

    out = gamma * (cond - uncond) + uncond,   masked to {cond >= log(0.1) + max cond}

Three Qwen2.5-VL-specific changes, all forced by the backbone rather than chosen:

1. PAI toggles ``self_attn.use_cfg`` on each perturbed layer to switch the attention
   intervention off during the unconditional forward. This port's intervention is one
   handle rather than per-layer attributes (see ``attention.PAIIntervention``), so the
   loop becomes a single flag. Same semantics, one owner.

2. Explicit ``position_ids`` / ``cache_position`` on the unconditional forwards. PAI
   relies on the LM deriving positions itself, which LLaMA does correctly. Qwen2.5-VL
   does not, in two separate ways:

   * The prefill would run ``get_rope_index`` and OVERWRITE the shared
     ``model.model.rope_deltas`` -- the attribute the *conditional* generation reads at
     every decode step to offset its own mRoPE positions. A text-only prefill sets it to
     0, silently corrupting the image-conditioned branch from the second generated token
     onward.
   * The incremental steps pass no ``cache_position``, so ``Qwen2_5_VLModel.forward``
     falls through to ``delta = zeros(...)`` and hands every decode step position 0.

   Passing 2-D ``position_ids`` explicitly skips that whole block (leaving
   ``rope_deltas`` untouched) and gives the text-only branch the plain
   ``0..L-1, L, L+1, ...`` positions it should have.

3. ``build_negative_prompt`` is provided here rather than in the model loader, so the
   lmms-eval wrapper and the standalone eval scripts build the negative prompt the same
   way: PAI's own recipe -- take the rendered prompt and delete the image placeholder,
   changing nothing else.
"""

import re

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
from transformers.cache_utils import DynamicCache

# The un-expanded vision block that Qwen2.5-VL's chat template writes. Deleting it is
# the exact analogue of PAI splitting the LLaVA prompt on `<ImageHere>` and re-joining
# the halves: the negative prompt is the conditional prompt minus the image, nothing
# else.
VISION_BLOCK_RE = re.compile(r"<\|vision_start\|>.*?<\|vision_end\|>", re.DOTALL)


def build_negative_prompt(text: str) -> str:
    """PAI's text-only negative prompt: the rendered prompt with the vision block
    removed."""
    negative, n_removed = VISION_BLOCK_RE.subn("", text)
    if n_removed == 0:
        raise ValueError(
            "No <|vision_start|>...<|vision_end|> block found; "
            "cannot build PAI's negative prompt."
        )
    return negative


class CFGLogits(LogitsProcessor):
    def __init__(
        self,
        guidance_scale,
        uncond,
        model,
        image=None,
        input_type="inputs_ids",
        intervention=None,
        apc=True,
    ):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        self.model = model
        self.image = image
        self.out = None
        self.input_type = input_type
        # `attention.PAIIntervention` handle, or None when --use-attn is off. Replaces
        # PAI's per-layer `self_attn.use_cfg` loop.
        self.intervention = intervention
        # The adaptive plausibility cutoff below is PAI's own and stays on by default.
        # Off is an ABLATION only: it isolates how much of PAI's effect the cutoff
        # carries.
        self.apc = apc

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores
        if self.intervention is not None:
            self.intervention.use_cfg = True

        try:
            if self.out is None:
                # Text-only prefill: plain 0..L-1 positions, and NOT via
                # `get_rope_index` (which would clobber the conditional branch's cached
                # `rope_deltas`).
                bsz, seq_len = self.uncond.shape[0], self.uncond.shape[1]
                position_ids = (
                    torch.arange(seq_len, device=self.uncond.device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
                cache_position = torch.arange(seq_len, device=self.uncond.device)
                if self.input_type == "inputs_ids":
                    # An EXPLICIT empty cache, not just `use_cache=True`. Qwen3-VL's
                    # forward drops `use_cache` into **kwargs and returns
                    # `past_key_values=None`, so the incremental branch below then fails
                    # with `'NoneType' object has no attribute 'get_seq_length'` on the
                    # SECOND decode step. Backbones that already honour `use_cache`
                    # build this same object internally, so handing them an empty one
                    # changes nothing.
                    self.out = self.model(
                        self.uncond,
                        use_cache=True,
                        past_key_values=DynamicCache(),
                        position_ids=position_ids,
                        cache_position=cache_position,
                    )
                elif self.input_type == "inputs_embeds":
                    self.out = self.model(
                        inputs_embeds=self.uncond,
                        use_cache=True,
                        past_key_values=DynamicCache(),
                        position_ids=position_ids,
                        cache_position=cache_position,
                    )
                else:
                    print("Neither input_ids nor inputs_embeds is provided.")
            else:
                past_len = self.out.past_key_values.get_seq_length()
                cache_position = torch.tensor([past_len], device=input_ids.device)
                position_ids = cache_position.view(1, 1).expand(input_ids.shape[0], -1)
                self.out = self.model(
                    input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=self.out.past_key_values,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )
        finally:
            # try/finally, unlike PAI's straight-line reset: leaving `use_cfg` raised
            # after an exception would silently disable the intervention for the rest of
            # the run.
            if self.intervention is not None:
                self.intervention.use_cfg = False

        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)

        cutoff = torch.log(torch.tensor(0.1)) + scores.max(dim=-1, keepdim=True).values
        out = (
            self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        )
        if not self.apc:
            return out
        cd_logits = out.masked_fill(scores < cutoff, -float("inf"))
        return cd_logits
