"""lmms-eval model `molmo2_base`: unguided Molmo2 through HF `generate` (fp16)."""

from typing import Any

from lmms_eval.api.instance import Instance
from tqdm import tqdm

from vlmcfg.backbones.molmo2 import Molmo2
from vlmcfg.models._common import GenerateOnly, cut_at_stop_strings, flatten


class Molmo2Base(GenerateOnly):
    """Greedy Molmo2, fed the same inputs as `molmo2_vlmcfg`'s conditional stream."""

    is_simple = True
    default_max_new_tokens = 64

    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-4B",
        batch_size: int = 1,
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError("the baseline runs one document at a time (batch_size=1)")
        self.backbone = Molmo2(pretrained, **backbone_kwargs)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        answers = []
        for ctx, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm(
            [r.args for r in requests], disable=self.rank != 0, desc="base"
        ):
            visuals = flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            max_new = int(
                gen_kwargs.get("max_new_tokens") or self.default_max_new_tokens
            )
            answer = cut_at_stop_strings(
                self._generate_one(ctx, visuals or None, max_new), gen_kwargs
            )
            answers.append(answer)
            self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), answer)
        return answers

    def _generate_one(self, text: str, images: list | None, max_new_tokens: int) -> str:
        """One answer; methods with their own decode loop override this."""
        return self.backbone.generate(
            self.backbone.build_inputs(images, text), max_new_tokens
        )
