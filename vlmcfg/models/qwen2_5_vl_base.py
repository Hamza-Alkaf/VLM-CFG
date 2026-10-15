"""lmms-eval model `qwen2_5_vl_base`: unguided Qwen2.5-VL, the paper's baseline.

The contrastive-decoding baselines subclass this wrapper: they swap the backbone's HF
model class for one that contrasts inside `forward`, and use the two hooks below to
build extra inputs or wrap the `generate` call.
"""

import contextlib
from typing import Any

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.protocol import ChatMessages
from tqdm import tqdm

from vlmcfg.backbones.qwen2_5_vl import Qwen25VL, first_image
from vlmcfg.models._common import GenerateOnly, resolve_gen_kwargs
from vlmcfg.negatives import RegionConfig, conditional_context


class Qwen25VLBase(GenerateOnly):
    """Greedy decoding through HF `generate`, fed lmms-eval's chat messages.

    Keywords named after a `RegionConfig` field steer the attention towards an annotated
    region (Qwen2.5-VL); any other besides `pretrained` goes to the backbone. Documents
    are processed in lmms-eval's `Collator` order (sorted by prompt), which matters for
    methods that draw random noise: the noise a document gets depends on the documents
    before it.
    """

    is_simple = False
    backbone_cls: Any = Qwen25VL
    default_pretrained = "Qwen/Qwen2.5-VL-7B-Instruct"
    default_backbone_kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}

    def __init__(
        self, pretrained: str | None = None, batch_size: int = 1, **backbone_kwargs: Any
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError("the baseline runs one document at a time (batch_size=1)")
        self.region, backbone_kwargs = RegionConfig.from_args(backbone_kwargs)
        if self.region.cond_attn_op != "none" and self.backbone_cls is not Qwen25VL:
            raise NotImplementedError("attention edits are implemented for Qwen2.5-VL")
        self.backbone = self.backbone_cls(
            pretrained or self.default_pretrained,
            **{**self.default_backbone_kwargs, **backbone_kwargs},
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        ordered = utils.Collator(
            [r.args for r in requests],
            lambda x: (x[0], x[0]),
            group_fn=lambda x: x[2],
            grouping=True,
        )
        answers = []
        for chunk in tqdm(
            ordered.get_batched(n=1, batch_fn=None),
            total=len(requests),
            disable=self.rank != 0,
            desc=type(self).__name__,
        ):
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = chunk[0]
            reset = getattr(self.backbone.model, "reset_sample_tracking", None)
            if reset is not None:
                reset(doc_id=doc_id)
            doc = self.task_dict[task][split][doc_id]
            self._doc, self._doc_id = doc, doc_id
            chat = ChatMessages(messages=doc_to_messages(doc))
            texts, inputs = self._chat_inputs(chat)
            gen = resolve_gen_kwargs(gen_kwargs)
            context, extra = self._generation_context(texts, inputs, chat, gen)
            with context:
                raw = self.backbone.generate(inputs, gen, **extra)
            answer = parse_reasoning_model_answer(raw)
            answers.append(answer)
            self.cache_hook.add_partial(
                "generate_until", (texts[0], gen_kwargs), answer
            )
        return ordered.get_original(answers)

    def _chat_inputs(self, chat: ChatMessages) -> tuple[list[str], dict]:
        """Rendered prompt and processor inputs for one document."""
        return self.backbone.build_chat_inputs(chat.to_hf_messages())

    def _generation_context(
        self, texts: list[str], inputs: dict, chat: ChatMessages, gen_kwargs: dict
    ) -> tuple[contextlib.AbstractContextManager, dict]:
        """A context to run `generate` in, and extra `generate` kwargs.

        By default the region's attention edit when one is configured, else inert.
        """
        if self.region.cond_attn_op == "none":
            return contextlib.nullcontext(), {}
        image = first_image(chat.to_hf_messages())
        edit = conditional_context(
            self.backbone, inputs, self.region, self._doc, self._doc_id, image
        )
        assert edit is not None
        return edit(), {}
