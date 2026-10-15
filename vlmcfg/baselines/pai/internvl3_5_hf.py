"""The InternVL3.5-HF model class PAI runs on: the plain class plus a per-document
hook."""

from transformers import InternVLForConditionalGeneration


class InternVLForConditionalGenerationPAI(InternVLForConditionalGeneration):
    """PAI acts from outside `forward` (an attention wrapper and a logits processor)."""

    def reset_sample_tracking(self, doc_id=None) -> None:
        """No per-document state: PAI's state lives on its handle, rebuilt every
        document."""
