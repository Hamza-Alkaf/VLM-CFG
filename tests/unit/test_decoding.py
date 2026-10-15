"""CPU tests of the guided decode loop against a fake backbone."""

import torch

from vlmcfg.decoding import Stream, guided_decode
from vlmcfg.guidance import GuidanceConfig, apc_mask, blend

VOCAB = 6
EOS = 5


class FakeBackbone:
    """Logits backbone whose prediction depends on the stream length;
    logs each forward."""

    blend_form = "reference"
    eos_ids = frozenset({EOS})

    def __init__(self, stop_after: int) -> None:
        self.stop_after = stop_after
        self.calls: list[tuple[str, int, bool]] = []

    @property
    def lm_head(self):
        return lambda h: h

    def past_length(self, stream: Stream) -> int:
        return stream.past

    def step(self, stream: Stream, first: bool, feature: str) -> torch.Tensor:
        self.calls.append((stream.state["name"], stream.ids.shape[1], first))
        stream.past = stream.ids.shape[1]
        n_generated = stream.ids.shape[1] - stream.state["prompt_len"]
        token = EOS if n_generated >= self.stop_after else n_generated % (VOCAB - 1)
        logits = torch.full((1, VOCAB), -5.0)
        logits[0, token] = 5.0
        return logits


def _stream(name: str, prompt_len: int) -> Stream:
    ids = torch.zeros((1, prompt_len), dtype=torch.long)
    return Stream(
        ids=ids,
        mask=torch.ones_like(ids),
        past=0,
        state={"name": name, "prompt_len": prompt_len},
    )


def _run(switch_k: int, stop_after: int) -> tuple[list[int], FakeBackbone]:
    backbone = FakeBackbone(stop_after)
    cfg = GuidanceConfig(alpha=1.5, feature="logits", switch_k=switch_k, guard="none")
    out = guided_decode(backbone, _stream("c", 4), _stream("n", 2), 20, cfg)
    return out, backbone


def test_negative_stream_is_skipped_before_switch_and_caught_up_one_token_at_a_time():
    out, backbone = _run(switch_k=3, stop_after=6)
    neg = [(length, first) for name, length, first in backbone.calls if name == "n"]
    # prefill, then nothing for tokens 1-2, then three single-token forwards at token 3,
    # then one forward per token (teacher forcing)
    assert neg == [
        (2, True),
        (3, False),
        (4, False),
        (5, False),
        (6, False),
        (7, False),
        (8, False),
    ]
    assert len(out) == 6


def test_switch_zero_is_teacher_forcing_from_the_start():
    _, backbone = _run(switch_k=0, stop_after=4)
    neg = [length for name, length, _ in backbone.calls if name == "n"]
    assert neg == [2, 3, 4, 5, 6]


def test_answer_shorter_than_switch_never_catches_up():
    _, backbone = _run(switch_k=8, stop_after=3)
    neg = [length for name, length, _ in backbone.calls if name == "n"]
    assert neg == [2]


def test_alpha_one_returns_the_conditional_in_both_forms():
    cond, ref = torch.randn(1, 8), torch.randn(1, 8)
    cfg = GuidanceConfig(alpha=1.0, feature="hidden")
    for form in ("reference", "conditional"):
        assert torch.allclose(blend(cond, ref, cfg, form, lambda h: h), cond)


def test_apc_keeps_only_plausible_tokens():
    lc = torch.log(torch.tensor([[0.6, 0.3, 0.05, 0.05]]))
    guided = torch.tensor([[0.0, 1.0, 9.0, 2.0]])
    masked = apc_mask(guided, lc, beta=0.1)
    assert int(masked.argmax()) == 1
    assert torch.isinf(masked[0, 2]) and torch.isinf(masked[0, 3])


def test_cached_reference_runs_the_negative_stream_once():
    backbone = FakeBackbone(stop_after=5)
    cfg = GuidanceConfig(feature="logits", reference="cached", guard="none")
    guided_decode(backbone, _stream("c", 4), _stream("n", 2), 20, cfg)
    assert [length for name, length, _ in backbone.calls if name == "n"] == [2]


def test_guide_steps_leaves_later_tokens_to_the_conditional():
    cfg = GuidanceConfig(
        feature="logits", reference="cached", guard="none", guide_steps=1
    )
    out = guided_decode(
        FakeBackbone(stop_after=4), _stream("c", 4), _stream("n", 2), 20, cfg
    )
    # the conditional alone emits 0, 1, 2, 3 and stops; only token 0 is guided
    assert out[1:] == [1, 2, 3]
