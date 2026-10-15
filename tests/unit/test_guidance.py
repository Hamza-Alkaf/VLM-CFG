"""CPU tests of the guidance arithmetic, guards, rules and configs."""

import pytest
import torch

from vlmcfg.guidance import (
    GuidanceConfig,
    apg_project,
    blend,
    eos_guard,
    rule_blend,
)
from vlmcfg.negatives import NegativeConfig, RegionConfig


def test_eos_guard_stops_exactly_when_the_conditional_does():
    eos = frozenset({3})
    guided = torch.tensor([[0.0, 5.0, 1.0, 4.0]])
    stop = eos_guard(
        guided, torch.log_softmax(torch.tensor([[0.0, 1.0, 0.0, 9.0]]), -1), eos
    )
    assert int(stop.argmax()) == 3 and torch.isinf(stop[0, :3]).all()
    go = eos_guard(
        torch.tensor([[0.0, 1.0, 0.0, 9.0]]),
        torch.log_softmax(torch.tensor([[0.0, 5.0, 0.0, 1.0]]), -1),
        eos,
    )
    assert int(go.argmax()) == 1 and torch.isinf(go[0, 3])


def test_logit_blend_against_a_narrower_vocabulary_pads_unreachable_ids():
    cfg = GuidanceConfig(feature="logits", guard="none")
    cond, ref = torch.randn(1, 8), torch.randn(1, 6)
    out = blend(cond, ref, cfg, "reference", lambda h: h)
    assert out.shape == (1, 8)
    assert torch.equal(out[0, :6], ref[0] + 1.5 * (cond[0, :6] - ref[0]))
    assert (out[0, 6:] == -1e4).all()


def test_apg_with_eta_one_is_the_plain_delta():
    d, c = torch.randn(1, 5), torch.randn(1, 5)
    assert torch.equal(apg_project(d, c, 1.0), d)


def test_cap_rule_at_step_zero_is_the_cached_blend():
    cfg = GuidanceConfig(switch_k=0, rule="cap", guard="none")
    c, u = torch.randn(1, 4), torch.randn(1, 4)
    out = rule_blend(c, u, 0, {}, cfg, lambda h: h)
    assert torch.equal(out, u + 1.5 * (c - u))


@pytest.mark.parametrize(
    "bad",
    [
        {"rule": "decay"},  # rules need switch_k=0
        {"rule": "cap", "switch_k": "0", "feature": "logits"},
        {"operator": "tcfg", "feature": "logits"},
        {"guide_steps": "1"},  # needs reference=cached
        {"reference": "window", "window": "-1"},
    ],
)
def test_invalid_guidance_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        GuidanceConfig.from_args(bad)


def test_configs_split_their_own_arguments_and_parse_booleans():
    cfg, rest = RegionConfig.from_args(
        {
            "region": "seg",
            "region_field": "segmentation",
            "random_region": "False",
            "x": 1,
        }
    )
    assert cfg.random_region is False and rest == {"x": 1}
    neg, rest = NegativeConfig.from_args({"negative": "early_exit", "exit_layer": "10"})
    assert neg.exit_layer == 10 and rest == {}
    with pytest.raises(ValueError):
        NegativeConfig.from_args({"negative": "weak_model"})
