"""Qwen-VL vision marker tokens and a token-pattern search (from the CFG harness)."""

import torch

VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653


def find_pattern_starts(input_ids: torch.Tensor, pattern: list[int]) -> list[list[int]]:
    """Return start indices of each pattern match for every sample in the batch."""
    B, L = input_ids.shape
    P = len(pattern)
    pat = torch.tensor(pattern, device=input_ids.device)
    starts_per_b = []
    for b in range(B):
        seq = input_ids[b]
        starts = []
        # naive scan; fine for moderate lengths
        for i in range(L - P + 1):
            if torch.equal(seq[i : i + P], pat):
                starts.append(i)
        starts_per_b.append(starts)
    return starts_per_b
