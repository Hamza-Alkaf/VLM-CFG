"""M3ID on Molmo2: where the image ends in the prompt (M3ID's t0)."""

import torch

IMAGE_PATCH_ID = 151938


def tokens_after_image(input_ids: torch.Tensor) -> int:
    """Prompt tokens after the last image-patch token (M3ID's t0); -1 if there is no
    image.
    """
    pos = (input_ids[0] == IMAGE_PATCH_ID).nonzero()
    if pos.numel() == 0:
        return -1
    return int(input_ids.shape[1]) - (int(pos[-1]) + 1)
