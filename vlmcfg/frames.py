"""The frame an image reached the model in, published by models and read by box scorers.

Qwen-VL answers boxes in the RESIZED frame (sides rounded to the patch grid), so a
scorer that divides by the original image size mis-scales x and y. Models record the
frame here when they build inputs; grounding tasks read it back when scoring.
"""

_FRAMES: dict[tuple[int, int], tuple[int, int]] = {}


def record_visual_input_frame(
    source_size: tuple[int, int], input_size: tuple[int, int]
) -> None:
    """Publish that a `source_size` image reached the model as `input_size` (w, h)."""
    _FRAMES[tuple(source_size)] = tuple(input_size)


def get_visual_input_frame(source_size: tuple[int, int]) -> tuple[int, int] | None:
    """The frame a `source_size` image reached the model in, if a model recorded one."""
    return _FRAMES.get(tuple(source_size))
