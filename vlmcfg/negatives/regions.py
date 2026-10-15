"""Annotated regions: the localized negative and the region split of an attention edit.

`black_out_region` blacks the image out inside a mask (the localized negative);
`region_pixel_mask` paints an annotated box or polygon at pixel resolution,
`region_token_mask` at the image-token grid, and `random_region` is the size-matched
control.
"""

import random
from collections.abc import Sequence

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter


def black_out_region(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """`image` with the pixels under `mask` (H x W in [0, 1], 1 = black) blacked out."""
    image = image.convert("RGB")
    width, height = image.size
    if mask.shape != (height, width):
        raise ValueError(
            f"mask shape {mask.shape} does not match image {(height, width)}"
        )
    black = Image.new("RGB", image.size, color=(0, 0, 0))
    alpha = np.clip(mask, 0.0, 1.0).astype(np.float32)[:, :, np.newaxis]
    img_np = np.array(image).astype(np.float32)
    aug_np = np.array(black).astype(np.float32)
    blended = np.clip(img_np * (1.0 - alpha) + aug_np * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# -------------------------------------------------------------------- annotated regions
def random_region(kind: str, seed: int, doc_id) -> list[float]:
    """A random rectangle covering 10-50% of each side, keyed by (seed, doc id).

    Returned as a normalised box ("bbox") or the same rectangle as a polygon ("seg").
    """
    rng = random.Random(f"{seed}-{doc_id}")
    w = rng.uniform(0.10, 0.50)
    h = rng.uniform(0.10, 0.50)
    x0 = rng.uniform(0.0, 1.0 - w)
    y0 = rng.uniform(0.0, 1.0 - h)
    if kind == "bbox":
        return [x0, y0, x0 + w, y0 + h]
    return [x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]


def _finish_mask(
    mask: np.ndarray, dilation_iters: int, feather_sigma: float
) -> np.ndarray:
    if dilation_iters >= 1:
        mask = binary_dilation(mask > 0.5, iterations=dilation_iters).astype(np.float32)
    if feather_sigma > 0:
        mask = np.clip(gaussian_filter(mask, sigma=feather_sigma), 0.0, 1.0)
    return mask


def _looks_normalized(coords: Sequence[float]) -> bool:
    """True when every coordinate is in [0, 1], i.e. image-relative."""
    return len(coords) > 0 and all(0.0 <= float(v) <= 1.0 for v in coords)


_BOX_DICT_KEYS = (
    (("x0", "y0", "x1", "y1"), False),
    (("xmin", "ymin", "xmax", "ymax"), False),
    (("left", "top", "right", "bottom"), False),
    (("x", "y", "w", "h"), True),
    (("x", "y", "width", "height"), True),
)


def _box_from_dict(box: dict) -> list[float]:
    for keys, is_wh in _BOX_DICT_KEYS:
        if all(k in box for k in keys):
            a, b, c, d = (float(box[k]) for k in keys)
            return [a, b, a + c, b + d] if is_wh else [a, b, c, d]
    raise ValueError(f"unrecognised box keys {sorted(box)}")


def _as_boxes(bboxs) -> list[list[float]]:
    """One box, a list of boxes, or a list of box dicts, as [x0, y0, x1, y1] boxes."""
    if isinstance(bboxs, dict):
        return [_box_from_dict(bboxs)]
    items = list(bboxs)
    if not items:
        return []
    if isinstance(items[0], dict):
        return [_box_from_dict(box) for box in items]
    if isinstance(items[0], (list, tuple, np.ndarray)):
        return [[float(v) for v in box] for box in items]
    if len(items) != 4:
        raise ValueError(f"a flat box needs 4 coordinates, got {len(items)}")
    return [[float(v) for v in items]]


def _as_polygons(segmentation) -> list[list[float]]:
    """A flat polygon, a list of them, or a list of {x, y} vertices,
    as flat polygons."""
    items = list(segmentation)
    if not items:
        return []
    if isinstance(items[0], dict):
        return [[float(c) for pt in items for c in (pt["x"], pt["y"])]]
    if isinstance(items[0], (list, tuple, np.ndarray)):
        return [[float(v) for v in part] for part in items]
    return [[float(v) for v in items]]


def bbox_mask(
    image_size: tuple[int, int],
    bboxs,
    source_size: tuple[int, int] | None = None,
    dilation_iters: int = 1,
    feather_sigma: float = 0.0,
) -> np.ndarray:
    """(H, W) float mask, 1 inside the boxes
    (pixels of `source_size`, or normalised)."""
    width, height = image_size
    src_w, src_h = source_size if source_size is not None else image_size
    scale_x, scale_y = width / src_w, height / src_h
    boxes = _as_boxes(bboxs)
    flat = [v for box in boxes for v in box]
    if min(width, height) > 1 and _looks_normalized(flat):
        boxes = [
            [b[0] * src_w, b[1] * src_h, b[2] * src_w, b[3] * src_h] for b in boxes
        ]
    mask = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x0, y0, x1, y1 = (float(v) for v in box)
        cols = sorted((int(round(x0 * scale_x)), int(round(x1 * scale_x))))
        rows = sorted((int(round(y0 * scale_y)), int(round(y1 * scale_y))))
        col0, col1 = max(0, cols[0]), min(width, max(cols[1], cols[0] + 1))
        row0, row1 = max(0, rows[0]), min(height, max(rows[1], rows[0] + 1))
        if col0 < col1 and row0 < row1:
            mask[row0:row1, col0:col1] = 1.0
    if not mask.any():
        logger.warning("no usable bounding box; the region is empty")
    return _finish_mask(mask, dilation_iters, feather_sigma)


def seg_mask(
    image_size: tuple[int, int],
    segmentation,
    source_size: tuple[int, int] | None = None,
    dilation_iters: int = 1,
    feather_sigma: float = 0.0,
) -> np.ndarray:
    """(H, W) float mask, 1 inside the polygons (pixels of `source_size`, or normalised)
    or inside a COCO RLE dict."""
    width, height = image_size
    if isinstance(segmentation, dict):
        return _rle_mask(segmentation, width, height, dilation_iters, feather_sigma)
    src_w, src_h = source_size if source_size is not None else image_size
    scale_x, scale_y = width / src_w, height / src_h
    polygons = _as_polygons(segmentation)
    if not polygons:
        logger.warning("no usable segmentation; the region is empty")
        return np.zeros((height, width), dtype=np.float32)
    for part in polygons:
        if len(part) % 2 != 0:
            raise ValueError(
                f"a polygon needs interleaved x, y; got {len(part)} values"
            )
    flat = [v for part in polygons for v in part]
    if min(width, height) > 1 and _looks_normalized(flat):
        polygons = [
            [v * (src_w if i % 2 == 0 else src_h) for i, v in enumerate(part)]
            for part in polygons
        ]
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for part in polygons:
        pts = [
            (part[i] * scale_x, part[i + 1] * scale_y) for i in range(0, len(part), 2)
        ]
        if len(pts) >= 3:
            drawer.polygon(pts, fill=255)
    mask = (np.array(canvas) > 127).astype(np.float32)
    if not mask.any():
        logger.warning("no usable segmentation; the region is empty")
    return _finish_mask(mask, dilation_iters, feather_sigma)


def _rle_mask(
    rle: dict, width: int, height: int, dilation_iters: int, feather_sigma: float
) -> np.ndarray:
    """A COCO RLE annotation, decoded at its own resolution and resampled (nearest)."""
    from pycocotools import mask as mask_util

    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    decoded = mask_util.decode(rle)
    if decoded.ndim == 3:  # one channel per instance
        decoded = decoded.max(axis=2)
    if not decoded.any():
        logger.warning("no usable segmentation; the region is empty")
        return np.zeros((height, width), dtype=np.float32)
    mask = decoded.astype(bool)
    if mask.shape != (height, width):
        resized = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
            (width, height), resample=Image.NEAREST
        )
        mask = np.array(resized) > 127
    return _finish_mask(mask.astype(np.float32), dilation_iters, feather_sigma)


def region_pixel_mask(
    kind: str,
    value,
    image_size: tuple[int, int],
    source_size: tuple[int, int],
    **finish,
) -> np.ndarray:
    """A box ("bbox") or polygon ("seg") annotation painted at pixel resolution."""
    if kind == "bbox":
        return bbox_mask(image_size, value, source_size=source_size, **finish)
    if kind == "seg":
        return seg_mask(image_size, value, source_size=source_size, **finish)
    raise ValueError(f"region kind must be 'bbox' or 'seg', got {kind!r}")


def region_token_mask(kind: str, value, image_grid_thw, source_size: tuple[int, int]):
    """Bool [h*w] over the post-merge image tokens (row-major): True inside the region.

    The annotation is painted directly at token-grid resolution, hard-edged.
    """
    import torch

    h, w = int(image_grid_thw[1]) // 2, int(image_grid_thw[2]) // 2
    grid = region_pixel_mask(
        kind, value, (w, h), source_size, dilation_iters=0, feather_sigma=0.0
    )
    return torch.from_numpy(grid.flatten() > 0.5)
