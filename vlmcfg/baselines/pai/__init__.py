"""PAI: Paying more Attention to Image (Liu et al., ECCV 2024).

Amplify the last query row's attention to image tokens in the selected decoder layers
(A += alpha * |A|), and guide the logits away from the same prompt without the image
(CFG with scale gamma). Ported from the authors' code
(https://github.com/LALBJ/PAI, MIT) to Qwen2.5-VL, Qwen3-VL, InternVL3.5-HF and Molmo2.
"""

# The settings used for every model in the paper: layers 2 to the last, alpha 0.2,
# gamma 1.1.
LAYERS, ALPHA, GAMMA = "2-end", 0.2, 1.1


def parse_layers(spec: str, num_layers: int) -> tuple[int, int]:
    """`"[start,end]"` or `"start-end"` (`end` may be the word `end`) -> half-open
    bounds."""
    s = str(spec).strip()
    if s.startswith("[") and s.endswith("]"):
        parts = s[1:-1].split(",")
    elif "-" in s:
        parts = s.split("-", 1)
    else:
        raise ValueError(
            f"pai_layers must be '[start,end]' or 'start-end', got {spec!r}"
        )
    if len(parts) != 2:
        raise ValueError(f"pai_layers must have exactly two bounds, got {spec!r}")
    start = int(parts[0])
    end = num_layers if parts[1].strip() == "end" else int(parts[1])
    if not 0 <= start < end <= num_layers:
        raise ValueError(f"pai_layers {spec!r} is not within [0, {num_layers}]")
    return start, end
