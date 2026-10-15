"""VCD: Visual Contrastive Decoding (Leng et al., CVPR 2024), contrasting a noised
image.

Port of the authors' decoding rule to Qwen2.5-VL, Qwen3-VL and InternVL3.5-HF; the noise
schedule follows the authors' code (https://github.com/DAMO-NLP-SG/VCD, Apache-2.0).
"""

# The published settings, used for every model in the paper.
CD_ALPHA, CD_BETA, NOISE_STEP = 1.0, 0.1, 500


def configure(model, cd_alpha: float, cd_beta: float, noise_step: int) -> None:
    """Set VCD's hyperparameters on a VCD model class (they are read in its
    `forward`)."""
    if not 0 <= int(noise_step) <= 999:
        raise ValueError(f"vcd_noise_step must be in [0, 999], got {noise_step}")
    model.cd_alpha = float(cd_alpha)
    model.cd_beta = float(cd_beta)
    model.noise_step = int(noise_step)
