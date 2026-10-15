# VLM-CFG

Classifier-free guidance for vision-language models: decode each answer with two streams
through the same model, one that sees the image and one that does not, and push every token
towards the evidence the image carries. No training, no per-benchmark tuning, about 1.04x the
cost of plain greedy decoding.

This repository is the code of *Contrastive Decoding Is Classifier-Free Guidance: Unifying
Visual Guidance for Vision-Language Models*, packaged as a plugin for
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## The method

At every step the conditional stream reads the image and the reference stream reads the same
prompt with the image deleted. Guidance blends their last hidden states and reads the result
out through the language-model head:

```
h_guided = h_ref + alpha * (h_cond - h_ref)        logits = lm_head(h_guided)
```

The reference stands on no answer prefix for the first `K` tokens (its step-0 state is
cached), then catches up and follows the answer (teacher forcing). An adaptive plausibility
constraint keeps only tokens with `p_cond >= beta * max p_cond`. The defaults are the paper's
configuration and are used unchanged for every model and benchmark:

| setting | value |
|---|---|
| negative stream | image deleted |
| blend | last hidden state |
| `alpha` | 1.5 |
| switch point `K` | 8 |
| plausibility `beta` | 0.1 |

## Installation

Requires Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Hamza-Alkaf/VLM-CFG.git vlm-cfg && cd vlm-cfg
bash scripts/setup_env.sh          # .venv from uv.lock, NLTK data, .env
source .venv/bin/activate
```

The environment is pinned (Python 3.11, torch 2.7.1 with CUDA 11.8, transformers 4.57.1,
lmms-eval at commit `654e1b0`). Guidance compares two forward passes in bf16/fp16, so other
versions can flip tokens whose top two logits are within rounding; keep the pins to reproduce
the paper exactly.

### Keys

`setup_env.sh` creates `.env` from `.env.example`. Put your keys there; `vlmcfg-eval` and
`prepare_data.sh` read it, and `.env` is gitignored, so never commit a copy of it.

| variable | purpose | when you need it |
|---|---|---|
| `HF_TOKEN` | [Hugging Face access token](https://huggingface.co/settings/tokens) (read access), used for model and dataset downloads | recommended: anonymous downloads are rate-limited, which `prepare_data.sh` hits when it fetches many files; required for gated models or datasets |
| `WANDB_API_KEY` | [Weights & Biases key](https://wandb.ai/authorize), for logging runs | only with lmms-eval's `--wandb_args` |

No other key is needed: every benchmark is scored locally, and no task calls an API judge.

## Usage

`vlmcfg-eval` is the lmms-eval command line with this plugin's models and tasks registered:

```bash
# VLM-CFG on Qwen2.5-VL-7B (the default checkpoint), paper settings
vlmcfg-eval --model qwen2_5_vl_vlmcfg --tasks vstar_bench_grounding \
    --batch_size 1 --log_samples --output_path outputs/

# the unguided baseline on the same model
vlmcfg-eval --model qwen2_5_vl_base --tasks vstar_bench_grounding \
    --batch_size 1 --log_samples --output_path outputs/

# another checkpoint, or a different guidance scale
vlmcfg-eval --model qwen3_vl_vlmcfg --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,alpha=2.0 \
    --tasks mmstar --batch_size 1 --output_path outputs/
```

| model family | VLM-CFG | unguided baseline | default checkpoint | dtype |
|---|---|---|---|---|
| Qwen2.5-VL | `qwen2_5_vl_vlmcfg` | `qwen2_5_vl_base` | Qwen/Qwen2.5-VL-7B-Instruct | bf16 |
| Qwen3-VL | `qwen3_vl_vlmcfg` | `qwen3_vl_base` | Qwen/Qwen3-VL-8B-Instruct | fp16, images capped at 1.6 MP |
| InternVL3.5 | `internvl3_5_hf_vlmcfg` | `internvl3_5_hf_base` | OpenGVLab/InternVL3_5-8B-HF | bf16 |
| Molmo2 | `molmo2_vlmcfg` | `molmo2_base` | allenai/Molmo2-4B | fp16 |

Guidance settings are `--model_args`: `alpha`, `switch_k` (0 = teacher forcing throughout),
`feature` (`hidden` or `logits`), `guard` (`apc` or `none`), `apc_beta`. Any other key goes
to the model loader (e.g. `pretrained`, `dtype`, `max_pixels`).

**GPUs without bf16 tensor cores (e.g. V100)** run bf16 by emulation, several times slower;
the paper ran Qwen3-VL and Molmo2 in fp16 for that reason. Changing a model's dtype changes
its outputs, so compare runs only at the same dtype.

## Benchmarks

The ten benchmarks of the main table ship as plugin tasks: `ok_vqa_val2014_lite`,
`vizwiz_grounding_val`, `o3`, `o3_box`, `phd_sub10_ccs`, `mmstar`, `treebench`,
`blink_iq_test`, `vstar_bench_grounding`, `mmerealworld_lite`. Those hosted on the
Hugging Face Hub download on first use. The others are built from their public sources:

```bash
bash scripts/prepare_data.sh          # all of them, or name some: ... treebench openimages
```

This writes VizWiz grounding, PhD, TreeBench, V\*, OpenImages, RefChartQA, ReasonSeg,
MathVerse-MCQ, OCRBench-v2-2k, HallusionBench, AMBER and DetailCaps-500 into `data/build/`
(set `VLMCFG_DATA_DIR` to put them elsewhere). Each builder reproduces, value for value,
the documents the paper's runs read.

The analysis sections also use `ocrbench_v2_2k`, `mathverse_mcq_<version>_{direct,cot}`,
`reasonseg_grounding_{val,test}`, `refchartqa`, `openimages`, `phd_sub10_ccs_rationale`,
`hallusionbench_visual_vs`, and the long-form `amber_generative` and `detailcaps_500`.
Scoring the long-form two needs extra packages (spaCy, CAPTURE):
`WITH_LONGFORM=1 bash scripts/setup_env.sh`.

## Baselines

The contrastive-decoding baselines are ports run through the same harness, one model per
backbone: `<family>_vcd`, `<family>_m3id`, `<family>_pai`, `<family>_sid`, plus
`<family>_layercd` and `<family>_acg` (e.g. `qwen2_5_vl_pai`). Their settings default to
the ones the paper used, set at the top of `vlmcfg/baselines/<method>/__init__.py`. Molmo2
has VCD, M3ID, PAI and SID.

## Ablations

Every ablation in the paper is a set of `--model_args` on `qwen2_5_vl_vlmcfg`: where the
reference stands, the guidance operator, the guard, the blended feature, and weaker
negatives (attention edits, early exit, a smaller model, region black-outs).
[docs/ablations.md](docs/ablations.md) maps each table row to its arguments.

## Contributors

- Hamza Alkaf
- Eslam Bakr

## License

MIT, see [LICENSE](LICENSE). Code adapted from other projects keeps its license, see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
