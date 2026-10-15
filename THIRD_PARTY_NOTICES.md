# Third-party notices

Parts of this repository are adapted from the projects below and remain under their licenses.

| code here | adapted from | license |
|---|---|---|
| `vlmcfg/tasks/` (task definitions and scorers) | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | MIT |
| `vlmcfg/baselines/vcd/` (noise schedule, contrastive rule) | [VCD](https://github.com/DAMO-NLP-SG/VCD) | Apache-2.0 |
| `vlmcfg/baselines/pai/` (attention amplification, guidance) | [PAI](https://github.com/LALBJ/PAI) | MIT |
| `vlmcfg/baselines/layercd/` (contrastive rule) | [LayerCD](https://github.com/maifoundations/LayerCD) | MIT |

M3ID, SID and ACG are re-implemented from their papers and contain no upstream code.
Changes from the upstream sources: ported to the backbones in `vlmcfg/backbones/` and
wired into the lmms-eval plugin.
