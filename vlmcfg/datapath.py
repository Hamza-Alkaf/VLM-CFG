"""The `!datapath` YAML tag:
task files built by `data/prepare_*.py`, under VLMCFG_DATA_DIR.

A task config writes `test: !datapath vstar_bench_grounding/vstar.parquet` and gets the
absolute path. VLMCFG_DATA_DIR defaults to `data/build` in the repository; a relative
value is taken relative to the repository root.
"""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "build"


def data_dir() -> Path:
    """Directory holding the locally built benchmark files."""
    path = Path(os.environ.get("VLMCFG_DATA_DIR") or DEFAULT_DATA_DIR)
    return path if path.is_absolute() else REPO_ROOT / path


def _construct(loader: yaml.Loader, node: yaml.Node) -> str:
    return str(data_dir() / loader.construct_scalar(node))


def register() -> None:
    """Teach PyYAML's loaders (including the one lmms-eval uses) the `!datapath` tag."""
    yaml.add_constructor("!datapath", _construct)
