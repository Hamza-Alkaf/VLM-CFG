"""`vlmcfg-eval`: the lmms-eval command line with this plugin's models and tasks.

Takes exactly the arguments of `python -m lmms_eval`, e.g.

    vlmcfg-eval --model qwen2_5_vl_vlmcfg --tasks blink_iq_test --batch_size 1
"""

import os
import sys
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def main() -> None:
    """Register the plugin, then hand over to lmms-eval's own CLI."""
    from dotenv import load_dotenv

    # tokens, cache and data locations (scripts/setup_env.sh); the shell's values win
    load_dotenv(ENV_FILE, override=False)
    plugins = [p for p in os.environ.get("LMMS_EVAL_PLUGINS", "").split(",") if p]
    if "vlmcfg" not in plugins:
        plugins.append("vlmcfg")
    # Must be set before lmms_eval.models is imported: models register at import time.
    os.environ["LMMS_EVAL_PLUGINS"] = ",".join(plugins)
    # lmms-eval builds its task index before it reads a plugin's task directory, so the
    # plugin's tasks are only found when passed as --include_path.
    if "--include_path" not in sys.argv:
        sys.argv[1:1] = ["--include_path", str(TASKS_DIR)]

    from vlmcfg.datapath import register

    register()
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
