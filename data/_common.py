"""Shared helpers of the `prepare_*.py` builders:
where files go and how sources are fetched.

Built files land under VLMCFG_DATA_DIR (default `data/build`), at the paths the task
YAMLs read through `!datapath`. Downloads are cached under VLMCFG_SOURCE_DIR (default
`data/build/.sources`), so a rebuild does not fetch them again.
"""

import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmcfg.datapath import data_dir  # noqa: E402


def output(relative: str) -> Path:
    """Absolute path of a built file, its directory created."""
    path = data_dir() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sources() -> Path:
    """Directory caching the downloaded sources."""
    path = Path(os.environ.get("VLMCFG_SOURCE_DIR") or data_dir() / ".sources")
    path.mkdir(parents=True, exist_ok=True)
    return path


def hub_file(repo: str, filename: str, revision: str | None = None) -> Path:
    """One file of a Hugging Face dataset repository, downloaded once."""
    from huggingface_hub import hf_hub_download

    local = sources() / repo.replace("/", "__")
    return Path(
        hf_hub_download(
            repo, filename, repo_type="dataset", revision=revision, local_dir=local
        )
    )


def hub_snapshot(
    repo: str, allow_patterns: list[str] | None = None, revision: str | None = None
) -> Path:
    """A Hugging Face dataset repository (or the files matching `allow_patterns`)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    local = sources() / repo.replace("/", "__")
    for attempt in range(6):
        try:
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=revision,
                local_dir=local,
                allow_patterns=allow_patterns,
                max_workers=4,
            )
            return local
        except HfHubHTTPError as err:  # rate limited on many small files: wait, resume
            if attempt == 5 or "429" not in str(err):
                raise
            time.sleep(60 * (attempt + 1))
    return local


def url_file(url: str, name: str | None = None) -> Path:
    """A file downloaded from `url` into the source cache, once."""
    dest = sources() / (name or url.rsplit("/", 1)[-1])
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp.rename(dest)
    return dest
