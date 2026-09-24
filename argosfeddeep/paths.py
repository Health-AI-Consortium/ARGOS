"""Where ARGOS reads its data and where it writes the files it derives from it.

The data folder is read-only by intent: a vantage6 node may mount it read-only, and the same
folder is shared by every algorithm the node allows. Everything ARGOS produces (resolved slice
indexes, parameters, checkpoints, logs) goes to a separate work folder.

Environment variables, all optional:

``DATA_PATH``
    The dataset folder (``Train/``, ``Validation/``, ``assets/``). On a vantage6 node this is
    the folder database the node mounts, for example ``/mnt/argos_lung1``.
``ARGOS_WORK_DIR``
    Where derived files are written. Defaults to ``TEMPORARY_FOLDER``, which the vantage6 node
    sets to a per-run scratch volume, and otherwise to the data folder itself.
``ARGOS_CHECKPOINT_DIR``
    Where the central part saves averaged weights. Defaults to ``<work dir>/model_checkpoints``.
"""

import os
from pathlib import Path

AUTHOR_PREFIX = "/Users/palan001/Desktop/Health-AI/ARGOS/data/nsclc_radiomics_v4/argos_layout"
"""Absolute prefix found in slice indexes made before the indexes became relative."""

DEFAULT_DATA_DIR = "argos_layout"


def data_dir(override: str | os.PathLike | None = None) -> Path:
    """The dataset folder: the argument if given, else ``DATA_PATH``, else ``./argos_layout``."""
    if override is not None:
        return Path(override)
    return Path(os.environ.get("DATA_PATH", DEFAULT_DATA_DIR))


def work_dir() -> Path:
    """The folder for derived files, created on first use with an ``assets`` subfolder."""
    chosen = os.environ.get("ARGOS_WORK_DIR") or os.environ.get("TEMPORARY_FOLDER")
    root = Path(chosen) if chosen else data_dir()
    (root / "assets").mkdir(parents=True, exist_ok=True)
    return root


def checkpoint_dir() -> Path:
    """Where averaged weights are saved during central training."""
    chosen = os.environ.get("ARGOS_CHECKPOINT_DIR")
    root = Path(chosen) if chosen else work_dir() / "model_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_slice_path(path: str, data: Path) -> str:
    """Turn one slice path from an index into an absolute path under the data folder.

    Indexes written by ``sort_slices`` hold paths relative to the data folder. Older indexes
    hold absolute paths from the machine they were made on; those are re-rooted when they
    start with the known prefix. Any other absolute path is returned unchanged.
    """
    text = path.strip()
    if text.startswith(AUTHOR_PREFIX):
        return str(data / text[len(AUTHOR_PREFIX):].lstrip("/"))
    if not os.path.isabs(text):
        return str(data / text)
    return text


def resolve_slice_line(line: str, data: Path) -> str:
    """Resolve the three paths of one ``ct,gtv,lung, label`` index line, keeping the label."""
    fields = line.split(",")
    resolved = [resolve_slice_path(field, data) for field in fields[:3]]
    return ",".join(resolved + [field.strip() and f" {field.strip()}" for field in fields[3:]])
