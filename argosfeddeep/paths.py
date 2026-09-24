"""Where ARGOS reads its data and where it writes the files it derives from it.

The data folder is read-only by intent: a vantage6 node may mount it read-only, and the same
folder is shared by every algorithm the node allows. Everything ARGOS produces (resolved slice
indexes, parameters, checkpoints, logs) goes to a separate work folder.

On a vantage6 node nothing has to be configured: the node mounts every database it is
configured with at ``/mnt/<label>`` and sets ``<LABEL>_DATABASE_URI`` to that path, and the
task names the database to use in ``USER_REQUESTED_DATABASE_LABELS``. The data folder is the
first database the task asked for, else the first the node has.

Environment variables, all optional:

``DATA_PATH``
    Overrides the dataset folder (``Train/``, ``Validation/``, ``assets/``), for running
    outside vantage6. Without it, and without a node's database variables, ``./argos_layout``.
``ARGOS_WORK_DIR``
    Where derived files are written. Defaults to ``TEMPORARY_FOLDER``, which the vantage6 node
    sets to a per-run scratch volume, and otherwise to the data folder itself.
``ARGOS_CHECKPOINT_DIR``
    Where the central part saves averaged weights. Defaults to ``<work dir>/model_checkpoints``.
"""

import os
from pathlib import Path

DEFAULT_DATA_DIR = "argos_layout"


def data_dir(override: str | os.PathLike | None = None) -> Path:
    """The dataset folder.

    In order: the argument if given; ``DATA_PATH``; the vantage6 database the task requested
    (or, failing that, the node's first database); ``./argos_layout``.
    """
    if override is not None:
        return Path(override)
    if os.environ.get("DATA_PATH"):
        return Path(os.environ["DATA_PATH"])
    from_node = vantage6_database_dir()
    if from_node is not None:
        return from_node
    return Path(DEFAULT_DATA_DIR)


def vantage6_database_dir() -> Path | None:
    """The folder of the database a vantage6 node exposes to this run, or None outside a node.

    The node sets ``USER_REQUESTED_DATABASE_LABELS`` (what the task asked for) and ``DB_LABELS``
    (everything it has), and one ``<LABEL>_DATABASE_URI`` per label with the path inside this
    container. Only labels whose URI is an existing directory qualify, so a node that also
    holds a CSV for another algorithm is not mistaken for the image folder.
    """
    for variable in ("USER_REQUESTED_DATABASE_LABELS", "DB_LABELS"):
        for label in filter(None, os.environ.get(variable, "").split(",")):
            uri = os.environ.get(f"{label.strip().upper()}_DATABASE_URI")
            if uri and os.path.isdir(uri):
                return Path(uri)
    return None


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

    Indexes written by ``sort_slices`` hold paths relative to the data folder. An absolute
    path is returned unchanged; an index that holds absolute paths from another machine has
    to be rebuilt with ``python -m argosfeddeep.sort_slices``.
    """
    text = path.strip()
    if os.path.isabs(text):
        return text
    return str(data / text)


def resolve_slice_line(line: str, data: Path) -> str:
    """Resolve the three paths of one ``ct,gtv,lung, label`` index line, keeping the label."""
    fields = line.split(",")
    resolved = [resolve_slice_path(field, data) for field in fields[:3]]
    return ",".join(resolved + [field.strip() and f" {field.strip()}" for field in fields[3:]])
