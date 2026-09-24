"""Where ARGOS reads its data and where it writes the files it derives from it.

ARGOS runs inside a vantage6 algorithm container and nowhere else, so every location comes
from what the node sets and nothing has to be configured:

- The node mounts each of its databases at ``/mnt/<label>`` and sets ``<LABEL>_DATABASE_URI``
  to that path. The task names the database to use in ``USER_REQUESTED_DATABASE_LABELS``;
  ``DB_LABELS`` lists everything the node has. The data folder is the first requested label
  whose path is a directory, else the node's first directory database.
- ``TEMPORARY_FOLDER`` is a scratch volume the node creates for this run and discards after
  it. Everything ARGOS produces (resolved slice indexes, parameters, checkpoints) goes there,
  so the data folder is only read and can be mounted read-only.

To run the functions outside a node, set those three variables by hand or use vantage6's
``MockAlgorithmClient``, which sets them the way the node does.
"""

import os
from pathlib import Path


def data_dir() -> Path:
    """The dataset folder (``Train/``, ``Validation/``, ``assets/``) the node mounted."""
    seen = []
    for variable in ("USER_REQUESTED_DATABASE_LABELS", "DB_LABELS"):
        for label in filter(None, os.environ.get(variable, "").split(",")):
            uri = os.environ.get(f"{label.strip().upper()}_DATABASE_URI")
            seen.append(f"{label.strip()}={uri}")
            if uri and os.path.isdir(uri):
                return Path(uri)
    raise RuntimeError(
        "no folder database found; the node must be configured with a database of type "
        f"folder holding the ARGOS layout (databases seen: {', '.join(seen) or 'none'})"
    )


def work_dir() -> Path:
    """The per-run scratch folder for derived files, with an ``assets`` subfolder."""
    root = Path(os.environ["TEMPORARY_FOLDER"])
    (root / "assets").mkdir(parents=True, exist_ok=True)
    return root


def checkpoint_dir() -> Path:
    """Where the central part saves averaged weights: scratch space on the node running it."""
    root = work_dir() / "model_checkpoints"
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
