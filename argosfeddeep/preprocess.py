"""Resolve the slice indexes for this machine and derive the per-client sample split.

Reads ``assets/slices_training_800200.json``, ``assets/slices_validation_800200.json`` and
``assets/params_base.json`` from the data folder, and writes ``slices_*_modified.json`` and
``params.json`` to the work folder. The data folder is never written to.
"""

import json
import math
from pathlib import Path

import numpy as np

from argosfeddeep.paths import data_dir, resolve_slice_line, work_dir


def update_slice_config(config_path, new_config_path, current_path) -> dict:
    """Resolve every path in a slice index against ``current_path`` and write the result."""
    with open(config_path, "r") as fp:
        train_slices_config = json.load(fp)

    data = Path(current_path)
    new_config = {
        key: [resolve_slice_line(line, data) for line in lines]
        for key, lines in train_slices_config.items()
    }

    Path(new_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(new_config_path, "w") as fp:
        json.dump(new_config, fp)

    return new_config


def divide_slices(slice_names, return_key):
    """Split patient names over three clients, reproducibly."""
    rng = np.random.default_rng(seed=7)
    samples_per_client = math.floor(len(slice_names) / 3)
    divided_samples = rng.choice(np.arange(len(slice_names)), size=(samples_per_client, 2), replace=False)
    last_client_idx = [i not in divided_samples for i in range(len(slice_names))]
    return {
        f'client0_{return_key}': list(slice_names[divided_samples[:, 0]]),
        f'client1_{return_key}': list(slice_names[divided_samples[:, 1]]),
        f'client2_{return_key}': list(slice_names[last_client_idx]),
    }


def preprocess() -> Path:
    """Write the resolved indexes and parameters for the node's data folder.

    Returns the work folder the files were written to.
    """
    data = data_dir()
    work = work_dir()

    train_base_path = data / "assets" / "slices_training_800200.json"
    train_updated_path = work / "assets" / "slices_training_modified.json"
    val_base_path = data / "assets" / "slices_validation_800200.json"
    val_updated_path = work / "assets" / "slices_validation_modified.json"

    train_slices_config = update_slice_config(train_base_path, train_updated_path, data)
    val_slices_config = update_slice_config(val_base_path, val_updated_path, data)

    with open(data / "assets" / "params_base.json", "r") as fp:
        parameters = json.load(fp)

    slice_names_train = np.array(list(train_slices_config.keys()))
    slice_names_val = np.array(list(val_slices_config.keys()))

    parameters.update(divide_slices(slice_names_train, "train_samples"))
    parameters.update(divide_slices(slice_names_val, "val_samples"))

    with open(work / "assets" / "params.json", "w") as fp:
        json.dump(parameters, fp)

    return work
