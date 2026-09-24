"""Build the slice indexes for a dataset in the ARGOS layout.

Usage::

    python -m argosfeddeep.sort_slices /path/to/argos_layout

Writes ``assets/slices_training_800200.json`` and ``assets/slices_validation_800200.json``
into the dataset folder. Every path in them is relative to that folder, so the dataset can be
moved or mounted anywhere; ``preprocess`` resolves them at run time.
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def sort_slices(path, name, relative_to=None):
    """Index every slice under ``path`` and write ``{patient: [lines]}`` to ``name``.

    Each line is ``<ct>,<gtv>,<lung>, <1 if the slice contains GTV else 0>``. Paths are made
    relative to ``relative_to`` when given, else left as found.
    """
    slice_dict = {}
    for patient in sorted(os.listdir(path)):
        patient_path = os.path.join(path, patient)
        ct_path = os.path.join(patient_path, 'CT')
        gt_path = os.path.join(patient_path, 'GT')
        gt_lung_path = os.path.join(gt_path, 'Lung')
        gt_gtv_path = os.path.join(gt_path, 'GTV')

        gt_slices = []
        for layer in range(0, len(os.listdir(ct_path))):
            gt_patch_gtv = nib.load(os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz')).get_fdata()
            label = 1 if np.max(gt_patch_gtv) == 1 else 0
            ct = os.path.join(ct_path, str(layer) + '.nii.gz')
            gtv = os.path.join(gt_gtv_path, str(layer) + '_gtv.nii.gz')
            lung = os.path.join(gt_lung_path, str(layer) + '_lung.nii.gz')
            if relative_to is not None:
                ct, gtv, lung = (os.path.relpath(p, relative_to) for p in (ct, gtv, lung))
            gt_slices.append(f"{ct},{gtv},{lung}, {label}")
        slice_dict[patient] = gt_slices

    Path(name).parent.mkdir(parents=True, exist_ok=True)
    with open(name, 'w') as fp:
        json.dump(slice_dict, fp)
    return slice_dict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("layout", type=Path, help="Dataset folder with Train/, Validation/ and assets/")
    args = parser.parse_args(argv)
    layout = args.layout.resolve()
    sort_slices(layout / "Train", layout / "assets" / "slices_training_800200.json", layout)
    sort_slices(layout / "Validation", layout / "assets" / "slices_validation_800200.json", layout)


if __name__ == "__main__":
    main()
