# ARGOS on vantage6

Federated deep learning for lung tumour (GTV) segmentation on CT, reimplemented as a
[vantage6](https://vantage6.ai) 4.15 algorithm. A central task averages model weights over
training rounds; each participating node trains locally on its own CT slices and returns its
weights and metrics.

## Functions

| Function | Runs on | Purpose |
| --- | --- | --- |
| `central(config, participating_ids)` | one node | Runs `num_rounds` rounds: sends the current weights to every participating organisation, averages what comes back, saves a checkpoint every `save_central_model_step` rounds, returns per-round metrics and, unless `return_final_weights` is false, the final averaged weights. |
| `train_locally(config, model_weights)` | every data node | Resolves the slice indexes for this machine, trains for `num_steps` steps from the given weights, returns `model weights` and `metrics`. |
| `test_gpu()` | any node | Prints what TensorFlow sees of the GPU. |

`config` is the content of [`config.json`](config.json): model and training parameters plus the
patient split per client (`client<i>_train_samples`, `client<i>_val_samples`). A node picks its
split with the `NODE_ID` environment variable (default `0`); with `central_eval: true` every
client's list is used.

## Data

Each node needs a folder in the *ARGOS layout*, reachable through the `DATA_PATH` environment
variable:

```
Train/<patient>/CT/<z>.nii.gz             one 512x512 slice, HU clipped to [-800, 200], scaled to [0, 1]
Train/<patient>/GT/GTV/<z>_gtv.nii.gz     GTV mask for that slice, 0/1
Train/<patient>/GT/Lung/<z>_lung.nii.gz   lung mask for that slice, 0/1
Validation/<patient>/...                  the same for validation patients
assets/slices_training_800200.json        {patient: ["<ct>,<gtv>,<lung>, <1 if GTV on slice else 0>", ...]}
assets/slices_validation_800200.json      paths relative to this folder
assets/params_base.json                   training parameters without the client split
```

Build the two index files with:

```sh
python -m argosfeddeep.sort_slices /path/to/argos_layout
```

Indexes made before this version hold absolute paths from the machine they were built on and
must be rebuilt once with the command above.

**The data folder is only read.** Everything ARGOS derives from it (resolved indexes,
`params.json`, checkpoints) is written to a work folder: `ARGOS_WORK_DIR` if set, else
`TEMPORARY_FOLDER`, which a vantage6 node sets to a per-run scratch volume, else the data
folder. The central part saves checkpoints under `ARGOS_CHECKPOINT_DIR`, default
`<work folder>/model_checkpoints`; since that is scratch space, the final averaged weights are
also part of the central result. Initial weights named by `config["init_weights"]` are read from
the data folder of the node that runs the central part.

## Running on a vantage6 node

The node configuration needs:

- a **folder database** whose host path holds the layout; the node mounts it at
  `/mnt/<label>` in the algorithm container, read-only is enough;
- `algorithm_env` with `DATA_PATH: /mnt/<label>` and, per node, `NODE_ID`;
- a GPU: `algorithm_device_requests: {gpu: true}` and the NVIDIA container toolkit on the host;
- the image on the allowlist.

## Development

```sh
uv sync
python -m unittest discover tests            # path handling and preprocessing, no TensorFlow needed
docker build -t argos .
```

`main.py`, `test.py` and `get_results.py` are client-side scripts and expect a `credentials.py`
next to them (ignored by git) with `server_url`, `server_port`, `server_api`, `usn`, `psswrd`.
