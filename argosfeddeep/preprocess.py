import json
from pathlib import Path
import numpy as np
import math

# print(f'current path: {current_path}')
# print(f'new path: {current_path}/argos_layout/')


def update_slice_config(config_path: str, new_config_path: str, current_path: str)-> dict:
    with open(config_path, "r") as fp:
        train_slices_config = json.load(fp)

    # change paths to have the correct path through all user-specific folders ###
    to_replace = "/Users/palan001/Desktop/Health-AI/ARGOS/data/nsclc_radiomics_v4/argos_layout"

    new_config = {}
    for key in train_slices_config.keys():
        new_config[key] = []
        for path in train_slices_config[key]:
            new_path = path.replace(to_replace, str(current_path) + "/")
            new_config[key].append(new_path)

    with open(new_config_path, "w") as fp:
        json.dump(new_config, fp)

    return new_config

def divide_slices(slice_names, return_key):
    rng = np.random.default_rng(seed=7)
    samples_per_client = math.floor(len(slice_names)/3)
    divided_samples = rng.choice(np.arange(len(slice_names)), size = (samples_per_client, 2), replace = False)
    last_client_idx = [i not in divided_samples for i in range(len(slice_names))]
    return {
        f'client0_{return_key}' : list(slice_names[divided_samples[:,0]]),
        f'client1_{return_key}' : list(slice_names[divided_samples[:,1]]),
        f'client2_{return_key}' : list(slice_names[last_client_idx])
    }


def preprocess(updated_path: str = None):

    if updated_path is not None:
        current_path = updated_path
    else:
        current_path = Path(__file__).parent

    # print(f'current_path: {current_path}')


    train_base_path = f'{current_path}/assets/slices_training_800200.json'
    train_updated_path = f'{current_path}/assets/slices_training_modified.json'

    val_base_path = f'{current_path}/assets/slices_validation_800200.json'
    val_updated_path = f'{current_path}/assets/slices_validation_modified.json'

    train_slices_config = update_slice_config(train_base_path, train_updated_path, current_path)
    val_slices_config = update_slice_config(val_base_path, val_updated_path, current_path)


    with open(f'{current_path}/assets/params_base.json', "r") as fp:
        parameters = json.load(fp)




    slice_names_train = np.array(list(train_slices_config.keys()))
    slice_names_val = np.array(list(val_slices_config.keys()))

    parameters.update(divide_slices(slice_names_train, "train_samples"))
    parameters.update(divide_slices(slice_names_val, "val_samples"))


    with open(f'{current_path}/assets/params.json', "w") as fp:
        json.dump(parameters, fp)

