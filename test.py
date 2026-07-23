from argosfeddeep.run_online import dice_bce
from argosfeddeep.models import mod_resnet
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from vantage6.client import UserClient as Client
from pathlib import Path
import json
import tensorflow as tf
import credentials
import numpy as np
import time
from dotenv import load_dotenv
import os


load_dotenv()

print(f'data path: {os.environ.get("DATA_PATH", None)}')



local_config_path = "/home/swier/Documents/argos_dataset/argos_layout/assets/params.json"

current_path = Path(__file__).parent
data_path = f'{current_path}/../argos_dataset/argos_layout/'

local_testing = True


# read in config file
with open(local_config_path, "r") as fp:
    config = json.load(fp)


client = MockAlgorithmClient(
    datasets=[[],[]],
    module="argosfeddeep"
)
selected_orgs = client.organization.list()
participating_org_ids = [selected_org['id'] for selected_org in selected_orgs]
central_org_id = participating_org_ids[1]


# setup vantage stuff
task_input={
    "method":"central",
    "kwargs": {
        "config": config,
        # "rounds": 2,
        # "model_size": 200,
        "participating_ids": participating_org_ids
        # "model_weights" : weights_list
    },
}

task = client.task.create(
    input_= task_input,
    organizations=[central_org_id],
    # organizations=[3],
    # image = "sgarst/argos:dev",
    # collaboration = collab,
    name="argos_central",
    description="central part of argos algorithm"
)

print(task)
results = client.wait_for_results(task.get("id"))
print(results)