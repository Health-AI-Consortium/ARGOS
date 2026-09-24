"""
main entry point for running the ARGOS usecase on Vantage6
"""

from vantage6.client import UserClient as Client
from pathlib import Path
import json
import credentials
import numpy as np
import time

def main(client: Client, input: dict, organizations: list):
    # Run the central method on 1 node and get the results (assuming we have one collab, while we're testing like this)
    # collab = client.collaboration.list()
    # print(f'collab: {collab}')
    # ['id']


    collab = client.collaboration.list()['data'][0]['id']

    task = client.task.create(
        input_= input,
        organizations=organizations,
        # organizations=[3],
        image = credentials.algorithm_image,
        collaboration = collab,
        name="argos_central",
        description="central part of argos algorithm"
    )

    print(task)
    results = client.wait_for_results(task.get("id"))
    print(results)

    return results




if __name__ == "__main__":
    local_config_path = Path(__file__).resolve().parents[1] / "config.json"

    # read in config file
    with open(local_config_path, "r") as fp:
        config = json.load(fp)



    client = Client(
        credentials.server_url, credentials.server_port, credentials.server_api, log_level = 'debug'
    )
    client.authenticate(credentials.usn, credentials.psswrd)
    client.setup_encryption(credentials.private_key)
    
    organizations = client.organization.list()['data']
    selected_orgs = np.array(organizations)[[org['name'].startswith('ARGOS') for org in organizations]]

    
    participating_org_ids = [selected_org['id'] for selected_org in selected_orgs]
    central_org_id = participating_org_ids[0]



    # setup vantage stuff
    task_input={
        "method":"central",
        # "method" : "test_gpu",
        "kwargs": {
            "config": config,
            # "rounds": 2,
            # "model_size": 200,
            "participating_ids": participating_org_ids[:3]
            # "model_weights" : weights_list
        },
    }

    # print(f'config: {config} ')

    # print(f'participating org ids: {participating_org_ids}')

    # print(f'selected org ids: {selected_org_ids}')

    results = main(client, task_input, [central_org_id])

    with open(f'argos_results_{time.time()}.json', "w") as fp:
        json.dump(results, fp)