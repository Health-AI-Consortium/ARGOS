from typing import Any


from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient
from argosfeddeep.models import mod_resnet
from argosfeddeep.run_online import dice_bce
from argosfeddeep.utils import compute_average_weight
import tensorflow as tf
import json
import os
import numpy as np
import datetime


@algorithm_client
def test_gpu(client: AlgorithmClient) -> Any:
    import subprocess
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        pass

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    info(f"Physical GPUs: {gpus}")

    # Enable allow_growth so you can see incremental allocation rather than TF grabbing
    # everything (or being capped) silently
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # After growth is enabled, do a small dummy op, then check actual usage
    import numpy as np
    with tf.device('/GPU:0'):
        a = tf.constant(np.ones((10, 10)))

    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
    info(f"TF memory info after dummy op: {gpu_info}")



@algorithm_client
def test_central(client: AlgorithmClient) -> Any:
    organizations = client.organization.list()

    return {"organizations": organizations}
    org_ids = [organization.get("id") for organization in organizations]

    config = json.load(config_path)
    

@algorithm_client
def central(
    client: AlgorithmClient, config: dict, participating_ids: list
) -> Any:

    # get all organizations (ids) within the collaboration so you can send a
    # task to them.
    # client.setup_encryption(os.environ['PRIVATE_KEY_PATH'])

    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # print(f'organizations: {organizations}')
    # config = json.load(config_path)

    #TODO: add loading of initial weights
    # Define model so we can pull initial weights off of it
    # Define optimizer with learning rate (we need this to define a model)
    loss_function = dice_bce
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'],
                                                                    decay_steps=config['decay_steps'],
                                                                    decay_rate=config['decay_rate'],
                                                                    staircase=True)
    optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    init_model = mod_resnet(config,
                    config['num_classes'],
                    optimizer=optimizer_function,
                    loss=loss_function)

    init_weights = init_model.get_weights()
    weights_list = [init_weight.tolist() for init_weight in init_weights]
    # rng = np.random.default_rng(seed=7)
    # mock_model = list(rng.random(size = config['model_size']))
    # weights_list = mock_model

    results = {}
    for round in range (config['num_rounds']):
        results[f'round_{round}'] = {}
        info(f"round {round + 1} of {config['num_rounds']}")

        # Define input parameters for a subtask
        info("Defining input parameters")
        input_ = {
            "method": "train_locally",
            "kwargs": {
                "config" : config,
                # "config" : 7,
                "model_weights" :weights_list
            }
        }
        # info("A line just to test if this is getting updated..")


        # create a subtask for all organizations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=participating_ids,
            name=f"training round {round}",
            description="maybe this has to be filled in, idk"
        )

        # wait for node to return results of the subtask.
        info("Waiting for results")
        task_results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        for i, task_result in enumerate(task_results):
            results[f'round_{round}'][f'client_{i}'] = task_result['metrics']


        # task_metrics = [task_result['metrics'] for task_result in task_results]
        # info(f'result fields: {task_results[0].keys()}')
        # info(f'metrics: {task_results[0]["metrics"]}')
        # # log metrics
        # # results[f'round_{round}']['client_0'] = {}
        # # results[f'round_{round}']['client_1'] = {}
        # # results[f'round_{round}']['client_2'] = {}
        # results[f'round_{round}']['client_0'] = task_results[0]['metrics']
        # results[f'round_{round}']['client_1'] = task_results[1]['metrics']
        # results[f'round_{round}']['client_2'] = task_results[2]['metrics']

        # TODO: consider adding error checking on responses
        local_weights_list = [result['model weights'] for result in task_results]
        
        # average models
        # TODO: change this function such that it responds with the averaged model instead
        # alternatively, we could have the averaging as a subtask as well, and simply store the global model in the blob storage as well.
        # question then becomes how we could give access to that model to the clients
        # arguably this might not matter much, we have to transfer the global model to the clients either way 
        averaged_weights = compute_average_weight(local_weights_list)
        if round % config['save_central_model_step'] == 0:
            info("saving model weights")
            save_dir = f'{os.environ.get("DATA_PATH")}/model_checkpoints/{current_time}'
            os.makedirs(save_dir, exist_ok=True)
            init_model.set_weights(averaged_weights)
            init_model.save_weights(f'{save_dir}/iteration{round}.weights.h5')
        # aggregated_model_path, model_name = avg.fed_average(node_model_path,iteration=variables['iteration'])
        weights_list  = [weight.tolist() for weight in averaged_weights ]
    return results
