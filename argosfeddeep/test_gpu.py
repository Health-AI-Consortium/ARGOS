from typing import Any
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient



@algorithm_client
def test_gpu(client: AlgorithmClient) -> Any:
    import subprocess
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        pass

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print("Physical GPUs:", gpus)

    # Enable allow_growth so you can see incremental allocation rather than TF grabbing
    # everything (or being capped) silently
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # After growth is enabled, do a small dummy op, then check actual usage
    import numpy as np
    with tf.device('/GPU:0'):
        a = tf.constant(np.ones((10, 10)))

    info = tf.config.experimental.get_memory_info('GPU:0')
    print("TF memory info after dummy op:", info)