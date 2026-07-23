from vantage6.client import UserClient as Client
import credentials
import numpy as np
import json
import time


task_id = 359



client = Client(
    credentials.server_url, credentials.server_port, credentials.server_api, log_level = 'debug'
)
client.authenticate(credentials.usn, credentials.psswrd)
client.setup_encryption("/home/swier/Documents/healthAI_argos/private_key")
# client.setup_encryption("/home/swier/Documents/private_key_ARGOS4")


results = client.result.from_task(task_id)

print(results)

# with open(f'timing_results_{time.time()}', "w") as fp:
#     json.dump(times, fp)