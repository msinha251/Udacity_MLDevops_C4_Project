import requests
import json
import os
import logging

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# load config file
with open('config.json','r') as f:
    config = json.load(f)

output_model_path = config['output_model_path']
test_data_path = config['test_data_path']
target = config['target']
log_level = config['log_level']
prod = config['prod']

# call logging_setup function
from logging_setup import setup_logging
setup_logging('INFO')

logging.info('**** apicalls.py ****')

# URL for the API call
#URL = "http://localhost:8000/"


# api 1 - get model predictions
logging.info('Calling API 1 - get model predictions')
prediction_post_response = requests.post(URL + "prediction", json={"dataset_path": test_data_path})

# api 2 - scoring
logging.info('Calling API 2 - scoring')
scoring_get_response = requests.get(URL + "scoring")

# api 3 - summary stats
logging.info('Calling API 3 - summary stats')
stats_get_response = requests.get(URL + "summarystats")

# api 4 - diagnostics
logging.info('Calling API 4 - diagnostics')
diagnostics_get_response = requests.get(URL + "diagnostics")

#combine all API responses
logging.info('Combining all API responses')
responses = [prediction_post_response, scoring_get_response, stats_get_response, diagnostics_get_response]

#write the responses to apireturns.txt
if prod:
    with open(os.path.join(output_model_path, 'apireturns2.txt'), 'w') as f:
        for response in responses:
            f.write(response.text)
    logging.info(f'Saved API responses to {output_model_path}/apireturns2.txt')
else:
    with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as f:
        for response in responses:
            f.write(response.text)
    logging.info(f'Saved API responses to {output_model_path}/apireturns.txt')





