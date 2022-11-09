from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging


##################Load config.json and correct path variable
logging.info('Loading config.json')
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(config['input_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
log_level = config['log_level']


# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)

logging.info('**** deployment.py ****')


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    #copy the latest pickle file
    os.system(f'cp {output_model_path}/trainedmodel.pkl {prod_deployment_path}')
    logging.info(f'Copied latest pickle file to {prod_deployment_path}')

    #copy the latestscore.txt value
    os.system(f'cp {output_model_path}/latestscore.txt {prod_deployment_path}')
    logging.info(f'Copied latestscore.txt value to {prod_deployment_path}')

    #copy the ingestfiles.txt file
    os.system(f'cp {input_folder_path}/ingestedfiles.txt {prod_deployment_path}') 
    logging.info(f'Copied ingestfiles.txt file to {prod_deployment_path}')

if __name__ == '__main__':
    store_model_into_pickle()
        
        

