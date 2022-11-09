
import pandas as pd
import numpy as np
import pickle
import timeit
import os
import json
import logging



##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
target = config['target']
log_level = config['log_level']


# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)

logging.info('**** diagnostics.py ****')


##################Function to get model predictions
def model_predictions(df_path=None):
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))

    # Load test data
    logging.info(f'Loading test data from {test_data_path}')
    if df_path is None:
        df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    else:
        logging.info('Using input dataframe path')
        # df = pd.read_csv(os.path.join(df_path, 'testdata.csv'))
        df = pd.read_csv(df_path)

    # Dropping corporation column
    logging.info('Dropping corporation column')
    df.drop('corporation', axis=1, inplace=True)

    # predict on test data
    logging.info('Dropping target column')
    X = df.drop(target, axis=1)

    logging.info('Calculating predictions')
    y_pred = model.predict(X)
    

    return y_pred, df[target].values

##################Function to get summary statistics
def dataframe_summary(df_path = None):
    #calculate summary statistics here

    if df_path is None:
        # load final dataset:
        logging.info(f'Loading final dataset from {output_folder_path}/finaldata.csv')
        df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv')) 
    else:
        logging.info('Using input dataframe path')
        df = pd.read_csv(os.path.join(df_path, 'finaldata.csv'))

    # calculate summary statistics (mean, median, std) in list for each column
    logging.info('Calculating summary statistics')
    summary = {}
    summary['mean'] = df.mean().to_dict()
    summary['median'] = df.median().to_dict()
    summary['std'] = df.std().to_dict()

    return summary

##################Function to get missing values
def missing_values(df = None):
    #calculate missing values here
    if df is None:
        # load final dataset:
        logging.info(f'Loading final dataset from {output_folder_path}/finaldata.csv')
        df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv')) 
    else:
        logging.info('Using input dataframe')

    # calculate missing values percentage in list for each column
    logging.info('Calculating missing values')
    missing = (df.isnull().sum()/len(df)).to_dict()

    logging.info(f'Missing values: \n{missing}')
    return missing


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py

    # capture execution time of ingestion.py
    start_time = timeit.default_timer()
    logging.info('Calculating ingestion.py execution time')
    os.system('python3 ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    logging.info(f'Ingestion.py execution time: {ingestion_time}')

    # capture execution time of training.py
    start_time = timeit.default_timer()
    logging.info('Calculating training.py execution time')
    os.system('python3 training.py')
    training_time = timeit.default_timer() - start_time
    logging.info(f'Training.py execution time: {training_time}')

    return {'ingestion_time': ingestion_time, 'training_time': training_time}


##################Function to check dependencies
def outdated_packages_list():
    # compare package version from requirements.txt with latest version
    logging.info('List of outdated packages')
    outdated_packages = os.popen('pip list --outdated').read()
    logging.info(f'Outdated packages: \n{outdated_packages}')

    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions(None)
    dataframe_summary(None)
    missing_values()
    execution_time()
    outdated_packages_list()





    
