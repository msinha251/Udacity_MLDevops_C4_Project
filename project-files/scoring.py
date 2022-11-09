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


#################Load config.json and get path variables
logging.info('Loading config.json')
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 
target = config['target']
log_level = config['log_level']
prod = config['prod']

# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)

logging.info('**** scoring.py ****')

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Load model
    logging.info(f'Loading model from {output_model_path}')
    model = pickle.load(open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'rb'))

    # # Load test data
    # if df_path is None:
    #     logging.info(f'Loading test data from {test_data_path}')
    #     df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    # else:
    #     logging.info(f'Loading from {df_path}')
    #     df = pd.read_csv(df_path)
    df_path = os.path.join(test_data_path, 'testdata.csv')
    logging.info(f'Loading test data from {df_path}')
    df = pd.read_csv(df_path)

    # Dropping corporation column
    logging.info('Dropping corporation column')
    df.drop('corporation', axis=1, inplace=True)

    # calculate F1 score
    X = df.drop(target, axis=1)
    y = df[target]
    logging.info('Calculating F1 score')
    y_pred = model.predict(X)
    f1 = metrics.f1_score(y, y_pred)
    logging.info(f'F1 score: {f1}')

    # write F1 score to latestscore.txt
    logging.info(f'Writing F1 score to {output_model_path}/latestscore.txt')
    with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1))

    return f1

# def get_f1_score(y_true, y_pred):
#     return metrics.f1_score(y_true, y_pred)


if __name__ == '__main__':
    score_model()




