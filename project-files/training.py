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


###################Load config.json and get path variables
logging.info('Loading config.json')
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path']) 
target = config['target']
log_level = config['log_level']

# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)

logging.info('** trining.py **')

#################Function for training the model
def train_model(df_path=None):
    logging.info('** Training the model ** using train_model()')

    # Load dataset
    if df_path is None:
        logging.info(f'Loading dataset from {dataset_csv_path}')
        df_original = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    else:
        logging.info(f'Loading from {df_path}')
        df_original = pd.read_csv(df_path)
    df = df_original.copy()

    # Dropping corporation column
    logging.info('Dropping corporation column')
    df.drop('corporation', axis=1, inplace=True)

    # Split dataset into train and test
    logging.info('Splitting dataset into train and test')
    X = df.drop(target, axis=1)
    y = df[target]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    #use this logistic regression for training
    logging.info('Training model')
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info(f'Writing model to {output_model_path}/trainedmodel.pkl')
    pickle.dump(model, open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'wb'))

if __name__ == '__main__':
    train_model()



