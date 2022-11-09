import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics
import logging


###############Load config.json and get path variables
logging.info('Loading config.json')
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
target = config['target']
log_level = config['log_level']
prod = config['prod']


# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)

logging.info('**** Starting reporting.py ****')

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model

    #load test data
    logging.info(f'Loading test data from {test_data_path}')
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    #get predictions
    logging.info('Getting predictions')
    y_pred, y_test = diagnostics.model_predictions(df_path=None)

    #get actual values
#    y_test = df[target]

    #calculate confusion matrix
    logging.info('Calculating confusion matrix')
    cm = metrics.confusion_matrix(y_test, y_pred)

    if prod:
        cm_path = os.path.join(output_model_path, 'confusionmatrix2.png')
    else:
        cm_path = os.path.join(output_model_path, 'confusionmatrix.png')


    #write the confusionmatrix.png to the output_model_path
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(metrics.accuracy_score(y_test, y_pred))
    plt.title(all_sample_title, size = 15);
    plt.savefig(cm_path)
    logging.info(f'Confusion Matrix saved to {cm_path}')



if __name__ == '__main__':
    score_model()
