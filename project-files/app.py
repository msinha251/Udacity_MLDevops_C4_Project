from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
import scoring#
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, missing_values, execution_time, outdated_packages_list 
#import predict_exited_from_saved_model
import json
import os
import logging


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# Load config.json
logging.info('Loading config.json')
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
log_level = config['log_level']

prediction_model = None

# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS', 'GET'])
def predict():        
    #call the prediction function you created in Step 3
    
    #get the dataset path from the request
    if request.method == 'POST':
        dataset_path = request.get_json()['dataset_path']
        logging.info(f'Using dataset path: {dataset_path}')
        dataset_path = None

        #get the prediction
        prediction, _  = model_predictions(dataset_path)
        logging.info(f'Got prediction: {prediction}')
    else:
        dataset_path = None
        logging.info('Using default dataset path')
        #get the prediction
        prediction, _ = model_predictions(dataset_path)
        logging.info(f'Got prediction: {prediction}')
    
    #return the prediction
    return jsonify({'predictions': prediction.tolist()})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model

    f1_score = score_model()

    return jsonify(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary_stats = dataframe_summary()
    
    # convert to json
    summary_stats = json.dumps(summary_stats)
    json_summary_stats = json.loads(summary_stats)

    # return list of summary stats
    return jsonify(json_summary_stats)


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing, percent NA and dependecy checks
    missing = {"missing %": missing_values()}
    timing = {"timing": execution_time()}
    dependency = {"packages_table": outdated_packages_list()}

    return jsonify(missing, timing, dependency)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
