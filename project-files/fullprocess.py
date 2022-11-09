

import training
import scoring
import deployment
import diagnostics
import reporting
import logging
import json
import os
import ast

# load config file
with open('config.json','r') as f:
    config = json.load(f)

# get config variables
input_folder_path = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path'] 
output_model_path = config['output_model_path']
log_level = config['log_level'] 
target = config['target']  
prod = config['prod']

# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)


##################Check and read new data
#first, read ingestedfiles.txt
if os.path.exists(os.path.join(prod_deployment_path, 'ingestedfiles.txt')):
    logging.info('Reading ingestedfiles.txt')
    with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
        ingestedfiles = ast.literal_eval(f.read())
    logging.info(f'ingestedfiles.txt: {ingestedfiles}')
else:
    logging.info('ingestedfiles.txt does not exist')
    ingestedfiles = []

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
logging.info('Determining whether the source data folder has files that aren\'t listed in ingestedfiles.txt')
newfiles = []
for file in os.listdir(input_folder_path):
    if file not in ingestedfiles and file.endswith('.csv'):
        newfiles.append(file)
logging.info(f'New files: {newfiles}')




##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(newfiles) > 0:
    logging.info('New files found. Proceeding with the ingestion process.')
    os.system('python3 ingestion.py')
else:
    logging.info('No new files found. Ending the ingestion process.')
    exit()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

# read latestscore.txt and get the score
with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
    old_score = float(f.read())
logging.info(f'old_score: {old_score}')

# get score on the latest ingested data
new_score = scoring.score_model()
logging.info(f'new_score: {new_score}')

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
drift = False
if new_score < old_score:
    logging.info('Model drift detected. Proceeding with the retraining process.')
    drift = True
# else:
#     logging.info('No model drift detected. Ending deployment process.')
#     exit()


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
if drift:
    logging.info('Re-training model')
    training.train_model('./ingesteddata/finaldata.csv')

    logging.info('Re-deploying model')
    os.system('python3 deployment.py')
elif prod == False:
    logging.info('Re-training model for testing purposes')
    training.train_model('./ingesteddata/finaldata.csv')
    logging.info('Re-deploying model')
    os.system('python3 deployment.py')
else:
    logging.info('No model drift detected. Ending deployment process.')
    exit()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
logging.info('Running reporting.py')
os.system('python3 reporting.py')

logging.info('Running apicalls.py')
os.system('python3 apicalls.py')










