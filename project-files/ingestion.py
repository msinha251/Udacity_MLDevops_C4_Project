import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging


#############Load config.json and get input and output paths
logging.info('Loading config.json')
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
log_level = config['log_level']


# call logging_setup function
from logging_setup import setup_logging
setup_logging(log_level)

logging.info('**** ingestion.py ****')

#############Function for data ingestion
def merge_multiple_dataframe():
    logging.info('** Merging multiple dataframes ** using merge_multiple_dataframe()')
    #check for datasets, compile them together, and write to an output file

    # Get list of files in input folder
    logging.info(f'Getting list of files from {input_folder_path}')
    files = os.listdir(input_folder_path)

    # Get list of csv files
    logging.info('Getting list of csv files')
    csv_files = [file for file in files if file.endswith('.csv')]

    # Load csv files into pandas dataframes
    logging.info('Loading csv files into pandas dataframes')
    df = pd.DataFrame()
    ingestedDataRecords = []
    for file in csv_files:
        print(file)
        # Read csv file and append to df
        logging.info(f'Reading {file} into pandas dataframe')
        df = df.append(pd.read_csv(os.path.join(input_folder_path, file)))

        # # Create ingestedDataRecord
        # ingestedDataRecord = {
        #     'filename': file,
        #     'ingestion_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # }
        ingestedDataRecords.append(file)

    # Reset index
    logging.info('Resetting index')
    df.reset_index(drop=True, inplace=True)

    # Remove duplicates
    logging.info('Removing duplicates')
    df.drop_duplicates(inplace=True)

    # Save to output folder
    logging.info(f'Saving to output folder: {output_folder_path}/finaldata.csv')
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    # Save ingestedDataRecords to output folder as list
    logging.info(f'Saving ingestedDataRecords to output folder: {input_folder_path}/ingestedfiles.txt')
    with open(os.path.join(input_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write(str(ingestedDataRecords))


if __name__ == '__main__':
    merge_multiple_dataframe()
    logging.info('** End of ingestion.py **')
