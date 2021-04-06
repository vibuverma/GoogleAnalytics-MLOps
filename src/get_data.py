# Read Parameters
# Process
# Return dataframe

import os
import yaml
import pandas as pd
import argparse
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def read_params(config_path):
    with open(config_path) as yaml_file:
        config= yaml.safe_load(yaml_file)
    logging.info("Trying to load .csv file.")
    return config




def get_data(config_path):
    logging.info(".csv file imported successfully.")
    config= read_params(config_path)
    #print(config)
    train_path= config["data_source"]["train_source"]
    test_path= config["data_source"]["test_source"]
    df= pd.read_csv(train_path, sep=",", encoding='utf-8', nrows=10000)
    df1= pd.read_csv(test_path, sep=",", encoding='utf-8', nrows=10000)
    return (df, df1)




if __name__=="__main__":
    args =argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args= args.parse_args()
    train_df, test_df = get_data(config_path=parsed_args.config)

