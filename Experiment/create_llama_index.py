import os
import sys
import time
import csv
import argparse
import pandas as pd    
import json
import re

import openai

import llama_index
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader


def load_datasets(train_input_path, val_input_path, test_input_path):
    # Load the dataset
    with open(train_input_path) as f:
        train_input = pd.DataFrame.from_dict(json.load(f))
    
    with open(val_input_path) as f:
        val_input = pd.DataFrame.from_dict(json.load(f))

    with open(test_input_path) as f:
        test_input = pd.DataFrame.from_dict(json.load(f))
    
    print("Loaded the train-val-test input datasets.")

    print(train_input.head())
    print(val_input.head())
    print(test_input.head())

    print("\n")

    print(train_input.shape)
    print(val_input.shape)
    print(test_input.shape)

    print("\n")

    return train_input, val_input, test_input


def create_llama_index(train_input, val_input, test_input):
    # documents_train_set = SimpleDirectoryReader("./Experiment/Data/pg_wiwo").load_data()
    documents_train_set = SimpleDirectoryReader("./Experiment/Data/LaMP_3/dev").load_data()     # https://gpt-index.readthedocs.io/en/latest/reference/readers.html#llama_index.readers.SimpleDirectoryReader
    # print(documents_train_set)

    index_train_set = GPTVectorStoreIndex.from_documents(documents_train_set, openai_api_key=openai.api_key, name="train_set")
    # print(index_train_set)

    query_engine = index_train_set.as_query_engine()
    # print(query_engine)

    response1 = query_engine.query("What is the text review of the user with id 210?")
    print(response1)

    response2 = query_engine.query("What is the second sample within the field profile of the user with id 210?")
    print(response2)

    response3 = query_engine.query("Given the text review of the user with id 210, output what samples within the field profile of the user with id 210 are the most informative sample in order to predict accurately the score?")
    print(response3)

    useful_profile_samples_ids = re.findall(r'\d+', response3)
    
    for sample_id in useful_profile_samples_ids:
        print(train_input[int(sample_id)])

    model_input_prompt = response1 + response3


if __name__ == '__main__':

    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument("--USE_OAI_API_KEY", action='store_true', default=True, help="If there is a valid OpenAI API key in a local .txt file.")
    command_line_parser.add_argument("--USE_OAI_ACCESS_TOKEN", action='store_true', default=False, help="If there is a valid access token to OpenAI API in a local .txt file. Works for V1")    
    command_line_parser.add_argument("--lamp_dataset_index", type=str, default=3, help="LaMP dataset index. E.g., 1, 2, 3, etc.")

    args = command_line_parser.parse_args()

    lamp_dataset_index = args.lamp_dataset_index

    if args.USE_OAI_API_KEY:
        # read the oai API key from the text file
        with open('./Experiment/oai_api_private_key.txt','r') as f:
            openai.api_key = f.read().replace('\n', '')
            OAI_API_KEY = openai.api_key

    if args.USE_OAI_ACCESS_TOKEN:
        with open('./Experiment/oai_api_access_token.txt','r') as f:
            OAI_ACCESS_TOKEN = f.read().replace('\n', '')

    data_folder_path = "./Experiment/Data/LaMP_" + str(lamp_dataset_index) + "/"

    train_input_path = data_folder_path + "train/" + "train_questions.json"
    val_input_path = data_folder_path + "dev/" + "dev_questions.json"
    test_input_path = data_folder_path + "test/" + "test_questions.json"

    # train_input, val_input, test_input = load_datasets(train_input_path, val_input_path, test_input_path)
    
    create_llama_index(train_input_path, val_input_path, test_input_path)
