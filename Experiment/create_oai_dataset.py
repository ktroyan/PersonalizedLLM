import os
import sys
import time
import csv
import argparse
import pandas as pd    
import hashlib
import base64
import re
import random
import string
from collections import OrderedDict
import json

def write_oai_dataset(new_dataset_folder_path, new_dataset_name, input_df, output_df=None):

    samples = []

    dataset_path = new_dataset_folder_path
    with open(f'{dataset_path}{new_dataset_name}', 'w') as oai_dataset:
        profile_texts = []
        for sample_index in range(len(input_df)):
            
            input_text = input_df.iloc[sample_index,1]
            # print(input_text)
            
            profile_field = input_df.iloc[sample_index,2]
            profile_texts += [profile_field[j]['text'] + ";" for j in range(len(profile_field))]
            # print(profile_texts)

            if output_df is not None:
                gold = output_df.iloc[sample_index,1]
                # print(gold)
                target_output = gold['output']
                # print(target_output)

                samples.append({"prompt": input_text, "completion": target_output})
            else:   # test set has no output dataset as in it is part of the private benchmark
                samples.append({"prompt": input_text, "completion": ""})
        
        json.dump(samples, oai_dataset, indent=4)


def create_openai_datasets(datasets, split_names, new_dataset_folder_path, new_dataset_name):

    # json to pandas dataframe
    train_input_df = pd.DataFrame.from_dict(datasets[split_names[0]])    # train_input
    train_output_df = pd.DataFrame.from_dict(datasets[split_names[1]])    # train_output

    val_input_df = pd.DataFrame.from_dict(datasets[split_names[2]])    # val_input
    val_output_df = pd.DataFrame.from_dict(datasets[split_names[3]])    # val_output

    test_input_df = pd.DataFrame.from_dict(datasets[split_names[4]])    # test_input

    # print(val_input_df.head())
    # print(val_output_df.head())

    # print(val_input_df.shape)
    # print(val_output_df.shape)

    write_oai_dataset(new_dataset_folder_path, f'{new_dataset_name}_train.json', train_input_df, train_output_df)
    write_oai_dataset(new_dataset_folder_path, f'{new_dataset_name}_val.json', val_input_df, val_output_df)
    write_oai_dataset(new_dataset_folder_path, f'{new_dataset_name}_test.json', test_input_df, None)

def get_datasets(data_folder_path, lamp_dataset_index, lamp_8_samples_version, print_info=False):

    if lamp_8_samples_version:
        lamp_dataset_index = "8"

    if lamp_dataset_index == "8" and not lamp_8_samples_version:
        lamp_8_samples_version = "3K"

    if (not lamp_dataset_index):
        print("Please provide the LaMP dataset version or the LaMP_8 version. Exiting...")
        quit()

    paths_data = []

    if lamp_dataset_index == "8":
        print(f"LaMP version: LaMP_{lamp_dataset_index}_{lamp_8_samples_version}")
        split_names = ['train_input', 'train_output', 'val_input', 'val_output', 'test_input', 'test_output']
        full_names = [full_name.split('_')[0] + "/" + "LaMP_8_" + lamp_8_samples_version + "_" + full_name for full_name in split_names]

    else:   # for all the other versions
        print(f"LaMP version: LaMP_{lamp_dataset_index}")
        split_names = ['train_questions', 'train_outputs', 'dev_questions', 'dev_outputs', 'test_questions']
        full_names = [full_name.split('_')[0] + "/" + full_name for full_name in split_names]


    for full_name, split_name in zip(full_names, split_names):
        paths_data.append(f'{data_folder_path}LaMP_{lamp_dataset_index}/{full_name}.json')

    # This is the final dataset that can be used further in the project
    datasets = {dataset_key: pd.read_json(path_data, orient='records') for path_data, dataset_key in zip(paths_data, split_names)}
    
    if print_info:
        print(datasets.keys())
        for key in datasets:
            print(datasets[key].shape)
            print(datasets[key].head())
            print("\n\n")

    oai_dataset_folder_path = f'{data_folder_path}OAI/'
    oai_dataset_name = f'LaMP_{lamp_dataset_index}_dataset'

    return datasets, split_names, oai_dataset_folder_path, oai_dataset_name


if __name__ == "__main__":
        
    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument("--lamp_dataset_index", type=str, default=None, help="LaMP dataset index. E.g., 1, 2, 3, etc.")
    command_line_parser.add_argument("--lamp_8_samples_version", type=str, default=None, help="Rounded down in thousands number of samples for LaMP_8 dataset. E.g., 3K, 10K, etc.")

    args = command_line_parser.parse_args()

    lamp_dataset_index = args.lamp_dataset_index
    lamp_8_samples_version = args.lamp_8_samples_version

    data_folder_path = "./Experiment/Data/"

    datasets, split_names, oai_dataset_folder_path, oai_dataset_name = get_datasets(data_folder_path, lamp_dataset_index, lamp_8_samples_version, print_info=False)

    print(oai_dataset_folder_path)
    print(oai_dataset_name)

    create_openai_datasets(datasets, split_names, oai_dataset_folder_path, oai_dataset_name)