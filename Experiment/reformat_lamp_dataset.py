"""
This scripts takes a LaMP dataset in its official format and reformats it for it to later
be used for evaluations of models such as (zero-shot) GPT-3.5-turbo.

It also creates subsets of the datasets for experimenting with smaller datasets. 
The number of samples and the number of elements in the profile field (of each user) are cut to k elements each.

Moreover, it also creates datasets without the target in the input samples. This allows to llama-index on those datasets without potential leakage.

Note that the names of the LaMP splits should be ['train_inputs', 'train_outputs', 'val_inputs', 'val_outputs', 'test_inputs', 'test_outputs']
and are expected to be in their respective folders (e.g., /train for train_inputs and train_outputs) in the data folder.

Note also that uid for LaMP_1 does not follow the official LaMP benchmark uid format (i.e., for validation it is 010, 011, 012, etc.) 
for ease of use through the programs. Instead, except in the uid of the profile field, the leading 0 was not used 
(i.e., for validation it is 10, 11, 12, etc.) and it can easily be prepended later if needed.

"""

import os
import sys
import argparse
import pandas as pd    
import json

from loguru import logger
from utils import setup_loguru
setup_loguru(logger, os.path.basename(sys.argv[0]))


def format_lamp(input_df, output_df, new_dataset_name, samples):

    if output_df is not None:
        for sample_index in range(len(input_df)):
            
            uid = int(input_df.iloc[sample_index, 0])   # convert to int otherwise cannot dump json

            input_text = input_df.iloc[sample_index, 1]
            
            profile_field = input_df.iloc[sample_index, 2]

            gold = output_df.iloc[sample_index, 1]

            target_output = gold['output']

            samples.append({"uid": uid, "input": input_text, "output": target_output, "profile": profile_field})

    elif output_df is None or "no_target" in new_dataset_name:   # test set has no output dataset as it is part of the private benchmark
        for sample_index in range(len(input_df)):
            
            uid = int(input_df.iloc[sample_index, 0])   # convert to int otherwise cannot dump json

            input_text = input_df.iloc[sample_index, 1]
            
            profile_field = input_df.iloc[sample_index, 2]
            
            samples.append({"uid": uid, "input": input_text, "output": "", "profile": profile_field})

    return samples

def create_formatted_dataset(new_dataset_folder_path, new_dataset_name, input_df, output_df):

    samples = []
    if not os.path.exists(new_dataset_folder_path):
        os.makedirs(new_dataset_folder_path)

    with open(f'{new_dataset_folder_path}{new_dataset_name}', 'w') as reformatted_dataset_file:

        if "LaMP_1" in new_dataset_name:
            samples = format_lamp(input_df, output_df, new_dataset_name, samples)

        elif "LaMP_2" in new_dataset_name:
            samples = format_lamp(input_df, output_df, new_dataset_name, samples)

        elif "LaMP_3" in new_dataset_name:
            samples = format_lamp(input_df, output_df, new_dataset_name, samples)

        json.dump(samples, reformatted_dataset_file, indent=4)


def create_train_formatted(datasets, split_names, new_datasets_folder_path, new_datasets_name, create_subset, only_split_considered, k):
    
    logger.debug(split_names)

    if only_split_considered:
        train_input_df = pd.DataFrame.from_dict(datasets[split_names[0]])    # train_inputs
        train_output_df = pd.DataFrame.from_dict(datasets[split_names[1]])    # train_outputs
    else:   # yes it is the same however write it out explicitly for consistency accross the handling of the other splits (i.e., val, test)
        train_input_df = pd.DataFrame.from_dict(datasets[split_names[0]])    # train_inputs
        train_output_df = pd.DataFrame.from_dict(datasets[split_names[1]])    # train_outputs
    
    if create_subset:
        # only take the first K samples for each dataset
        train_input_df = train_input_df.iloc[:k, :]
        train_output_df = train_output_df.iloc[:k, :]
        train_input_df['profile'] = train_input_df['profile'].apply(lambda x: x[:k])
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train_subset.json', train_input_df, train_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train_subset_no_target.json', train_input_df, None)
    else:
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train.json', train_input_df, train_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train_no_target.json', train_input_df, None)
    
def create_val_formatted(datasets, split_names, new_datasets_folder_path, new_datasets_name, create_subset, only_split_considered, k):

    logger.debug(split_names)

    if only_split_considered:
        val_input_df = pd.DataFrame.from_dict(datasets[split_names[0]])    # val_input
        val_output_df = pd.DataFrame.from_dict(datasets[split_names[1]])    # val_output
    else:
        val_input_df = pd.DataFrame.from_dict(datasets[split_names[2]])    # val_input
        val_output_df = pd.DataFrame.from_dict(datasets[split_names[3]])    # val_output

    if create_subset:
        val_input_df = val_input_df.iloc[:k, :]
        val_output_df = val_output_df.iloc[:k, :]
        val_input_df['profile'] = val_input_df['profile'].apply(lambda x: x[:k])
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val_subset.json', val_input_df, val_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val_subset_no_target.json', val_input_df, None)

    else:
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val.json', val_input_df, val_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val_no_target.json', val_input_df, None)

def create_test_formatted(datasets, split_names, new_datasets_folder_path, new_datasets_name, create_subset, only_split_considered, k):

    logger.debug(split_names)

    if only_split_considered:
        test_input_df = pd.DataFrame.from_dict(datasets[split_names[0]])    # test_input
    else:
        test_input_df = pd.DataFrame.from_dict(datasets[split_names[4]])    # test_input

    if "LaMP_8" in new_datasets_name:
        if only_split_considered:
            test_output_df = pd.DataFrame.from_dict(datasets[split_names[1]])   # test_output
        else:
            test_output_df = pd.DataFrame.from_dict(datasets[split_names[5]])   # test_output

        if create_subset:
            test_output_df = test_output_df.iloc[:k, :]
    else:
        test_output_df = None

    if create_subset:
        test_input_df = test_input_df.iloc[:k, :]
        test_input_df['profile'] = test_input_df['profile'].apply(lambda x: x[:k])
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test_subset.json', test_input_df, test_output_df) # LaMP_8 test output is available
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test_subset_no_target.json', test_input_df, None) # LaMP_8 test output is available
    
    else:
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test.json', test_input_df, test_output_df) # LaMP_8 test output is available
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test_no_target.json', test_input_df, None) # LaMP_8 test output is available

def reformat_lamp_datasets(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, only_split_considered, create_subset=False, nb_samples_subset=None):

    if nb_samples_subset is not None:
        k = nb_samples_subset

    # json to pandas dataframe
    if only_split_considered == "train":
        create_train_formatted(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, create_subset, only_split_considered, k)
    elif only_split_considered == "val":
        create_val_formatted(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, create_subset, only_split_considered, k)
    elif only_split_considered == "test":
        create_test_formatted(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, create_subset, only_split_considered, k)

    elif only_split_considered is None:
        create_train_formatted(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, create_subset, only_split_considered, k)
        create_val_formatted(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, create_subset, only_split_considered, k)
        create_test_formatted(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, create_subset, only_split_considered, k)

def get_datasets(data_folder_path, lamp_dataset_index, lamp_8_samples_version, only_split_considered):

    if not lamp_dataset_index:
        logger.warning("Please provide the LaMP dataset index. Exiting...")
        quit()

    paths_data = []

    if lamp_dataset_index == "8":
        split_names = ['train_inputs', 'train_outputs', 'val_inputs', 'val_outputs', 'test_inputs', 'test_outputs']
        lamp_dataset_index = lamp_dataset_index + "_" + lamp_8_samples_version
        logger.info(f"LaMP_8 version: LaMP_{lamp_dataset_index}")
        full_names = [split_name.split('_')[0] + "/" + "LaMP_8_" + lamp_8_samples_version + "_" + split_name for split_name in split_names]

    else:   # for all the other LaMP datasets
        split_names = ['train_inputs', 'train_outputs', 'val_inputs', 'val_outputs', 'test_inputs']
        logger.info(f"LaMP index: LaMP_{lamp_dataset_index}")
        full_names = [split_name.split('_')[0] + "/" + split_name for split_name in split_names]

    logger.debug("BEFORE ITERATION OVER NAMES...")
    logger.debug(f"Full names: {full_names}")
    logger.debug(f"Split names: {split_names}")

    split_names_temp = split_names.copy()
    for full_name, split_name in zip(full_names, split_names):
        original_dataset_path = f'{data_folder_path}lamp_original/LaMP_{lamp_dataset_index}/{full_name}.json'

        # check if the file with path original_dataset_path exists
        if not os.path.exists(original_dataset_path):
            logger.debug(f"Removing split name: {split_name} from {split_names_temp} because path does not exist")
            split_names_temp.remove(split_name)

        elif only_split_considered:
            if only_split_considered not in split_name:
                logger.debug(f"Removing split name: {split_name} from {split_names_temp} because only_split_considered not in split_name")
                split_names_temp.remove(split_name)
            else: 
                logger.debug(f"Appending to path data: {original_dataset_path}")
                paths_data.append(original_dataset_path)
        else:
            logger.debug(f"Appending to path data: {original_dataset_path}")
            paths_data.append(original_dataset_path)


    logger.debug(f"Paths data: {paths_data}")
    logger.debug(f"Make sure that the split names remaining are only that of VAL: {split_names_temp}")

    datasets = {dataset_key: pd.read_json(path_data, orient='records') for path_data, dataset_key in zip(paths_data, split_names_temp)}
    
    logger.debug(datasets.keys())
    for key in datasets:
        logger.debug(datasets[key].shape)
        logger.debug(datasets[key].head())

    formatted_dataset_folder_path = f'{data_folder_path}lamp_reformatted/LaMP_{lamp_dataset_index}/'
    formatted_dataset_name = f'LaMP_{lamp_dataset_index}_dataset'


    return datasets, split_names_temp, formatted_dataset_folder_path, formatted_dataset_name


if __name__ == "__main__":
        
    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument("--lamp_dataset_index", type=str, default=None, help="LaMP dataset index. E.g., 1, 2, 3, etc.")
    command_line_parser.add_argument("--lamp_8_samples_version", type=str, default="3K", help="Rounded down in thousands number of samples for LaMP_8 dataset. E.g., 3K, 10K, etc.")
    command_line_parser.add_argument("--only_split_considered", type=str, default=None, help="The name of the split to reformat if only a particular split has to be reformatted. E.g.: train, val, test")
    command_line_parser.add_argument("--create_subset", action='store_true', default=False, help="Whether to create llamaindex-dataset subsets too for experimenting.")
    command_line_parser.add_argument("--nb_samples_subset", type=int, default=None, help="Number of samples in the subset to create.")


    args = command_line_parser.parse_args()

    lamp_dataset_index = args.lamp_dataset_index
    lamp_8_samples_version = args.lamp_8_samples_version
    only_split_considered = args.only_split_considered
    create_subset = args.create_subset
    if create_subset:
        nb_samples_subset = args.nb_samples_subset
        if nb_samples_subset is None:
            logger.warning("Please provide the number of samples in the subset to create. Exiting...")
            quit()

    data_folder_path = "./Experiment/Data/"

    datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name = get_datasets(data_folder_path, lamp_dataset_index, lamp_8_samples_version, only_split_considered)

    logger.info(f"Formatted datasets folder path: {formatted_datasets_folder_path}")
    logger.info(f"Formatted datasets prefix name: {formatted_datasets_name}")

    reformat_lamp_datasets(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, only_split_considered, create_subset, nb_samples_subset)