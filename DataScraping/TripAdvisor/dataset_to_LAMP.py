import pandas as pd
import numpy as np
import argparse
import json
from collections import defaultdict
import copy

import os


def reindex(prefix, dataset):
    # reindex the dataset with the prefix

    for index, item in enumerate(dataset):
        dataset[index] = {'id':f'{prefix}{index}', **dataset[index]}
    
    return dataset

def reindex_profile(input_dataset):
    # reindex the profile of the input_dataset with the prefix and reorder the dict keys

    for index, item in enumerate(input_dataset):
        history = item['profile']['history']
        for history_index, history_item in enumerate(history):
            input_dataset[index]['profile']['history'][history_index]= {'id': f"{item['id']}{history_index}", **input_dataset[index]['profile']['history'][history_index]}
    
    return input_dataset

def remove_canonical_id(dataset):
    # remove the canonical_id field from the dataset

    for index, item in enumerate(dataset):
        del dataset[index]['canonical_id']
        # remove canonical id in the 'profile' field
        history = item['profile']['history']
        for history_index, history_item in enumerate(history):
            del dataset[index]['profile']['history'][history_index]['input']['canonical_id']
    
    return dataset


def create_correct_id(prefix, input_dataset, output_dataset):
    # create a correct id for each input and output dataset

    input_dataset = reindex(prefix, input_dataset)
    input_dataset = reindex_profile(input_dataset)
    output_dataset = reindex(prefix, output_dataset)  # there is no profile in the output dataset

    remove_canonical_id(input_dataset)

    return input_dataset, output_dataset


if __name__ == '__main__':

    # Following essentially the LaMP paper, we create a dataset with the following format:
    # Input/X: everything about the review (TODO: Do we also make use of UP data from the field ['profile']['user_data'] turned into a template prompt?
    # Output/Y: review rating (or some other feature, in that case removed from the input or UP data)
    # Profile: dictionary of two fields: user_data and history. History is a list of all the other (input, output) pairs of the same user (empty list if no other pairs). Note that the UP info in the input field are raw (i.e., not turned into a prompt). 

    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("--output_column", type=str, default="review_rating", help="the output/target feature/variable/column")
    command_line_parser.add_argument("--language", type=str, default="en", help="language of the samples (e.g., en, fr, pt, es, it, de, zhCN, etc.)")
    command_line_parser.add_argument("--nb_k_samples", type=int, default=3, help="number of samples in thousands (e.g., 3 for 3K samples, 10 for 10K samples, etc.) being the number of samples in the (existing) final dataset")
    command_line_parser.add_argument("--lamp_benchmark_index", type=int, default=8, help="the index of the LaMP benchmark dataset being created/formatted (e.g., 8 for LaMP-8, 9 for LaMP-9, etc.)")

    args = command_line_parser.parse_args()

    output_column = args.output_column
    language_to_scrape = args.language
    nb_k_samples = args.nb_k_samples
    lamp_benchmark_index = args.lamp_benchmark_index

    # open csv file
    path_data = f'./TripAdvisor/Data/TA_final_dataset_{language_to_scrape.upper()}_{nb_k_samples}K.csv'
    dataset_df = pd.read_csv(path_data, delimiter="\t", encoding="utf-8", )

    # we can have an other output/target than review_rating, e.g.: review_city, or user_age_range, or user_sex, user_location, or user_nb_cities_visited, or user_tags. Of course, we would remove it from the input or UP.
    output_header = output_column    
    input_header = ['restaurant_reviewed_url', 'review_date', 'review_city', 'review_lang', 'review_rating', 'review_title', 'review']
    if output_header in input_header:
        input_header.remove(output_header)     # define the input header and remove the output/target if presents

    user_data_header = ['user_id_link', 'user_id', 'user_name', 'user_ta_level', 'user_age_range', 'user_sex', 'user_location', 'user_nb_contributions', 'user_nb_cities_visited', 'user_nb_helpful_votes', 'user_nb_photos', 'user_tags']
    if output_header in user_data_header:
        user_data_header.remove(output_header)     # define the user data header and remove the output/target if presents
    
    input_dataset = []
    output_dataset = []

    user_id_to_history = defaultdict(list)

    for index, review in dataset_df.iterrows():
        review_data = {key: review[key] for key in input_header}
        output = {output_header: review[output_header]}

        # TODO: Replace the value in the input field by the template prompt plus the review data (and user data?). --> refer to the LaMP paper to see the prompts/examples
        # TODO: Experiment with models on the dataset to see what template prompts would work well.
        
        input_item = {'canonical_id': index, 'input': "HERE PUT PROMPT", 'review_data': review_data}
        
        input_dataset.append(input_item)
        output_dataset.append(output)
        
        input_item_copy = copy.deepcopy(input_item)
        output_item_copy = copy.deepcopy(output)

        user_id = review['user_id']
        user_id_to_history[user_id].append({'input': input_item_copy, 'output': output_item_copy})
    
    for index, review in dataset_df.iterrows():
        user_id = review['user_id']
        user_data = {key: review[key] for key in user_data_header}
        user_history = user_id_to_history[user_id]
        history = [copy.deepcopy(item) for item in user_history if item['input']['canonical_id'] != index]
        input_dataset[index]['profile'] = {'user_data': user_data, 'history': history}


    # split the datasetuser_data into 3 parts: train, val (also called dev in LaMP), test

    # get indexes to randomly suffle the dataset
    indexes = list(range(len(input_dataset)))
    np.random.shuffle(indexes)
    
    # suffle the dataset
    input_dataset = [input_dataset[index] for index in indexes]
    output_dataset = [output_dataset[index] for index in indexes]

    # define the split ratio
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1

    nb_samples = len(input_dataset)
    nb_train_samples = int(nb_samples * train_split)
    nb_val_samples = int(nb_samples * val_split)
    nb_val_samples = int(nb_samples * test_split)

    train_input_dataset = input_dataset[:nb_train_samples]
    val_input_dataset = input_dataset[nb_train_samples:nb_train_samples+nb_val_samples]
    test_input_dataset = input_dataset[nb_train_samples+nb_val_samples:]

    train_output_dataset = output_dataset[:nb_train_samples]
    val_output_dataset = output_dataset[nb_train_samples:nb_train_samples+nb_val_samples]
    test_output_dataset = output_dataset[nb_train_samples+nb_val_samples:]

    # reindex the data samples to match the LaMP dataset indexing format: first digit for the LaMP dataset index minus 1, second digit for the split (0 for train, 1 for val, 2 for test), and the rest for the sample index in the dataset splits
    create_correct_id(str(lamp_benchmark_index - 1) + "0", train_input_dataset, train_output_dataset)
    create_correct_id(str(lamp_benchmark_index - 1) + "1", val_input_dataset, val_output_dataset)
    create_correct_id(str(lamp_benchmark_index - 1) + "2", test_input_dataset, test_output_dataset)
    
    # save the 6 datasets in 6 files; only the first 5 should be made public
    lamp_folder = f'./TripAdvisor/Data/LaMP/'
    if not os.path.exists(lamp_folder):
        os.makedirs(lamp_folder)

    lamp_version = "LaMP_" + str(lamp_benchmark_index) + f"_{nb_k_samples}K_"

    # we index it as 70 (7 for the index of the LaMP dataset in the benchmark and 0 because it is train)
    path_lamp_train_input = f'{lamp_folder}{lamp_version}train_input.json'
    with open(path_lamp_train_input, 'w') as f:
        json.dump(train_input_dataset, f)

    # we index it as 70 (7 for the index of the LaMP dataset in the benchmark and 0 because it is train)
    path_lamp_train_output = f'{lamp_folder}{lamp_version}train_output.json'
    with open(path_lamp_train_output, 'w') as f:
        json.dump({'task': "LaMP_8", 'golds': train_output_dataset}, f)

    # we index it as 71 (7 for the index of the LaMP dataset in the benchmark and 1 because it is val)
    path_lamp_val_input = f'{lamp_folder}{lamp_version}val_input.json'
    with open(path_lamp_val_input, 'w') as f:
        json.dump(val_input_dataset, f)
    
    # we index it as 71 (7 for the index of the LaMP dataset in the benchmark and 1 because it is val)
    path_lamp_val_output = f'{lamp_folder}{lamp_version}val_output.json'
    with open(path_lamp_val_output, 'w') as f:
        json.dump({'task': "LaMP_8", 'golds': val_output_dataset}, f)

    # we index it as 72 (7 for the index of the LaMP dataset in the benchmark and 2 because it is test)
    path_lamp_test_input = f'{lamp_folder}{lamp_version}test_input.json'
    with open(path_lamp_test_input, 'w') as f:
        json.dump(test_input_dataset, f)

    # Note that this dataset file should not be made public
    # we index it as 72 (7 for the index of the LaMP dataset in the benchmark and 2 because it is test)
    path_lamp_test_output = f'{lamp_folder}{lamp_version}test_output.json'
    with open(path_lamp_test_output, 'w') as f:
        json.dump({'task': "LaMP_8", 'golds': test_output_dataset}, f)