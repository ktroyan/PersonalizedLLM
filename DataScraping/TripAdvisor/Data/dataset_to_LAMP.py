import pandas as pd
import numpy as np
import argparse
import json
from collections import defaultdict
import copy


def reindex(prefix, dataset):
    # reindex the dataset with the prefix
    # dataset is the dataset to reindex

    for index, item in enumerate(dataset):
        dataset[index] = {'id':f'{prefix}{index}', **dataset[index]}
    
    return dataset

def reindex_profile(input_dataset):
    # reindex the profile of the input_dataset with the prefix and reorder the dict keys
    # input_dataset is the dataset to reindex

    for index, item in enumerate(input_dataset):
        profile = item['profile']
        for profile_index, profile_item in enumerate(profile):
            input_dataset[index]['profile'][profile_index]= {'id': f"{item['id']}{profile_index}", **input_dataset[index]['profile'][profile_index]}
    
    return input_dataset

def remove_canonical_id(dataset):
    # remove the canonical_id field from the dataset

    for index, item in enumerate(dataset):
        del dataset[index]['canonical_id']
        # remove canonical id in the 'profile' field
        profile = item['profile']
        for profile_index, profile_item in enumerate(profile):
            del dataset[index]['profile'][profile_index]['input']['canonical_id']
    
    return dataset


def create_correct_id(prefix, input_dataset, output_dataset):
    # create a correct id for each input and output dataset

    input_dataset = reindex(prefix, input_dataset)
    input_dataset = reindex_profile(input_dataset)
    output_dataset = reindex(prefix, output_dataset)  # there is no profile in the output dataset

    remove_canonical_id(input_dataset)

    return input_dataset, output_dataset


if __name__ == '__main__':


    # What I did because it seems to be what LaMP is doing (see page 16 of the paper):
    # Input/X: everything about the review and the user profile turned into a template prompt
    # Output/Y: review rating (or something else)
    # Profile: list of all the other (input, output) pairs of the same user (empty list if no other pairs). Note that the UP info in the input field are raw (i.e., not turned into a prompt). 

    # open csv file
    path_data = f'./TripAdvisor/Data/TA_final_dataset_EN_3K.csv'
    dataset_df = pd.read_csv(path_data, delimiter="\t", encoding="utf-8", )

    user_id_to_history = defaultdict(list)

    input_header = ['restaurant_reviewed_url', 'review_date', 'review_city', 'review_lang', 'review_title', 'review']
    user_profile_header = ['user_id_link', 'user_id', 'user_name', 'user_ta_level', 'user_age_range', 'user_sex', 'user_location', 'user_nb_contributions', 'user_nb_cities_visited', 'user_nb_helpful_votes', 'user_nb_photos', 'user_tags']
    output_header = ['review_rating']
    input_dataset = []
    output_dataset = []

    for index, review in dataset_df.iterrows():
        review_data = {key: review[key] for key in input_header}
        output = {key: review[key] for key in output_header}
        user_data = {key: review[key] for key in user_profile_header}

        # TODO: replace the value in the input field by the template prompt plus the review data and user data. --> refer to the LaMP paper to see the prompts/examples
        input_item = {'canonical_id': index, 'input': "HERE PUT PROMPT", 'review_data': review_data, 'user_data': user_data}   # user_data is in fact the user profile (UP); later we call profile the history
        
        input_dataset.append(input_item)
        output_dataset.append(output)
        
        input_item_copy = copy.deepcopy(input_item)
        output_item_copy = copy.deepcopy(output)

        user_id = review['user_id']
        user_id_to_history[user_id].append({'input': input_item_copy, 'output': output_item_copy})
    
    for index, review in dataset_df.iterrows():
        user_id = review['user_id']
        user_history = user_id_to_history[user_id]
        profile = [copy.deepcopy(item) for item in user_history if item['input']['canonical_id'] != index]
        input_dataset[index]['profile'] = profile


    # split the dataset into 3 parts: train, val (also called dev in LaMP), test

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
    create_correct_id("80", train_input_dataset, train_output_dataset)
    create_correct_id("81", val_input_dataset, val_output_dataset)
    create_correct_id("82", test_input_dataset, test_output_dataset)
    
    # save the 6 datasets in 6 files

    # we index it as 80 (8 for the index of the LaMP dataset in the benchmark and 0 because it is train)
    path_lamp_9_train_input = f'./TripAdvisor/Data/LaMP_9_3K_train_input.json'
    with open(path_lamp_9_train_input, 'w') as f:
        json.dump(train_input_dataset, f)

    # we index it as 80 (8 for the index of the LaMP dataset in the benchmark and 0 because it is train)
    path_lamp_9_train_output = f'./TripAdvisor/Data/LaMP_9_3K_train_output.json'
    with open(path_lamp_9_train_output, 'w') as f:
        json.dump({'task': "LaMP_9", 'golds': train_output_dataset}, f)

    # we index it as 81 (8 for the index of the LaMP dataset in the benchmark and 1 because it is val)
    path_lamp_9_val_input = f'./TripAdvisor/Data/LaMP_9_3K_val_input.json'
    with open(path_lamp_9_val_input, 'w') as f:
        json.dump(val_input_dataset, f)
    
    # we index it as 81 (8 for the index of the LaMP dataset in the benchmark and 1 because it is val)
    path_lamp_9_val_output = f'./TripAdvisor/Data/LaMP_9_3K_val_output.json'
    with open(path_lamp_9_val_output, 'w') as f:
        json.dump({'task': "LaMP_9", 'golds': val_output_dataset}, f)

    # we index it as 82 (8 for the index of the LaMP dataset in the benchmark and 2 because it is test)
    path_lamp_9_test_input = f'./TripAdvisor/Data/LaMP_9_3K_test_input.json'
    with open(path_lamp_9_test_input, 'w') as f:
        json.dump(test_input_dataset, f)

    # Note that this dataset file should not be made public
    # we index it as 82 (8 for the index of the LaMP dataset in the benchmark and 2 because it is test)
    path_lamp_9_test_output = f'./TripAdvisor/Data/LaMP_9_3K_test_output.json'
    with open(path_lamp_9_test_output, 'w') as f:
        json.dump({'task': "LaMP_9", 'golds': test_output_dataset}, f)