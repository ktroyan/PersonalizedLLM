import TA_utility

import os
import time
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':

    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("--language", type=str, default="en", help="language of the scraped data (e.g., en, fr, pt, es, it, de, zhCN, etc.)")

    args = command_line_parser.parse_args()

    language_to_scrape = args.language

    path_data = f'./TripAdvisor/Data/TA_data_{language_to_scrape}.csv'
    path_cleaned_data = f'./TripAdvisor/Data/TA_cleaned_data_{language_to_scrape}.csv'
    path_data_xml = f'./Data/TA_cleaned_data_{language_to_scrape}.xml'
    
    dataset_df = pd.read_csv(path_data, delimiter="\t", encoding="utf-8", )

    # path_dummy_file = f'./TripAdvisor/Data/dummy.csv'
    # dummy_df = pd.read_csv(path_dummy_file, delimiter="\t", encoding="utf-8", )

    # columns used as part of the collection criterion making up for "good" UPs
    column_names = ['user_age_range', 'user_sex', 'user_location', 'user_tags']

    ## Print some information about the uncleaned dataset
    TA_utility.print_dataframe_info(dataset_df, column_names)

    ## Remove duplicated samples in dataset
    dataset = pd.read_csv(path_data, delimiter="\t", encoding="utf-8", )
    clean_dataset = TA_utility.remove_duplicated_samples(dataset)
    clean_dataset.to_csv(path_cleaned_data, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file
    
    ## Remove samples with "...More" (i.e., review that did not get expanded) in the review text column of the dataset
    dataset = pd.read_csv(path_cleaned_data, delimiter="\t", encoding="utf-8", )
    clean_dataset = TA_utility.remove_samples_unexpanded_review(dataset)
    clean_dataset.to_csv(path_cleaned_data, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    ## Create and save a dataset that is the cleaned dataset without any NaN values 
    # This is the final dataset that can be used further in the project
    dataset = pd.read_csv(path_cleaned_data, delimiter="\t", encoding="utf-8", )
    columns_to_drop = ['user_id_hash']
    final_dataset, path_final_dataset = TA_utility.create_final_dataset(dataset, language_to_scrape, columns_to_drop)
   
    ## Print some information about the cleaned final dataset
    # final_dataset_df = pd.read_csv(path_final_dataset, delimiter="\t", encoding="utf-8", )
    # TA_utility.print_dataframe_info(final_dataset_df, column_names)

    ## Update scraped languages of TA_restaurants_urls.csv file according to the current state of the dataset for one language (e.g., "en")
    # path_restaurants_urls_file = f'./TripAdvisor/Data/TA_restaurants_urls.csv'
    # restaurants_urls_df = pd.read_csv(path_restaurants_urls_file, delimiter="\t", encoding="utf-8")
    # TA_utility.update_scraped_lang(final_dataset_df, 'restaurant_reviewed_url', restaurants_urls_df, 'restaurant_url', path_restaurants_urls_file, language_to_scrape)



    
                        