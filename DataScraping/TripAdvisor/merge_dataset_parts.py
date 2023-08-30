"""
This script merges two TripAdvisor datasets in one dataset. 
The two datasets are the result of two different scraping runs performed using different data (e.g., the cities to scrape).
It also merges the associates restaurants_urls files.
"""

import os 
import pandas as pd
import argparse

def merge_datasets():
    data_file_name = f"TA_data_{language}.csv"
    
    data_part2_folder_path = "./DataScraping/TripAdvisor/Data/data_part2/"
    data_part2_file_name = f"TA_data_{language}_part2.csv"
    
    print(os.path.join(data_folder_path, data_file_name))
    print(os.path.join(data_part2_folder_path, data_part2_file_name))

    df = pd.read_csv(data_folder_path + data_file_name, delimiter="\t", encoding="utf-8")
    print("First dataframe read successfully.")
    df_part2 = pd.read_csv(data_part2_folder_path + data_part2_file_name, delimiter="\t", encoding="utf-8")
    print("Second dataframe read successfully.")

    print(df.head())
    print(df_part2.head())

    print(df.shape)
    print(df_part2.shape)

    df_merged = pd.concat([df, df_part2], ignore_index=True, axis = 0)

    print(df_merged.head())
    print(df_merged.shape)

    new_data_file_name = f"TA_data_{language}_merged.csv"

    df_merged.to_csv(data_folder_path + new_data_file_name, sep="\t", index=False, encoding="utf-8")

    return df_merged

def merge_restaurants_urls_files():
    restaurants_urls_folder_path = "./DataScraping/TripAdvisor/Data/"
    restaurants_urls_file_name = f"TA_restaurants_urls.csv"

    restaurants_urls_part2_folder_path = "./DataScraping/TripAdvisor/Data/data_part2/"
    restaurants_urls_part2_file_name = f"TA_restaurants_urls_part2.csv"

    df_restaurants_urls = pd.read_csv(restaurants_urls_folder_path + restaurants_urls_file_name, delimiter="\t", encoding="utf-8")
    print("First restaurants_urls dataframe read successfully.")
    df_restaurants_urls_part2 = pd.read_csv(restaurants_urls_part2_folder_path + restaurants_urls_part2_file_name, delimiter="\t", encoding="utf-8")
    print("Second restaurants_urls dataframe read successfully.")

    print(df_restaurants_urls.head())
    print(df_restaurants_urls_part2.head())

    print(df_restaurants_urls.shape)
    print(df_restaurants_urls_part2.shape)

    df_restaurants_urls_merged = pd.concat([df_restaurants_urls, df_restaurants_urls_part2], ignore_index=True, axis = 0)

    print(df_restaurants_urls_merged.head())
    print(df_restaurants_urls_merged.shape)

    new_restaurants_urls_file_name = f"TA_restaurants_urls_merged.csv"
    df_restaurants_urls_merged.to_csv(data_folder_path + new_restaurants_urls_file_name, sep="\t", index=False, encoding="utf-8")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--language", type=str, default="en", help="Language of the dataset.")

    args = argparser.parse_args()

    language = args.language

    data_folder_path = "./DataScraping/TripAdvisor/Data/"

    # merge the two datasets
    merge_datasets(data_folder_path)

    # merge the two associated restaurants_urls.csv files
    merge_restaurants_urls_files(data_folder_path)