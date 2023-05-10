import time
import argparse
import pandas as pd

def compute_stats(dataset_df):
    print("Number of samples with (at least one) NaN value: ", len(dataset_df) - len(dataset_df.dropna()), "\n")

    # compute statistics for each column of the dataframe
    for column_name in dataset_df.columns:
        print("\nStatistics for column/feature: ", column_name)
        print(dataset_df[column_name].describe())

    # print the unique elements of each column with categorical values
    print("\nUnique values for each column with categorical values:\n")

    for column_name in ['user_ta_level', 'user_age_range', 'user_sex', 'user_location', 'user_nb_contributions', 'user_nb_cities_visited', 'user_nb_helpful_votes', 'user_nb_photos', 'user_tags', 'restaurant_reviewed_url', 'review_date', 'review_city', 'review_lang', 'review_rating']:
            if column_name == 'user_nb_contributions' or column_name == 'user_nb_cities_visited' or column_name == 'user_nb_helpful_votes' or column_name == 'user_nb_photos':
                print(f"\n{column_name} has max value {max(dataset_df[column_name].unique())} and min value {min(dataset_df[column_name].unique())} in this dataset.")
            elif column_name == 'user_tags':
                unique_tags_combination = dataset_df[column_name].unique()
                unrolled_tags = ";".join(unique_tags_combination).split(';')
                unique_tags = set(unrolled_tags)
                print(f"\n{column_name} has {len(unique_tags)} unique tags in this dataset.")
                print("The unique tags are the following: ", unique_tags)
                      
            elif column_name == 'restaurant_reviewed_url':
                unique_restaurants_urls = dataset_df[column_name].unique()
                print("\nThe number of different restaurants in this dataset is: ", len(unique_restaurants_urls))

            elif column_name == 'review_date':
                unique_review_dates = dataset_df[column_name].unique()
                date_years = [int(date[-4:]) for date in unique_review_dates]
                print(f"The oldest review year date is {min(date_years)} and the most recent review date year is {max(date_years)}.")

            elif column_name == 'user_location':
                unique_locations = dataset_df[column_name].unique()
                print("\n", column_name, " has ", len(unique_locations), " unique values in this dataset which are: ", unique_locations)

                
            else:
                print("\n", column_name, " has the following unique values in this dataset: ", dataset_df[column_name].unique())


    # compute statistics for the number of samples per user
    print("\nStatistics for the number of samples per user:\n")
    print(dataset_df.groupby('user_id').size().describe())

    # compute statistics for the number of samples per restaurant
    print("\nStatistics for the number of samples per restaurant:\n")
    print(dataset_df.groupby('restaurant_reviewed_url').size().describe())

    # compute statistics for the number of samples per city
    print("\nStatistics for the number of samples per city:\n")
    print(dataset_df.groupby('review_city').size().describe())

    # compute statistics for the number of samples per language
    print("\nStatistics for the number of samples per language:\n")
    print(dataset_df.groupby('review_lang').size().describe())

    # compute statistics for the number of samples per rating
    print("\nStatistics for the number of samples per rating:\n")
    print(dataset_df.groupby('review_rating').size().describe())

    # compute statistics for the number of samples per date
    print("\nStatistics for the number of samples per date:\n")
    print(dataset_df.groupby('review_date').size().describe())

    # compute the average review text length
    print("\nAverage review length: ", dataset_df['review'].str.len().mean(), " characters.")

    # compute the average review text length per language
    # ...





if __name__ == "__main__":
    
    # start time when running the script
    start_time = time.time()

    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("--dataset_version", type=str, default="EN_3K", help="language of the scraped data and rounded down number of samples in thousands. E.g., EN_10K, FR_3K, etc.")

    args = command_line_parser.parse_args()

    dataset_version = args.dataset_version

    path_dataset = f'./TripAdvisor/Data/TA_final_dataset_{dataset_version}.csv'
    dataset_df = pd.read_csv(path_dataset, delimiter="\t", encoding="utf-8", )

    compute_stats(dataset_df)

    # time spent for the full scraping run
    end_time = time.time()
    print("Time elapsed to compute stats: ", int(end_time - start_time) // 60, " minutes and ", int(end_time - start_time) % 60, " seconds")
