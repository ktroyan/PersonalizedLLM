import TA_utility
import pandas as pd
import argparse

def main():
    
    ## Print some information about the uncleaned dataset
    TA_utility.print_dataframe_info(dataset_df, column_names)

    ## Create the final dataset (processing steps involve cleaning, filtering, etc.) and save it to a file
    final_dataset, path_final_dataset = TA_utility.create_final_dataset(path_data, path_data_cleaned, partial_path_final_dataset, language_to_scrape)
   
    ## Print some information about the cleaned final dataset
    if args.print_info:
        final_dataset_df = pd.read_csv(path_final_dataset, delimiter="\t", encoding="utf-8", )
        TA_utility.print_dataframe_info(final_dataset_df, column_names)

    ## Update scraped languages of TA_restaurants_urls.csv file according to the current state of the dataset for one language (e.g., "en")
    if args.update_restaurants_urls_file:
        final_dataset_df = pd.read_csv(path_final_dataset, delimiter="\t", encoding="utf-8", )
        path_restaurants_urls_file = f'./TripAdvisor/Data/TA_restaurants_urls.csv'
        restaurants_urls_df = pd.read_csv(path_restaurants_urls_file, delimiter="\t", encoding="utf-8")
        TA_utility.update_scraped_lang(final_dataset_df, 'restaurant_reviewed_url', restaurants_urls_df, 'restaurant_url', path_restaurants_urls_file, language_to_scrape)

    ## Convert the dataset to an XML file
    if args.convert_to_xml:
        path_dataset_xml = f'./Data/TA_final_data_{language_to_scrape}.xml'
        TA_utility.convert_csv_to_xml(path_final_dataset, path_dataset_xml)


if __name__ == '__main__':

    command_line_parser = argparse.ArgumentParser()
    command_line_parser.add_argument("-lang", "--language", type=str, default="en", help="language of the scraped data (e.g., en, fr, pt, es, it, de, zhCN, etc.)")
    command_line_parser.add_argument("-pf", "--print_info", action='store_true', default=False, help="Whether or not to print info about the prepared dataset.")
    command_line_parser.add_argument("-uruf", "--update_restaurants_urls_file", action='store_true', default=False, help="Whether or not to update the associated (csv) file of restaurants urls by marking which restaurants have been newly scraped in a specific language.")
    command_line_parser.add_argument("-ctxml", "--convert_to_xml", action='store_true', default=False, help="Whether or not to convert the resulting dataset file to a XML file.")

    args = command_line_parser.parse_args()

    language_to_scrape = args.language

    path_data = f'./DataScraping/TripAdvisor/Data/TA_data_{language_to_scrape}.csv'
    path_data_cleaned = f'./DataScraping/TripAdvisor/Data/TA_cleaned_data_{language_to_scrape}.csv'
    partial_path_final_dataset = f'./DataScraping/TripAdvisor/Data/TA_final_dataset_{language_to_scrape.upper()}'
    
    dataset_df = pd.read_csv(path_data, delimiter="\t", encoding="utf-8", )

    # path_dummy_file = f'./TripAdvisor/Data/dummy.csv'
    # dummy_df = pd.read_csv(path_dummy_file, delimiter="\t", encoding="utf-8", )

    # columns used as part of the collection criterion making up for "good" UPs
    column_names = ['user_age_range', 'user_sex', 'user_location', 'user_tags']

    main()



    
                        