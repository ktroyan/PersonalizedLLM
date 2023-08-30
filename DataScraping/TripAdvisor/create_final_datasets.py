"""
Create the raw datasets. 
I.e., the datasets that are the result of the scraping process are minimally final (e.g., remove duplicated rows).

Create the final/final datasets.
I.e., N/A values are replaced by 0 when appropriate, samples with too many N/A values are removed, float values that are essentially int values are converted to int values, etc.

"""

import TA_utility
import pandas as pd

def create_raw_dataset(dataset_df):
    columns_to_drop = ['user_id_hash']
    dataset_df = TA_utility.drop_unrequired_columns(dataset_df, columns_to_drop)
    
    dataset_df = TA_utility.remove_duplicated_samples(dataset_df)
    
    clean_dataset = dataset_df.copy()
    return clean_dataset
    
def create_final_dataset(dataset_df):
    
    dataset_df = TA_utility.remove_samples_unexpanded_review(dataset_df)

    columns_to_nan_to_zero = ['user_nb_contributions', 'user_nb_cities_visited', 'user_nb_helpful_votes', 'user_nb_photos']     # columns/features for which we should replace NaN values by 0
    dataset_df = TA_utility.replace_nan_values_by_zero(dataset_df, columns_to_nan_to_zero)

    dataset_df = TA_utility.remove_nan_valued_samples(dataset_df)

    float_to_int_columns = ['user_ta_level', 'user_nb_contributions', 'user_nb_cities_visited', 'user_nb_helpful_votes', 'user_nb_photos', "review_rating"]
    dataset_df = TA_utility.convert_float_to_int_in_columns(dataset_df, float_to_int_columns)

    clean_dataset = dataset_df.copy()
    return clean_dataset


def main():
    
    # For English
    path_data_en = f'./DataScraping/TripAdvisor/Data/TA_data_en.csv'
    path_data_en_final = f'./DataScraping/TripAdvisor/Data/TA_data_en_final.csv'
    dataset_df_en = pd.read_csv(path_data_en, delimiter="\t", encoding="utf-8", )
    raw_dataset_en = create_raw_dataset(dataset_df_en)
    raw_dataset_en.to_csv(path_data_en, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file
    final_dataset_en = create_final_dataset(raw_dataset_en)
    final_dataset_en.to_csv(path_data_en_final, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    # For French
    path_data_fr = f'./DataScraping/TripAdvisor/Data/TA_data_fr.csv'
    path_data_fr_final = f'./DataScraping/TripAdvisor/Data/TA_data_fr_final.csv'
    dataset_df_fr = pd.read_csv(path_data_fr, delimiter="\t", encoding="utf-8", )
    raw_dataset_fr = create_raw_dataset(dataset_df_fr)
    raw_dataset_fr.to_csv(path_data_fr, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file
    final_dataset_fr = create_final_dataset(raw_dataset_fr)
    final_dataset_fr.to_csv(path_data_fr_final, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    # For Portuguese
    path_data_pt = f'./DataScraping/TripAdvisor/Data/TA_data_pt.csv'
    path_data_pt_final = f'./DataScraping/TripAdvisor/Data/TA_data_pt_final.csv'
    dataset_df_pt = pd.read_csv(path_data_pt, delimiter="\t", encoding="utf-8", )
    raw_dataset_pt = create_raw_dataset(dataset_df_pt)
    raw_dataset_pt.to_csv(path_data_pt, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file
    final_dataset_pt = create_final_dataset(raw_dataset_pt)
    final_dataset_pt.to_csv(path_data_pt_final, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    # For Spanish
    path_data_es = f'./DataScraping/TripAdvisor/Data/TA_data_es.csv'
    path_data_es_final = f'./DataScraping/TripAdvisor/Data/TA_data_es_final.csv'
    dataset_df_es = pd.read_csv(path_data_es, delimiter="\t", encoding="utf-8", )
    raw_dataset_es = create_raw_dataset(dataset_df_es)
    raw_dataset_es.to_csv(path_data_es, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file
    final_dataset_es = create_final_dataset(raw_dataset_es)
    final_dataset_es.to_csv(path_data_es_final, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    # For all the above languages together
    # take the minimum of samples per each dataset and concatenate all the final datasets and store them in a file at path "path_data_mix_final"
    path_data_mix_final = f'./DataScraping/TripAdvisor/Data/TA_data_mix_final.csv'
    min_samples = min(len(final_dataset_en), len(final_dataset_fr), len(final_dataset_pt), len(final_dataset_es))
    dataset_df_mix_final = pd.concat([final_dataset_en[:min_samples], final_dataset_fr[:min_samples], final_dataset_pt[:min_samples], final_dataset_es[:min_samples]], ignore_index=True)
    dataset_df_mix_final.to_csv(path_data_mix_final, sep='\t', encoding='utf-8', index=False)
    
    # Print some information about the English-only and final datasets
    TA_utility.print_dataframe_info(raw_dataset_en, column_names)
    TA_utility.print_dataframe_info(final_dataset_en, column_names)
    TA_utility.print_dataframe_info(dataset_df_mix_final, column_names)
   
    ## Convert each dataset to an XML file
    path_dataset_xml = f'./DataScraping/TripAdvisor/Data/TA_final_data_EN.xml'
    TA_utility.convert_csv_to_xml(path_data_en_final, path_dataset_xml)

    path_dataset_xml = f'./DataScraping/TripAdvisor/Data/TA_final_data_FR.xml'
    TA_utility.convert_csv_to_xml(path_data_fr_final, path_dataset_xml)

    path_dataset_xml = f'./DataScraping/TripAdvisor/Data/TA_final_data_PT.xml'
    TA_utility.convert_csv_to_xml(path_data_pt_final, path_dataset_xml)

    path_dataset_xml = f'./DataScraping/TripAdvisor/Data/TA_final_data_ES.xml'
    TA_utility.convert_csv_to_xml(path_data_es_final, path_dataset_xml)

    path_dataset_xml = f'./DataScraping/TripAdvisor/Data/TA_final_data_MIX.xml'
    TA_utility.convert_csv_to_xml(path_data_mix_final, path_dataset_xml)


if __name__ == '__main__':

    # path_dummy_file = f'./TripAdvisor/Data/dummy.csv'
    # dummy_df = pd.read_csv(path_dummy_file, delimiter="\t", encoding="utf-8", )

    # columns used as part of the collection criterion making up for "good" UPs
    column_names = ['user_age_range', 'user_sex', 'user_location', 'user_tags']

    main()