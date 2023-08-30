import random
import string
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from webdriver_manager.chrome import ChromeDriverManager

# Get the webdriver object through the driver exe file and set its options
def get_driver():
    # load the chrome driver with options
    # chrome_driver_path = "./DataScraping/chromedriver_win32/chromedriver.exe"  # path to the chromedriver	
    
    chrome_options = Options()
    user_agent = "researcher" + ''.join(random.choices(string.ascii_lowercase, k=20))  # random user agent name
    chrome_options.add_argument(f'user-agent={user_agent}')
    # chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument('--load-extension=extension_3_4_4_0.crx')
    chrome_options.add_extension('./chromedriver_win32/istilldontcareaboutcookies-chrome-1.1.1_0.crx')     # to get a crx to load: https://techpp.com/2022/08/22/how-to-download-and-save-chrome-extension-as-crx/
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("disable-infobars")
    # chrome_options.add_argument(r'--profile-directory=Default')
    # chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument("--disable-dev-shm-usage")
    # chrome_options.add_argument("--headless")     # run the script without having a browser window open
    
    # driver = webdriver.Chrome(executable_path=chrome_driver_path, chrome_options=chrome_options)  # creates a web driver; general variable (will not be passed to a function)
    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chrome_options)   # this allows to manage better the chrome driver version (without having to download it manually and specify the path)
    
    # driver.maximize_window()

    return driver


# Print some information about the dataframe given
def print_dataframe_info(dataset_df, column_names):
    pd.set_option('display.max_columns', None)
    print("Dataset shape: ", dataset_df.shape)
    # print("A sample looks like: ", dataset_df.loc[0].head(dataset_df.shape[1]))
    # print("\n")

    print("Number of samples with at least one NaN value: ", len(dataset_df) - len(dataset_df.dropna()))
    print("Number of samples without NaN value in the dataset: ", len(dataset_df) - dataset_df.isnull().any(axis=1).sum())

    print("\n")

    print("Considering only the important columns/features from now.")
    print("Column names are: ", column_names)

    subset_df = dataset_df[column_names].copy()
    print("Number of samples with at least one NaN value in the important UP features: ", len(subset_df) - len(subset_df.dropna()))
    print("Number of samples without a NaN value in the important UP features: ", len(subset_df) - subset_df.isnull().any(axis=1).sum())
    
    # print("A sample with only the important features/columns looks like: ", subset_df.loc[0].head(subset_df.shape[1]))

    # for i, column_name in enumerate(column_names):
        # print(f"Number of rows in {column_name} WITH NaN value: ", len(dataset_df[dataset_df[column_name].isna()]))
        # print(f"Number of rows in {column_name} WITHOUT NaN value: ", len(dataset_df) - len(dataset_df[dataset_df[column_name].isna()]))
        # for j in range(i+1, len(column_names)):
            # print(f"Number of rows in {column_names[i]} AND {column_names[j]} WITH NaN value: ", len(dataset_df[dataset_df[column_names[i]].isna() & dataset_df[column_names[j]].isna()]))
            # print(f"Number of rows in {column_names[i]} AND {column_names[j]} WITHOUT NaN value: ", len(dataset_df) - len(dataset_df[dataset_df[column_names[i]].isna() & dataset_df[column_names[j]].isna()]))
    print("\n")


# Clean the dataset by removing duplicates; duplicates are detected considering only the two columns/features user_id_link and review
# Important NOTE: as the scraping was being done, a same user could be scraped several times. But since the samples were scraping at different
# moments in time, the UP data could be different (e.g., the user took posted more pictures). Therefore, we have to decide what to do with
# such samples (e.g., homogenize the UP data, or simply keep them with slightly different UP data in some columns/features) 
def remove_duplicated_samples(dataset):
    print("Removing duplicated samples...")
    print("Shape of uncleaned dataset: ", dataset.shape)
    dataset.drop_duplicates(subset = ['user_id_link', 'review'], keep=False, inplace=True)
    print("Shape of cleaned dataset: ", dataset.shape)
    return dataset

# Clean the dataset by removing samples that contain a review that has not been expanded
def remove_samples_unexpanded_review(dataset):
    print("Removing samples that contain a review that has not been expanded...")
    print("Shape of uncleaned dataset: ", dataset.shape)
    dataset['review_has_substring'] = dataset['review'].str.contains('...More')
    dataset.drop(dataset[dataset.review_has_substring == True].index, inplace=True)
    dataset.drop('review_has_substring', axis=1, inplace=True)
    print("Shape of cleaned dataset: ", dataset.shape)
    return dataset

# convert float values to int; values were scraped as float although they are int
def convert_float_to_int_in_columns(dataset, column_names):
    print("Converting float to int in columns: ", column_names)
    for column in column_names:
        dataset[column] = dataset[column].astype(int)
    return dataset

# replace NaN values by 0 for the relevant columns/features
def replace_nan_values_by_zero(dataset, column_names):
    print(f"Replacing NaN values by 0 for the relevant columns/features [{column_names}]...")
    for column in column_names:
        dataset[column].fillna(0, inplace=True)
    return dataset

def drop_unrequired_columns(dataset, columns_to_drop):
    # TODO: here, remove columns that are not allowed for privacy reasons (e.g., user_id_link, user_id, user_name, etc.)
    # Drop columns that should not be present in the final dataset
    print(f"Droping column(s) [{columns_to_drop}] that should not be part of the final dataset...")
    print("Shape of uncleaned dataset: ", dataset.shape)
    print("Columns of uncleaned dataset: ", dataset.columns)
    for column in columns_to_drop:
        # check if the column is present in the dataframe
        if column in dataset.columns:
            dataset.drop(columns=column, inplace=True)
    print("Shape of cleaned dataset: ", dataset.shape)
    return dataset

def remove_nan_valued_samples(dataset):
    print("Removing samples that still contain NaN values...")
    print("Shape of uncleaned dataset: ", dataset.shape)
    dataset.dropna(inplace=True)    # whether the the value is empty or has N/A or NaN value, the sample will be dropped
    print("Shape of cleaned dataset: ", dataset.shape)
    return dataset

# Create the final dataset to be used further in the project; essentially drop columns that are not needed, drop rows that contain NaN values, and name the file properly before saving it
def create_final_dataset(path_data, path_data_cleaned, partial_path_final_dataset, language_scraped):

    ## Drop columns that should not be present in the final dataset
    dataset = pd.read_csv(path_data, delimiter="\t", encoding="utf-8", )
    columns_to_drop = ['user_id_hash']
    clean_dataset = drop_unrequired_columns(dataset, columns_to_drop)
    clean_dataset.to_csv(path_data_cleaned, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    ## Remove duplicated samples in dataset
    dataset = pd.read_csv(path_data_cleaned, delimiter="\t", encoding="utf-8", )    # initially, load the untouched/uncleaned dataset
    clean_dataset = remove_duplicated_samples(dataset)
    clean_dataset.to_csv(path_data_cleaned, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file
    
    ## Remove samples with "...More" (i.e., review that did not get expanded) in the review text column of the dataset
    dataset = pd.read_csv(path_data_cleaned, delimiter="\t", encoding="utf-8", )
    clean_dataset = remove_samples_unexpanded_review(dataset)
    clean_dataset.to_csv(path_data_cleaned, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    ## TODO: transform the city names that were incorrectly saved due to scraping issues (see the Doc file)
    # ...

    ## Some columns/features that particularly matter (e.g., user tags) and that have NaN values should have their NaN values replaced by 0 IF relevant (e.g., nb_photos, nb_helpful_votes, etc.)
    dataset = pd.read_csv(path_data_cleaned, delimiter="\t", encoding="utf-8", )
    columns_to_nan_to_zero = ['user_nb_contributions', 'user_nb_cities_visited', 'user_nb_helpful_votes', 'user_nb_photos']     # columns/features for which we should replace NaN values by 0
    clean_dataset = replace_nan_values_by_zero(dataset, columns_to_nan_to_zero)
    clean_dataset.to_csv(path_data_cleaned, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    ## Some columns/features whose samples containing NaN values should be discarded 
    # Such columns/features are: user_ta_level   user_age_range	user_sex	user_location   user_tags restaurant_reviewed_url	review_date	review_city	review_lang	review_rating	review_title	review
    dataset = pd.read_csv(path_data_cleaned, delimiter="\t", encoding="utf-8", )
    clean_dataset = remove_nan_valued_samples(dataset)
    clean_dataset.to_csv(path_data_cleaned, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    ## Convert float values to int; values were scraped as float although they are int
    dataset = pd.read_csv(path_data_cleaned, delimiter="\t", encoding="utf-8", )
    float_to_int_columns = ['user_ta_level', 'user_nb_contributions', 'user_nb_cities_visited', 'user_nb_helpful_votes', 'user_nb_photos', "review_rating"]
    clean_dataset = convert_float_to_int_in_columns(dataset, float_to_int_columns)
    clean_dataset.to_csv(path_data_cleaned, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file

    ## Name the file properly before saving it
    dataset = pd.read_csv(path_data_cleaned, delimiter="\t", encoding="utf-8", )
    naming_k = str(len(dataset) // 1000) + "K"
    path_final_dataset = partial_path_final_dataset + f'_{naming_k}.csv'
    dataset.to_csv(path_final_dataset, sep="\t", encoding="utf-8", index=False)   # convert the pandas dataframe to a CSV file
    
    print("Final dataset shape: ", dataset.shape)

    # This yields the final dataset (and its path) that can be used further in the project
    return dataset, path_final_dataset


# Update scraped languages of TA_restaurants_urls.csv file according to the current state of the dataset for one language (e.g., "en")
def update_scraped_lang(df1, df1_col_name, df2, df2_col_name, updated_file_path, language_to_scrape):
    for row_value_df1 in df1[df1_col_name]:
        for index, row_value_df2 in enumerate(df2[df2_col_name]):
            if df2.scraped_lang.iloc[index] != df2.scraped_lang.iloc[index]:    # this checks if the value at df2.scraped_lang.iloc[index] is nan in pandas dataframe
                df2.scraped_lang.iloc[index] = ""
            if row_value_df1 == row_value_df2 and (language_to_scrape not in df2.scraped_lang.iloc[index].split(',')):
                df2.scraped_lang[index] = df2.scraped_lang.iloc[index] + "," + language_to_scrape   # update the scraped_lang column of df2
    
    # TODO: remove the first comma in the scraped_lang column

    df2.to_csv(updated_file_path, sep="\t", encoding="utf-8", index=False)  


# Convert a csv file to a xml file
def convert_csv_to_xml(path_dataset_csv, path_dataset_xml):
    dataset = pd.read_csv(path_dataset_csv, delimiter="\t", encoding="utf-8", )
    dataset.to_xml(path_dataset_xml, parser="etree", index=False)