import argparse
import pandas as pd    
import json

"""
This scripts takes a LaMP dataset in its official format and reformats it for it to later
be used for evaluations of models such as zero-shot GPT-3.5-turbo with Llama-index.
"""

def create_formatted_dataset(new_dataset_folder_path, new_dataset_name, input_df, output_df=None):

    samples = []

    with open(f'{new_dataset_folder_path}{new_dataset_name}', 'w') as reformatted_dataset_file:

        if output_df is not None:
            for sample_index in range(len(input_df)):
                
                uid = int(input_df.iloc[sample_index, 0])   # convert to int otherwise cannot dump json

                input_text = input_df.iloc[sample_index, 1]
                # print(input_text)
                
                profile_field = input_df.iloc[sample_index, 2]
                # print(profile_field)

                gold = output_df.iloc[sample_index, 1]
                # print(gold)
                target_output = gold['output']
                # print(target_output)
                samples.append({"uid": uid, "prompt": input_text, "completion": target_output, "profile": profile_field})

        elif output_df is None:   # test set has no output dataset as it is part of the private benchmark
            for sample_index in range(len(input_df)):
                
                uid = int(input_df.iloc[sample_index, 0])   # convert to int otherwise cannot dump json

                input_text = input_df.iloc[sample_index, 1]
                # print(input_text)
                
                profile_field = input_df.iloc[sample_index, 2]
                # print(profile_field)
                
                samples.append({"uid": uid, "prompt": input_text, "completion": "", "profile": profile_field})

        elif "no_target" in new_dataset_name:
            for sample_index in range(len(input_df)):
                
                uid = int(input_df.iloc[sample_index, 0])   # convert to int otherwise cannot dump json

                input_text = input_df.iloc[sample_index, 1]
                # print(input_text)
                
                profile_field = input_df.iloc[sample_index, 2]
                # print(profile_field)
                
                samples.append({"uid": uid, "prompt": input_text, "completion": "", "profile": profile_field})



        # for sample_index in range(len(input_df)):
            
        #     uid = int(input_df.iloc[sample_index, 0])   # convert to int otherwise cannot dump json

        #     input_text = input_df.iloc[sample_index, 1]
        #     # print(input_text)
            
        #     profile_field = input_df.iloc[sample_index, 2]
        #     # print(profile_field)

        #     if output_df is not None:
        #         gold = output_df.iloc[sample_index, 1]
        #         # print(gold)
        #         target_output = gold['output']
        #         # print(target_output)

        #         samples.append({"uid": uid, "prompt": input_text, "completion": target_output, "profile": profile_field})
        #     else:   # test set has no output dataset as it is part of the private benchmark
        #         samples.append({"uid": uid, "prompt": input_text, "completion": "", "profile": profile_field})

        #     if "no_target" in new_dataset_name:
        #         samples.append({"uid": uid, "prompt": input_text, "completion": "", "profile": profile_field})


        json.dump(samples, reformatted_dataset_file, indent=4)


def reformat_lamp_datasets(datasets, split_names, new_datasets_folder_path, new_datasets_name, create_subset=False, nb_samples_subset=10):

    k = nb_samples_subset

    # json to pandas dataframe
    train_input_df = pd.DataFrame.from_dict(datasets[split_names[0]])    # train_input
    train_output_df = pd.DataFrame.from_dict(datasets[split_names[1]])    # train_output

    val_input_df = pd.DataFrame.from_dict(datasets[split_names[2]])    # val_input
    val_output_df = pd.DataFrame.from_dict(datasets[split_names[3]])    # val_output

    test_input_df = pd.DataFrame.from_dict(datasets[split_names[4]])    # test_input

    if "LaMP_8" in new_datasets_name:
        test_output_df = pd.DataFrame.from_dict(datasets[split_names[5]])    # test_output
    else:
        test_output_df = None

    if create_subset:
        # only take the first K samples for each dataset
        train_input_df = train_input_df.iloc[:k, :]
        train_output_df = train_output_df.iloc[:k, :]
        val_input_df = val_input_df.iloc[:k, :]
        val_output_df = val_output_df.iloc[:k, :]
        test_input_df = test_input_df.iloc[:k, :]

        # only take the first K elements of the field profile for each dataset
        train_input_df['profile'] = train_input_df['profile'].apply(lambda x: x[:k])
        val_input_df['profile'] = val_input_df['profile'].apply(lambda x: x[:k])
        test_input_df['profile'] = test_input_df['profile'].apply(lambda x: x[:k])

        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train_subset.json', train_input_df, train_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val_subset.json', val_input_df, val_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test_subset.json', test_input_df, test_output_df) # LaMP_8 test output is available

        # set the completion field to empty string for each dataset
        train_input_df['completion'] = ""
        val_input_df['completion'] = ""
        test_input_df['completion'] = ""

        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train_subset_no_target.json', train_input_df, train_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val_subset_no_target.json', val_input_df, val_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test_subset_no_target.json', test_input_df, test_output_df) # LaMP_8 test output is available


    else:
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train.json', train_input_df, train_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val.json', val_input_df, val_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test.json', test_input_df, test_output_df) # LaMP_8 test output is available

        # set the completion field to empty string for each dataset
        train_input_df['completion'] = ""
        val_input_df['completion'] = ""
        test_input_df['completion'] = ""

        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_train_no_target.json', train_input_df, train_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_val_no_target.json', val_input_df, val_output_df)
        create_formatted_dataset(new_datasets_folder_path, f'{new_datasets_name}_test_no_target.json', test_input_df, test_output_df) # LaMP_8 test output is available

def get_datasets(data_folder_path, lamp_dataset_index, lamp_8_samples_version, print_info=False):

    if not lamp_dataset_index:
        print("Please provide the LaMP dataset index. Exiting...")
        quit()

    paths_data = []

    if lamp_dataset_index == "8":
        lamp_dataset_index = lamp_dataset_index + "_" + lamp_8_samples_version
        print(f"LaMP_8 version: LaMP_{lamp_dataset_index}")
        split_names = ['train_input', 'train_output', 'val_input', 'val_output', 'test_input', 'test_output']
        full_names = [full_name.split('_')[0] + "/" + "LaMP_8_" + lamp_8_samples_version + "_" + full_name for full_name in split_names]

    else:   # for all the other versions
        print(f"LaMP index: LaMP_{lamp_dataset_index}")
        split_names = ['train_questions', 'train_outputs', 'dev_questions', 'dev_outputs', 'test_questions']
        full_names = [full_name.split('_')[0] + "/" + full_name for full_name in split_names]


    for full_name, split_name in zip(full_names, split_names):
        original_dataset_path = f'{data_folder_path}lamp_original/LaMP_{lamp_dataset_index}/{full_name}.json'
        paths_data.append(original_dataset_path)
        print("Collecting dataset at path: ", original_dataset_path)

    # This is the final dataset that can be used further in the project
    datasets = {dataset_key: pd.read_json(path_data, orient='records') for path_data, dataset_key in zip(paths_data, split_names)}
    
    if print_info:
        print(datasets.keys())
        for key in datasets:
            print(datasets[key].shape)
            print(datasets[key].head())
            print("\n\n")

    formatted_dataset_folder_path = f'{data_folder_path}lamp_reformatted/LaMP_{lamp_dataset_index}/'
    formatted_dataset_name = f'LaMP_{lamp_dataset_index}_dataset'

    return datasets, split_names, formatted_dataset_folder_path, formatted_dataset_name


if __name__ == "__main__":
        
    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument("--lamp_dataset_index", type=str, default=None, help="LaMP dataset index. E.g., 1, 2, 3, etc.")
    command_line_parser.add_argument("--lamp_8_samples_version", type=str, default="3K", help="Rounded down in thousands number of samples for LaMP_8 dataset. E.g., 3K, 10K, etc.")
    command_line_parser.add_argument("--create_subset", action='store_true', default=False, help="Whether to create llamaindex-dataset subsets too for experimenting.")
    command_line_parser.add_argument("--nb_samples_subset", type=int, default=10, help="Number of samples in the subset to create.")


    args = command_line_parser.parse_args()

    lamp_dataset_index = args.lamp_dataset_index
    lamp_8_samples_version = args.lamp_8_samples_version
    create_subset = args.create_subset
    nb_samples_subset = args.nb_samples_subset

    data_folder_path = "./Experiment/Data/"

    datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name = get_datasets(data_folder_path, lamp_dataset_index, lamp_8_samples_version, print_info=False)

    print("Formatted datasets folder path: ", formatted_datasets_folder_path)
    print("Formatted datasets prefix name: ", formatted_datasets_name)

    reformat_lamp_datasets(datasets, split_names, formatted_datasets_folder_path, formatted_datasets_name, create_subset, nb_samples_subset)