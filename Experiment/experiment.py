import os
import sys
import time
import csv
import argparse
import pandas as pd    
import json

import openai

import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

from revChatGPT.V1 import Chatbot as ChatbotV1
from revChatGPT.V3 import Chatbot as ChatbotV3
import revChatGPT

import re

import requests


def get_path_to_data(lamp_dataset_index, lamp_8_samples_version):

    if not lamp_8_samples_version:
        dataset_version = "LaMP_" + lamp_dataset_index
    else:
        dataset_version = "LaMP_" + lamp_dataset_index + "_" + lamp_8_samples_version
    
    data_folder_path = "./Experiment/Data/OAI/"
    datasets_path = data_folder_path + dataset_version + "_dataset_"

    return datasets_path

def get_dataset_splits(datasets_path):
    train_dataset = pd.read_json(datasets_path + "train.json", orient='records')
    val_dataset = pd.read_json(datasets_path + "val_subcopy.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    test_dataset = pd.read_json(datasets_path + "test.json", orient='records')

    return train_dataset, val_dataset, test_dataset

def train(dataset_df):
    pass

def compute_metrics(gt_targets, predictions):
    
    print("Accuracy score: ", accuracy_score(gt_targets, predictions))
    # print("F1 score: ", f1_score(gt_targets, predictions, average='macro'))
    # print("Precision score: ", precision_score(gt_targets, predictions, average='macro'))
    # print("Recall score: ", recall_score(gt_targets, predictions, average='macro'))
    print("Mean Absolute Error (MAE): ", mean_absolute_error(gt_targets, predictions))
    print("Root Mean Squared Error (RMSE): ", mean_squared_error(gt_targets, predictions, squared=False))

def separate_output(output):
    return output.split("\n")   # TODO: maybe here there is an issue for V1

def find_score_in_string(result):

    # m = re.search(r"\d", result)
    # if m:
    #     score_index_in_string = m.start()
    #     review_score = int(float(result[score_index_in_string]))
    #     print("Prediction: ", review_score)
    # else:
    #     print("No digit found. Appending a review score of -1.")
    #     review_score = -1

    review_score = result[-1]
        
    return review_score

def append_predictions(outputs, batch_output):
    for output_index in range(len(batch_output)):
        review_score = find_score_in_string(batch_output[output_index])    # find the score in the output
        outputs.append(review_score) 
    return outputs

def get_predictions_from_api_output(outputs, output):
    # separate the different samples outputs for the input batch. We get a list of string outputs for each sample of the batch
    separated_batch_output = separate_output(output)

    outputs = append_predictions(outputs, separated_batch_output)

    return outputs

def predict(dataset_df, evaluation_state):
    print("We consider the following dataset/dataframe: ")
    print(dataset_df.shape)
    print(dataset_df.head())

    total_nb_samples = len(dataset_df['prompt'])

    outputs = []

    # messages = []   # TODO: Prompt buffering needed? What when prompt gets too long (> 4000 tokens)? What is discarded? Concretely, should I append the prediction to "messages" to keep track of the "conversation"? Wouldn't this be like having the model remember the predictions for the previous samples?

    request_batch_size = 5
    print("Using batches of size ", request_batch_size, " for the API requests.")

    # batch the samples to reduce the total number of API requests to perform
    input_data_batches = [dataset_df['prompt'][i:i+request_batch_size] for i in range(0,len(dataset_df['prompt']), request_batch_size)]

    if api_choice == "official":

        task_context = "This is a score prediction task. Score predictions are 1, 2, 3, 4 or 5."
        task = {"role": "assistant", "content": task_context}

        for i, input_batch in enumerate(input_data_batches):
            # print("Input: \n", input_batch)

            # format the input for the model
            input_batch = [f"sample {id}: {value}" for id, value in enumerate(input_batch)]
            input_batch_concatenated = "\n".join(input_batch)
            
            # print("Input: \n", input_batch)
                
            messages = []   # remove this line if prompt buffering is needed

            messages.append(task)
            prompt_request = {"role": "user", "content": input_batch_concatenated}
            messages.append(prompt_request)

            response = openai.ChatCompletion.create(model=model_name, 
                                                    messages = messages,
                                                    temperature=.5,
                                                    max_tokens=500,
                                                    top_p=1,
                                                    frequency_penalty=0,
                                                    presence_penalty=0
                                                    )
            # print("Raw prediction: ", response)

            output = response.choices[0].message.content
            print("Prediction: ", output)

            # messages.append({"role": "assistant", "content": output})   # uncomment if prompt buffering is needed
            
            outputs.append(int(float(output)))


    elif api_choice == "free":  # Here using free API: https://github.com/acheong08/ChatGPT
        
        if free_api_version == "V1":
            config={
            "access_token": OAI_ACCESS_TOKEN
            }

            # revChatGPT V1
            chatbot = ChatbotV1(config=config, 
                                )
            
            for i, input_batch in enumerate(input_data_batches):
                # print("Input: \n", input_batch)

                # format the input for the model
                input_batch = [f"sample {id}: {value}" for id, value in enumerate(input_batch)]
                input_batch_concatenated = "\n".join(input_batch)
                
                # print("Input: \n", input_batch_concatenated)
                nb_samples_in_batch = len(input_batch)
                

                output = ""
                
                wrong_formatting_attempts = 0

                while not(output.startswith("Sample")) or output.count("Sample") != nb_samples_in_batch:

                    full_prompt = f"This is a score prediction task. Score predictions are 1, 2, 3, 4 or 5. For each sample, you have to respond with a score. There are           {request_batch_size} samples to predict for. They are separated by a newline. The output format for a sample has to be: Sample <id> score: <score_value>\\n. Here are the samples: {input_batch_concatenated}"

                    try:
                        output_data_full = chatbot.ask(full_prompt)
                        for data in output_data_full:
                            output = data['message']
                    except (requests.exceptions.HTTPError, revChatGPT.typings.Error) as e:
                        print("Exception caught: ", e)
                        print("Trying again to request the model API...")

                    if output.startswith("Sample") and output.count("Sample") != nb_samples_in_batch:
                        print("Raw prediction: ", output)
                    
                    else:
                        if wrong_formatting_attempts > 3:  # we set the predictions for this batch to 3(median possible score)
                            output = "\\n".join([f"Sample {id} score: 3" for id in range(request_batch_size)])
                            print("Raw prediction: ", output)
                        else:
                            print("Wrongly formatted model output: ", output)
                            print("Model did not follow the output format. Requesting the model API again...")
                        
                        wrong_formatting_attempts += 1

                outputs = get_predictions_from_api_output(outputs, output)
            
                # explicit (linear) wait time to avoid OAI API rate limit
                time.sleep(0.1 * i)
                
                evaluation_state['nb_samples_processed'] += nb_samples_in_batch

                print(f"{evaluation_state['nb_samples_processed']} samples processed out of {total_nb_samples}")
        
        elif free_api_version == "V3":

            # revChatGPT V3
            chatbot = ChatbotV3(api_key=openai.api_key,
                                engine = model_name,
                                proxy = None,
                                timeout = None,
                                max_tokens = None,
                                temperature = 0.5,
                                top_p = 1.0,
                                presence_penalty = 0.0,
                                frequency_penalty = 0.0,
                                reply_count = 1,
                                system_prompt = "" #"You are a state-of-the-art predictive model. You will receive data samples produce some output prediction. Respect the output format."   # "You are ChatGPT, a large language model trained by OpenAI. Respond as concisely, straightforwardly and accurately as possible."
                                )

            for i, input_batch in enumerate(input_data_batches):
                # print("Input: \n", input_batch)

                nb_samples_in_batch = len(input_batch)

                # format the input for the model
                input_batch = [f"sample {id}: {value}" for id, value in enumerate(input_batch)]
                input_batch_concatenated = "\n".join(input_batch)
                
                # print("Input: \n", input_batch_concatenated)

                output = ""

                wrong_formatting_attempts = 0
                
                while not(output.startswith("Sample")) or output.count("Sample") != nb_samples_in_batch:

                    full_prompt = f"This is a score prediction task. Score predictions are 1, 2, 3, 4 or 5. For each sample, you have to respond with a score. There are           {request_batch_size} samples to predict for. They are separated by a newline. The output format for a sample has to be: Sample <id> score: <score_value>\\n. Here are the samples: {input_batch_concatenated}"

                    output = chatbot.ask(full_prompt)

                    if output.startswith("Sample") and output.count("Sample") != nb_samples_in_batch:
                        print("Raw prediction: ", output)
                    
                    else:
                        if wrong_formatting_attempts > 3:  # we set the predictions for this batch to 3(median possible score)
                            output = "\\n".join([f"Sample {id} score: 3" for id in range(request_batch_size)])
                            print("Raw prediction: ", output)
                        else:
                            print("Wrongly formatted model output: ", output)
                            print("Model did not follow the output format. Requesting the model API again...")
                        
                        wrong_formatting_attempts += 1

                    time.sleep(1 * i)
        
                outputs = get_predictions_from_api_output(outputs, output)
            
                # explicit (linear) wait time to avoid OAI API rate limit
                time.sleep(0.1 * i)
                
                evaluation_state['nb_samples_processed'] += nb_samples_in_batch

                print(f"{evaluation_state['nb_samples_processed']} samples processed out of {total_nb_samples}")

    return outputs
            
def evaluate(dataset_df, test_mode=False):

    evaluation_state = {'nb_samples_processed': 0}
    try:
        predictions = predict(dataset_df, evaluation_state)

    except (openai.error.RateLimitError, revChatGPT.typings.APIConnectionError) as e:    # OAI API rate limit is reached
        print("Exception: ", e)

        print(f"Predictions for {evaluation_state['nb_samples_processed']} samples: {predictions}")

        if not test_mode:
            gt_targets = dataset_df['completion'].tolist()
            gt_targets = gt_targets[:evaluation_state['nb_samples_processed']]
            print("GT targets: ", gt_targets)

            print(f"Computing metrics for {evaluation_state['nb_samples_processed']} samples...")
            compute_metrics(gt_targets, predictions)

        else:
            print("Test mode. Hence, no ground truth targets available (None).)")
            gt_targets = None
        
        return gt_targets, predictions 

    print("Predictions: ", predictions)
    
    if not test_mode:
        gt_targets = dataset_df['completion'].tolist()
        print("GT targets: ", gt_targets)

        print(f"Computing metrics for {len(gt_targets)} samples...")
        compute_metrics(gt_targets, predictions)

    else:
        print("Test mode. Hence, no ground truth targets available (None).)")
        gt_targets = None
        
    return gt_targets, predictions


def save_results_to_file(gt_targets, outputs):

    # convert the lists of integers to lists of strings
    iterator_gt_targets = map(lambda integer_value: str(integer_value), gt_targets)
    gt_targets = list(iterator_gt_targets)

    iterator_outputs = map(lambda integer_value: str(integer_value), outputs)
    outputs = list(iterator_outputs)

    # save the ground truth targets and the predictions in files
    with open('./Experiment/Data/OAI/gt_targets.txt','w') as f:
        f.write('\n'.join(gt_targets))

    with open('./Experiment/Data/OAI/predictions.txt','w') as f:
        f.write('\n'.join(outputs))

if __name__ == '__main__':

    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument("--USE_OAI_ACCESS_TOKEN", action='store_true', default=False, help="If there is a valid access token to OpenAI API in a local .txt file. Works for V1")
    command_line_parser.add_argument("--USE_OAI_API_KEY", action='store_true', default=False, help="If there is a valid OpenAI API key in a local .txt file.")
    command_line_parser.add_argument("--API_CHOICE", type=str, default="free", help="The API to use. I.e.: official or free.")
    command_line_parser.add_argument("--FREE_API_VERSION", type=str, default="V3", help="The version of the free API to use. I.e.: V1 or V3.")

    command_line_parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Model name (e.g., GPT-3.5-turbo, Flan-T5-base, etc.)")

    command_line_parser.add_argument("--lamp_dataset_index", type=str, default=None, help="LaMP dataset index. E.g., 1, 2, 3, etc.")
    command_line_parser.add_argument("--lamp_8_samples_version", type=str, default=None, help="Rounded down in thousands number of samples for LaMP_8 dataset. E.g., 3K, 10K, etc.")

    args = command_line_parser.parse_args()

    if args.USE_OAI_ACCESS_TOKEN:
        with open('./Experiment/oai_api_access_token.txt','r') as f:
            OAI_ACCESS_TOKEN = f.read().replace('\n', '')

    if args.USE_OAI_API_KEY:
            # read the oai API key from the text file
            with open('./Experiment/oai_api_private_key.txt','r') as f:
                openai.api_key = f.read().replace('\n', '')
    
    api_choice = args.API_CHOICE
    print(f"Currently using the {api_choice} OAI API.")

    if api_choice == "free":
        free_api_version = args.FREE_API_VERSION
        print(f"Currently using the version {free_api_version} of the free OAI API.")
    
    lamp_dataset_index = args.lamp_dataset_index
    lamp_8_samples_version = args.lamp_8_samples_version
    model_name = args.model_name

    datasets_path = get_path_to_data(lamp_dataset_index, lamp_8_samples_version)

    train_df, val_df, test_df = get_dataset_splits(datasets_path)

    # trained_model = train(train_df)

    gt_targets, outputs = evaluate(val_df)

    save_results_to_file(gt_targets, outputs)

    # predict(test_df)