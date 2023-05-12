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

from revChatGPT.V3 import Chatbot


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
    val_dataset = pd.read_json(datasets_path + "val_subcopy.json", orient='records')
    test_dataset = pd.read_json(datasets_path + "test.json", orient='records')

    return train_dataset, val_dataset, test_dataset

def train(dataset_df):
    pass

def compute_metrics(predictions, gt_targets):
    
    print("Accuracy score: ", accuracy_score(gt_targets, predictions))
    # print("F1 score: ", f1_score(gt_targets, predictions, average='macro'))
    # print("Precision score: ", precision_score(gt_targets, predictions, average='macro'))
    # print("Recall score: ", recall_score(gt_targets, predictions, average='macro'))
    print("Mean Absolute Error (MAE): ", mean_absolute_error(gt_targets, predictions))
    print("Root Mean Squared Error (RMSE): ", mean_squared_error(gt_targets, predictions, squared=False))


def evaluate(dataset_df):

    print("We consider the following dataset/dataframe: \n")
    print(dataset_df.shape)
    print(dataset_df.head())

    outputs = []
    messages = []   # TODO: Prompt buffering needed? What when prompt gets too long (> 4000 tokens)? What is discarded? Concretely, should I append the prediction to "messages" to keep track of the "conversation"? Wouldn't this be like having the model remember the predictions for the previous samples?
    
    try:
        for i, input in enumerate(dataset_df['prompt']):
            print("Input: \n", input)

            messages = []   # remove this line if prompt buffering is needed
            task_context = "This is a score prediction task. Score predictions are 1, 2, 3, 4 or 5."
            task = {"role": "assistant", "content": task_context}
            messages.append(task)
            prompt_request = {"role": "user", "content": input}
            messages.append(prompt_request)


            # Here using: https://github.com/acheong08/ChatGPT
            # chatbot = Chatbot(api_key=openai.api_key)
            # chatbot.ask("Hello world")

            # Here using the official OpenAI API
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
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

    except openai.error.RateLimitError as e:    # OAI API rate limit is reached
        print("Exception: ", e)
        gt_targets = dataset_df['completion'].tolist()
        gt_targets = gt_targets[:i]
        print("GT targets: ", gt_targets)
        print("Predictions: ", outputs)

        print(f"Computing metrics for {i} samples...")
        compute_metrics(outputs, gt_targets)

        return gt_targets, outputs 

    gt_targets = dataset_df['completion'].tolist()
    print("GT targets: ", gt_targets)
    print("Predictions: ", outputs)

    print(f"Computing metrics for {len(gt_targets)} samples...")
    compute_metrics(outputs, gt_targets)

    return gt_targets, outputs


def predict(dataset_df):
    pass

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

    command_line_parser.add_argument("--OPENAI_API_KEY", type=str, default=None, help="A valid OpenAI API key.")
    command_line_parser.add_argument("--lamp_dataset_index", type=str, default=None, help="LaMP dataset index. E.g., 1, 2, 3, etc.")
    command_line_parser.add_argument("--lamp_8_samples_version", type=str, default=None, help="Rounded down in thousands number of samples for LaMP_8 dataset. E.g., 3K, 10K, etc.")
    command_line_parser.add_argument("--model_name", type=str, default="GPT-3.5-turbo", help="Model name (e.g., GPT-3.5-turbo, Flan-T5-base, etc.)")

    args = command_line_parser.parse_args()

    openai.api_key = args.OPENAI_API_KEY
    lamp_dataset_index = args.lamp_dataset_index
    lamp_8_samples_version = args.lamp_8_samples_version
    model_name = args.model_name

    # openai.api_key = os.getenv("OPENAI_API_KEY")

    datasets_path = get_path_to_data(lamp_dataset_index, lamp_8_samples_version)

    train_df, val_df, test_df = get_dataset_splits(datasets_path)

    # trained_model = train(train_df)

    gt_targets, outputs = evaluate(val_df)

    save_results_to_file(gt_targets, outputs)

    # predict(test_df)