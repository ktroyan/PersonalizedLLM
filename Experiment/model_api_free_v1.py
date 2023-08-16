"""
For this free version, an access token or OAI API private key is required.

"""

from config import config, CONSTANTS as C

import revChatGPT
from revChatGPT.V1 import Chatbot as ChatbotV1

import os
import re
import requests
import time

from utils import wb_setup
from wandb.sdk.data_types import trace_tree

from utils import nb_tokens_in_string
from utils import print_in_green, print_in_red

import json

def separate_output(output):

    # # find the index of the first "```json" in the output
    # # m = re.search(r"```json", output)
    # m = re.search(r"{", output)
    # if m:
    #     # output = output[m.end():]    # could use m.end()
    #     output = output[m.end()-1:]    # could use m.end()
    #     # print("Output transformed: ", output)
    
    # # find the index of the "```" after the first "```json"
    # # m = re.search(r"```", output)
    # m = re.search(r"}", output)

    # if m:
    #     # output = output[:m.end()]    # could use m.end()
    #     output = output[:m.end()+1]    # could use m.end()

    print_in_green("\nOutput transformed: \n" + output)

    output = output.split(",")
    return output

def find_score_in_string(sample, batch_index, index_in_batch):
    print(f"Processing sample {index_in_batch} in batch {batch_index}...")

    print("Sample: ", sample)

    m1 = re.search(r':', sample)
    if m1:
        sample = sample[m1.start()+1:]    # could use m1.end()
        print("Sample transformed: ", sample)

    print("Current sample being processed to find the review score is: ", sample, "\n")

    m = re.search(r"\d", sample)
    if m:
        score_index_in_string = m.start()
        review_score = int(float(sample[score_index_in_string]))
        print(f"Review score collected: {review_score}")

    else:
        print(f"No digit found for sample {index_in_batch} in batch {batch_index}. Appending a review score of -1.")
        review_score = -1
        print(f"Review score collected: {review_score}")

    return review_score

def extract_scores(input_string):

    print_in_red("\n\n[IN extract_scores] INPUT string: \n" + input_string + "\n\n")

    # Find the starting index of JSON string within the input
    start_index = input_string.find('{')
    # Find the ending index of JSON string within the input
    end_index = input_string.rfind('}') + 1

    # Extract the JSON string from the input
    json_string = input_string[start_index:end_index]

    print_in_red("\n\n[IN extract_scores] JSON string: \n" + json_string + "\n\n")

    try:
        # Load the JSON data
        data = json.loads(json_string)
        # Extract scores from the JSON data and return as a list
        scores = [data[sample] for sample in data]
    except json.decoder.JSONDecodeError:
        print_in_red("\n\n[IN extract_scores] JSONDecodeError caught. Returning a list of -1.\n\n")
        scores = None

    return scores

def append_predictions(saved_outputs, raw_output, uids, evaluation_state, batch_index, nb_samples_in_batch):
    
    predictions = extract_scores(raw_output)

    if not predictions:
        predictions = [-1] * nb_samples_in_batch
    
    print_in_green("\n\nPredictions: \n" + str(predictions) + "\n\n")

    for index_in_batch, review_score in enumerate(predictions):

        print(f"\n\nWhile appending predictions\n predicted uids: {evaluation_state['predicted_uids']} \n batch index: {batch_index} \n all uids: {uids}\n uid index: {evaluation_state['uid_index']}")

        if review_score in [-1, 1, 2, 3, 4, 5]:
            saved_outputs.append(review_score)
            print(f"Getting uid to predict at position: {batch_index * len(predictions) + index_in_batch}")
            evaluation_state['predicted_uids'].append(uids[batch_index * len(predictions) + index_in_batch])
        else:
            print(f"Model output for sample {index_in_batch} in batch {batch_index} is not a valid score. Appending a review score of -1.")
            saved_outputs.append(-1)
            evaluation_state['predicted_uids'].append(-1)

        evaluation_state['uid_index'] += 1

    return saved_outputs, evaluation_state


def get_predictions_from_api_output(saved_outputs, raw_output, uids, evaluation_state, batch_index, nb_samples_in_batch):

    # separate the different samples outputs for the input batch. We get a list of string outputs for each sample of the batch
    saved_outputs, evaluation_state = append_predictions(saved_outputs, raw_output, uids, evaluation_state, batch_index, nb_samples_in_batch)
    return saved_outputs, evaluation_state

def free_v1_get_response(chatbot, prompt):
    output_data_full = chatbot.ask(prompt)
    for data in output_data_full:
        output = data['message']

    return output

def run_api(total_nb_samples, uids, input_data_batches, saved_outputs, evaluation_state):
    print_in_green("--- Entered in model_api_free_v1.py ---")
    
    api_config={
    "access_token": config.OAI_ACCESS_TOKEN
    }

    # revChatGPT V1
    chatbot = ChatbotV1(config=api_config)
    
    retry_formatting = False

    for batch_index, input_batch in enumerate(input_data_batches):
        print("Input batch: \n", input_batch)
        print("Number of samples in batch: ", len(input_batch))

        # format the input for the model
        input_batch = [f"sample_{id}: {text}" for id, text in enumerate(input_batch)]
        input_batch_concatenated = "\n".join(input_batch)
        
        # print("Input: \n", input_batch_concatenated)
        nb_samples_in_batch = len(input_batch)

        response = ""

        full_prompt = f"""
        For each sample, predict one of the following rating score: 1, 2, 3, 4, 5. 
        You have {nb_samples_in_batch} samples to predict. 
        The samples are separated by a newline.
        You have to respond in json format.
        The key of the json is sample_id and the associated value is the predicted score.
        Samples are contained in the triple backticks.
        ```{input_batch_concatenated}```
        """

        # full_prompt = f"""
        # For each sample, predict one of the following rating score: 1, 2, 3, 4, 5. 
        # You have {nb_samples_in_batch} samples to predict. 
        # The samples are separated by a newline.
        # You have to respond in json format where the key of the json is sample_id and the associated value is the predicted score.
        # Samples are contained in the triple backticks.
        # ```{input_batch_concatenated}```
        # """

        print_in_green("\n\nFull prompt sent: \n" + full_prompt + "\n\n")
        
        if retry_formatting:
            wrong_formatting_attempts = 0

            while not(response.startswith("Sample")) or response.count("Sample") != nb_samples_in_batch:
                nb_tokens_in_llm_prompt = 0

                try:
                    response = free_v1_get_response(chatbot, full_prompt)

                    nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
                    nb_tokens_in_llm_response = nb_tokens_in_string(response, encoding_name="gpt-3.5-turbo")
                    print("Number of tokens in llm prompt: ", nb_tokens_in_llm_prompt)
                    print("Number of tokens in llm response: ", nb_tokens_in_llm_response)

                    llm_index_span = trace_tree.Span(name="llm", span_kind = trace_tree.SpanKind.TOOL)
                    wb_setup.root_span.add_child_span(llm_index_span)
                    llm_index_span.add_named_result({"llm input": full_prompt}, {"llm response": response})
                    llm_index_span.attributes = {"prediction format try n°": wrong_formatting_attempts+1}
                    llm_index_span.attributes = {"batch index": batch_index}
                    llm_index_span.attributes = {"nb of samples processed": evaluation_state['nb_samples_processed'] + nb_samples_in_batch}
                    
                    tokens_used = nb_tokens_in_llm_prompt + nb_tokens_in_llm_response
                    llm_index_span.attributes = {"token_usage_one_llm_exchange": tokens_used}

                except (requests.exceptions.HTTPError, revChatGPT.typings.Error) as e:
                    print("Exception caught: ", e)
                    print("Trying again to request the model API...")
                    print("\n")

                if response.startswith("Sample") and response.count("Sample") != nb_samples_in_batch:
                    print("Raw output: ", response)
                    print("\n")
                else:
                    if wrong_formatting_attempts >= 5:  # we set the predictions for this batch to 3 (median possible score)
                        response = "\\n".join([f"Sample {id} score: 3" for id in range(nb_samples_in_batch)])
                        print("Raw response: ", response)
                        print("\n")
                    else:
                        print("Wrongly formatted model response: ", response)
                        print("Model did not follow the response format. Requesting the model API again...")
                        print("\n")

                    wrong_formatting_attempts += 1

        else:
            response = free_v1_get_response(chatbot, full_prompt)

            nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
            nb_tokens_in_llm_response = nb_tokens_in_string(response, encoding_name="gpt-3.5-turbo")
            print("Number of tokens in llm prompt: ", nb_tokens_in_llm_prompt)
            print("Number of tokens in llm response: ", nb_tokens_in_llm_response)
                
            llm_index_span = trace_tree.Span(name="llm", span_kind = trace_tree.SpanKind.TOOL)
            wb_setup.root_span.add_child_span(llm_index_span)
            llm_index_span.add_named_result({"llm input": full_prompt}, {"llm response": response})
        
            llm_index_span.attributes = {"batch index": batch_index}
            llm_index_span.attributes = {"nb of samples processed": evaluation_state['nb_samples_processed'] + nb_samples_in_batch}
                    
            tokens_used = nb_tokens_in_llm_prompt + nb_tokens_in_llm_response
            llm_index_span.attributes = {"token_usage_one_llm_exchange": tokens_used}
            
        print("\nRaw output in [get_predictions_from_api_output]: ", response, "\n")

        saved_outputs, evaluation_state = get_predictions_from_api_output(saved_outputs, response, uids, evaluation_state, batch_index, nb_samples_in_batch)
    
        # explicit (linear) wait time to avoid OApredicted_uidsI API rate limit
        time.sleep(0.5 * batch_index)
        
        evaluation_state['nb_samples_processed'] += nb_samples_in_batch

        print(f"{evaluation_state['nb_samples_processed']} samples processed out of {total_nb_samples}\n")
        
    return saved_outputs, evaluation_state