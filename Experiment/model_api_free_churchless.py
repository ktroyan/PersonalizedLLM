"""
For this free version, no access token or OAI API private key is required.

"""

from config import config, CONSTANTS as C

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

    # find the index of the first "```json" in the output
    # m = re.search(r"```json", output)
    m = re.search(r"{", output)
    if m:
        # output = output[m.end():]    # could use m.end()
        output = output[m.end()-1:]    # could use m.end()
        # print("Output transformed: ", output)
    
    # find the index of the "```" after the first "```json"
    # m = re.search(r"```", output)
    m = re.search(r"}", output)

    if m:
        # output = output[:m.end()]    # could use m.end()
        output = output[:m.end()+1]    # could use m.end()

    print_in_green("\nOutput transformed: \n" + output)

    output = output.split(",")
    return output

def find_score_in_string(sample, batch_index, index_in_batch):

    print(f"Processing sample {index_in_batch} in batch {batch_index}...")

    print("Sample: ", sample)

    m1 = re.search(r':', sample)
    if m1:
        sample = sample[m1.start()+1:]    # could use m1.end()
        # print("Sample transformed: ", sample)

    print("Current sample being processed to find the review score is: ", sample, "\n")

    m = re.search(r"\d", sample)
    if m:
        score_index_in_string = m.start()
        review_score = int(float(sample[score_index_in_string]))
        print(f"Review score collected: {review_score}")

    else:
        print(f"No review digit found for sample {index_in_batch} in batch {batch_index}. Appending a review score of -1.")
        review_score = -1
        print(f"Review score collected: {review_score}")

    return review_score

def append_predictions(saved_outputs, batch_output, uids, evaluation_state, batch_index):

    print("\nBatch output in [append_predictions]: ", batch_output)

    for index_in_batch, sample in enumerate(batch_output):
        print("Sample in [append_predictions]: ", sample)

        review_score = find_score_in_string(sample, batch_index, index_in_batch)    # find the score for sample in the output batch
        
        print(f"\n\nWhile appending predictions\n predicted uids: {evaluation_state['predicted_uids']} \n batch index: {batch_index} \n all uids: {uids}\n uid index: {evaluation_state['uid_index']}")

        if review_score in [-1, 1, 2, 3, 4, 5]:
            saved_outputs.append(review_score)
            predicted_id_index = batch_index * len(batch_output) + index_in_batch
            evaluation_state['predicted_uids'].append(uids[predicted_id_index])
        else:
            print(f"Model output for sample {index_in_batch} in batch {batch_index} is not a valid score. Appending a review score of -1.")
            saved_outputs.append(-1)
            evaluation_state['predicted_uids'].append(-1)

        evaluation_state['uid_index'] += 1

    return saved_outputs, evaluation_state


def get_predictions_from_api_output(saved_outputs, raw_output, uids, evaluation_state, batch_index):
    # separate the different samples outputs for the input batch. We get a list of string outputs for each sample of the batch
    batch_output = separate_output(raw_output)
    saved_outputs, evaluation_state = append_predictions(saved_outputs, batch_output, uids, evaluation_state, batch_index)
    return saved_outputs, evaluation_state

def churchless_get_response(prompt):
    print("Calling churchless_get_response...")
    url = 'https://free.churchless.tech/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ChatGPT-Hackers'
    }

    messages = [{"role": "user", "content": prompt}]

    data = {"model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.1, }
    response = requests.post(url, headers=headers, json=data)

    return response

def run_api(total_nb_samples, uids, input_data_batches, saved_outputs, evaluation_state):

    print_in_green("--- Entered in model_api_free_churchless.py ---")
    
    retry_formatting = False    # TODO: make it work for True

    already_predicted_uids = []

    if os.path.exists(f"./Experiment/Data/lamp_reformatted/LaMP_{config.lamp_dataset_index}/results/predicted_uids.txt"):
        # read in a list all the lines of the file ./Experiment/Data/lamp_reformatted/LaMP_{lamp_dataset_index}/results/predicted_uids.txt
        with open(f"./Experiment/Data/lamp_reformatted/LaMP_{config.lamp_dataset_index}/results/predicted_uids.txt", 'r') as f:
            already_predicted_uids = f.readlines()
            already_predicted_uids = [uid.strip() for uid in already_predicted_uids]
    
    print_in_green(f"\n\n ==========================Already predicted uids: {already_predicted_uids}")

    for batch_index, input_batch in enumerate(input_data_batches):
        print("Input batch: \n", input_batch)
        nb_samples_in_batch = len(input_batch)
        print("Number of samples in batch: ", nb_samples_in_batch)

        # format the input for the model
        input_batch = [f"sample_{index}: {text}" for index, text in enumerate(input_batch) if uids[batch_index * nb_samples_in_batch + index] not in already_predicted_uids]
        print([uids[batch_index * nb_samples_in_batch + index] for index, text in enumerate(input_batch) if uids[batch_index * nb_samples_in_batch + index] not in already_predicted_uids])
        
        input_batch_concatenated = "\n".join(input_batch)
        
        print_in_green(f"INPUT BATCH: {input_batch} \n\n ================================= \n\n")

        output = ""

        full_prompt = f"""
        For each sample, predict one of the following rating score: 1, 2, 3, 4, 5. 
        You have {nb_samples_in_batch} samples to predict. 
        The samples are separated by a newline.
        You have to respond in json format where the key of the json is sample_id and the associated value is the predicted score.
        Samples are contained in the triple backticks.
        ```{input_batch_concatenated}```
        """

        print("\n\nFull prompt sent: \n", full_prompt, "\n\n")
        
        if retry_formatting:
            wrong_formatting_attempts = 0

            while not(output.startswith("Sample")) or output.count("Sample") != nb_samples_in_batch:
                nb_tokens_in_llm_prompt = 0

                try:
                    response = churchless_get_response(full_prompt)
                    output = response.json()
                    output = output['choices'][0]['message']['content']
                    
                    nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
                    nb_tokens_in_llm_response = nb_tokens_in_string(output, encoding_name="gpt-3.5-turbo")
                    print("Number of tokens in llm prompt: ", nb_tokens_in_llm_prompt)
                    print("Number of tokens in llm response: ", nb_tokens_in_llm_response)

                    llm_index_span = trace_tree.Span(name="llm", span_kind = trace_tree.SpanKind.TOOL)
                    wb_setup.root_span.add_child_span(llm_index_span)
                    llm_index_span.add_named_result({"llm input": full_prompt}, {"llm response": output})
                    llm_index_span.attributes = {"prediction format try n°": wrong_formatting_attempts+1}
                    llm_index_span.attributes = {"batch index": batch_index}
                    llm_index_span.attributes = {"nb of samples processed": evaluation_state['nb_samples_processed'] + nb_samples_in_batch}
                    
                    tokens_used = nb_tokens_in_llm_prompt + nb_tokens_in_llm_response
                    llm_index_span.attributes = {"token_usage_one_llm_exchange": tokens_used}

                except (requests.exceptions.HTTPError) as e:
                    print("Exception caught: ", e)
                    print("Trying again to request the model API...")
                    print("\n")

                if output.startswith("Sample") and output.count("Sample") != nb_samples_in_batch:
                    print("Raw output: ", output)
                    print("\n")
                else:
                    if wrong_formatting_attempts >= 5:  # we set the predictions for this batch to 3 (median possible score)
                        output = "\\n".join([f"Sample {id} score: 3" for id in range(nb_samples_in_batch)]) # NOTE: assuming average score to be 3
                        print("Raw output: ", output)
                        print("\n")
                    else:
                        print("Wrongly formatted model output: ", output)
                        print("Model did not follow the output format. Requesting the model API again...")
                        print("\n")

                    wrong_formatting_attempts += 1

        else:
            response = churchless_get_response(full_prompt)
            response = response.json()
            
            print("LLM response: \n", response, "\n\n")

            if 'choices' not in response:
                print("Response does not contain a 'choices' attribute. Not able to get the output text.")
                output = response
            else:
                output = response['choices'][0]['message']['content']

            nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
            nb_tokens_in_llm_response = nb_tokens_in_string(output, encoding_name="gpt-3.5-turbo")
            print("Number of tokens in llm prompt: ", nb_tokens_in_llm_prompt)
            print("Number of tokens in llm response: ", nb_tokens_in_llm_response)
                
            llm_index_span = trace_tree.Span(name="llm", span_kind = trace_tree.SpanKind.TOOL)
            wb_setup.root_span.add_child_span(llm_index_span)
            llm_index_span.add_named_result({"llm input": full_prompt}, {"llm response": output})
        
            llm_index_span.attributes = {"batch index": batch_index}
            llm_index_span.attributes = {"nb of samples processed": evaluation_state['nb_samples_processed'] + nb_samples_in_batch}
                    
            tokens_used = nb_tokens_in_llm_prompt + nb_tokens_in_llm_response
            llm_index_span.attributes = {"token_usage_one_llm_exchange": tokens_used}
            
        print_in_green("\nRaw output in [get_predictions_from_api_output]: " + output + "\n")

        saved_outputs, evaluation_state = get_predictions_from_api_output(saved_outputs, output, uids, evaluation_state, batch_index)
    
        # explicit (linear) wait time to avoid OAI API rate limit
        time.sleep(0.5 * batch_index)
        
        evaluation_state['nb_samples_processed'] += nb_samples_in_batch

        print(f"{evaluation_state['nb_samples_processed']} samples processed out of {total_nb_samples}\n")
        
    return saved_outputs, evaluation_state

