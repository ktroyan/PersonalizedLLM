"""
For this free version, no access token or OAI API private key is required.

"""

from config import config, CONSTANTS as C

import re
import requests
import time

from utils import wb_setup
from wandb.sdk.data_types import trace_tree

from utils import nb_tokens_in_string

def separate_output(output):
    return output.split(",")

def find_score_in_string(sample, batch_index, index_in_batch):

    # print("Sample: ", sample)
    m1 = re.search(r':', sample)
    if m1:
        sample = sample[m1.start()+1:]    # could use m1.end()
        print("Sample transformed: ", sample)

    m = re.search(r"\d", sample)
    if m:
        score_index_in_string = m.start()
        review_score = int(float(sample[score_index_in_string]))

    else:
        print(f"No digit found for sample {index_in_batch} in batch {batch_index}. Appending a review score of -1.")
        review_score = -1

    return review_score

def append_predictions(saved_outputs, batch_output, uids, evaluation_state, batch_index):

    for index_in_batch, sample in enumerate(batch_output):
        review_score = find_score_in_string(sample, batch_index, index_in_batch)    # find the score for sample in the output batch
        
        if review_score in [-1, 1, 2, 3, 4, 5]:
            saved_outputs.append(review_score)
            evaluation_state['predicted_uids'].append(uids[evaluation_state['uid_index']])
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

def get_response(prompt):
    url = 'https://free.churchless.tech/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer MyDiscord'
    }

    messages = [{"role": "user", "content": prompt}]

    data = {"model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 1.0, }
    response = requests.post(url, headers=headers, json=data)

    return response

def run_api(total_nb_samples, uids, input_data_batches, saved_outputs, evaluation_state):
    
    retry_formatting = False
    
    for batch_index, input_batch in enumerate(input_data_batches):
        print("Input batch: \n", input_batch)

        # format the input for the model
        input_batch = [f"sample_{id}: {text}" for id, text in enumerate(input_batch)]
        input_batch_concatenated = "\n".join(input_batch)
        
        # print("Input: \n", input_batch_concatenated)
        nb_samples_in_batch = len(input_batch)

        output = ""

        full_prompt = f"""
        For each sample, predict one of the following rating score: 1, 2, 3, 4, 5. 
        You have {nb_samples_in_batch} samples to predict. 
        The samples are separated by a newline.
        You have to respond in json format.
        The key of the json is sample_id and the associated value is the predicted score.
        Samples are contained in the triple backticks.
        ```{input_batch_concatenated}```
        """

        # print("Full prompt sent: \n", full_prompt)
        
        if retry_formatting:
            wrong_formatting_attempts = 0

            while not(output.startswith("Sample")) or output.count("Sample") != nb_samples_in_batch:
                nb_tokens_in_llm_prompt = 0

                try:
                    output = get_response(full_prompt)

                    nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
                    nb_tokens_in_llm_response = nb_tokens_in_string(output, encoding_name="gpt-3.5-turbo")
                    print("Number of tokens in llama-index prompt: ", nb_tokens_in_llm_prompt)
                    print("Number of tokens in llama-index response: ", nb_tokens_in_llm_response)

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
                        output = "\\n".join([f"Sample {id} score: 3" for id in range(nb_samples_in_batch)])
                        print("Raw output: ", output)
                        print("\n")
                    else:
                        print("Wrongly formatted model output: ", output)
                        print("Model did not follow the output format. Requesting the model API again...")
                        print("\n")

                    wrong_formatting_attempts += 1

        else:
            response = get_response(full_prompt)
            response = response.json()
            print(response)

            if 'choices' not in response:
                print("Response does not contain a 'choices' attribute. Not able to get the output text.")
                output = response
            output = response['choices'][0]['message']['content']

            nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
            nb_tokens_in_llm_response = nb_tokens_in_string(output, encoding_name="gpt-3.5-turbo")
            print("Number of tokens in llama-index prompt: ", nb_tokens_in_llm_prompt)
            print("Number of tokens in llama-index response: ", nb_tokens_in_llm_response)
                
            llm_index_span = trace_tree.Span(name="llm", span_kind = trace_tree.SpanKind.TOOL)
            wb_setup.root_span.add_child_span(llm_index_span)
            llm_index_span.add_named_result({"llm input": full_prompt}, {"llm response": output})
        
            llm_index_span.attributes = {"batch index": batch_index}
            llm_index_span.attributes = {"nb of samples processed": evaluation_state['nb_samples_processed'] + nb_samples_in_batch}
                    
            tokens_used = nb_tokens_in_llm_prompt + nb_tokens_in_llm_response
            llm_index_span.attributes = {"token_usage_one_llm_exchange": tokens_used}
            
        print("Final raw output: ", output)

        saved_outputs, evaluation_state = get_predictions_from_api_output(saved_outputs, output, uids, evaluation_state, batch_index)
    
        # explicit (linear) wait time to avoid OAI API rate limit
        time.sleep(0.5 * batch_index)
        
        evaluation_state['nb_samples_processed'] += nb_samples_in_batch

        print(f"{evaluation_state['nb_samples_processed']} samples processed out of {total_nb_samples}\n")
        
    return saved_outputs, evaluation_state

