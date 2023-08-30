"""
File used to run the predictive model API on the LaMP_1 dataset and get the predictions.

The LLM API used can be chosen through the config.py file.

The different APIs that can be used are:
- OAI API (official)
- RevChatGPT V1 API (free)
- RevChatGPT V3 API (paying through secret OAI API key)
- Churchless API (free)

The LLM API response for a batch of input samples is processed, the predictions made are extracted and saved in a list.
The list of predictions is saved in a text file.

All the predictions in the text file are used to evaluate the model performance on the dataset (i.e., metrics are computed).

"""

from config import config, CONSTANTS as C

import os
import time
import re
import requests

import revChatGPT

from oai_api import get_response_from_oai
from revchatgpt_v1_api import chatbot as chatbot_v1, get_response_from_revchatgpt_v1
from revchatgpt_v3_api import chatbot as chatbot_v3, get_response_from_revchatgpt_v3
from churchless_api import get_response_from_churchless

from utils import wb_setup
from wandb.sdk.data_types import trace_tree

from utils import nb_tokens_in_string

from utils import setup_loguru
from loguru import logger
setup_loguru(logger)

import json

def extract_predictions(input_string):

    # Find the starting index of JSON string within the input
    start_index = input_string.find('{')
    # Find the ending index of JSON string within the input
    end_index = input_string.rfind('}') + 1

    # Extract the JSON string from the input
    json_string = input_string[start_index:end_index]

    try:
        # Load the JSON data
        data = json.loads(json_string)
        # Extract reference classes from the JSON data and return as a list
        reference_classes = [data[sample] for sample in data]
    except json.decoder.JSONDecodeError as e:
        logger.warning(e + "\nReturning None.")
        reference_classes = None

    return reference_classes

# def find_reference_class_in_string(input_string, batch_index, index_in_batch):
#     logger.info(f"Processing sample {index_in_batch} in batch {batch_index}...")

#     logger.debug("Sample: ", input_string)

#     pattern = r'\[[1-2]\]'  # pattern matches "[1]" or "[2]"

#     match = re.search(pattern, input_string)
#     if match:
#         start_index = match.start()
#         end_index = match.end()
#         print(f"Substring found at index {start_index}-{end_index}.")

#         reference_class = input_string[start_index:end_index]

#     else:
#         logger.warning(f"No reference_class found for sample {index_in_batch} in batch {batch_index}. Appending the reference class [-1].")
#         reference_class = "[-1]"

#     return reference_class

def append_predictions(saved_outputs, predictions, uids, evaluation_state, batch_index, nb_samples_in_batch):

    if not predictions:
        predictions = ["[-1]"] * nb_samples_in_batch
    
    for index_in_batch, reference_class in enumerate(predictions):

        logger.debug(f"\n\nWhile appending predictions\n predicted uids: {evaluation_state['predicted_uids']} \n batch index: {batch_index} \n all uids: {uids}\n uid index: {evaluation_state['uid_index']}")

        logger.debug(f"Getting uid predicted at position: {batch_index * len(predictions) + index_in_batch}")

        if reference_class in ["[1]", "[2]"]:
            saved_outputs.append(reference_class)
            evaluation_state['predicted_uids'].append(uids[batch_index * len(predictions) + index_in_batch])
        
        elif reference_class in [1, 2]:
            # wrap square brackets around the reference class
            reference_class = f"[{reference_class}]"
            saved_outputs.append(reference_class)
            evaluation_state['predicted_uids'].append(uids[batch_index * len(predictions) + index_in_batch])

        else:
            logger.warning(f"Model output for sample {index_in_batch} in batch {batch_index} is not a valid reference_class. Appending the reference class [-1].")
            saved_outputs.append("[-1]")
            evaluation_state['predicted_uids'].append(-1)

        evaluation_state['uid_index'] += 1

    return saved_outputs, evaluation_state


def get_predictions_from_api_output(saved_outputs, raw_output, uids, evaluation_state, batch_index, nb_samples_in_batch):
    # separate the different samples outputs for the input batch. We get a list of string outputs for each sample of the batch
    predictions = extract_predictions(raw_output)
    saved_outputs, evaluation_state = append_predictions(saved_outputs, predictions, uids, evaluation_state, batch_index, nb_samples_in_batch)
    return saved_outputs, evaluation_state


def run_api(total_nb_samples, uids, input_data_batches, saved_outputs, evaluation_state, retry_formatting=False):
    
    if config.API_CHOICE == "oai":
        logger.info("Starting to predict using the OAI API...")
    elif config.API_CHOICE == "V1":
        logger.info("Starting to predict using the RevChatGPT V1 API...")
        chatbot = chatbot_v1
    elif config.API_CHOICE == "V3":
        logger.info("Starting to predict using the RevChatGPT V3 API...")
        chatbot = chatbot_v3
    elif config.API_CHOICE == "churchless":
        logger.info("Starting to predict using the Churchless API...")
    
    already_predicted_uids = []

    if os.path.exists(f"./Experiment/Data/lamp_reformatted/LaMP_{config.lamp_dataset_index}/results/predicted_uids.txt"):
        # read in a list all the lines of the file ./Experiment/Data/lamp_reformatted/LaMP_{lamp_dataset_index}/results/predicted_uids.txt
        with open(f"./Experiment/Data/lamp_reformatted/LaMP_{config.lamp_dataset_index}/results/predicted_uids.txt", 'r') as f:
            already_predicted_uids = f.readlines()
            already_predicted_uids = [int(uid.strip()) for uid in already_predicted_uids if (uid.strip() != '-1' and uid.strip() != '')]
    

    for batch_index, input_batch in enumerate(input_data_batches):
        logger.debug(f"Input batch: \n {input_batch}")

        # format the input for the model
        input_batch = [f"sample_{index}: {text}" for index, text in enumerate(input_batch) if uids[batch_index * len(input_batch) + index] not in already_predicted_uids]

        nb_samples_in_batch = len(input_batch)
        logger.debug(f"Number of samples in batch: {len(input_batch)}")

        if nb_samples_in_batch == 0:
            continue

        input_batch_concatenated = "\n".join(input_batch)

        response = ""

        if nb_samples_in_batch == 1:
            full_prompt = f""". 
            You are given a sample for which you only have to predict [1] or [2].
            You have to respond in json format.
            The key of the json is sample_id and the associated value is the predicted reference class.
            The sample is contained in the triple backticks.
            ```{input_batch_concatenated}```
            """
        else:
            full_prompt = f"""
            You are given several samples for which you only have to predict [1] or [2].
            You have {nb_samples_in_batch} samples to predict. 
            The samples are separated by a newline.
            You have to respond in json format.
            The key of the json is sample_id and the associated value is the predicted reference class.
            Samples are contained in the triple backticks.
            ```{input_batch_concatenated}```
            """

        logger.info(f"Full prompt sent: \n {full_prompt}")
        
        if retry_formatting:
            wrong_formatting_attempts = 0

            while not(response.startswith("Sample")) or response.count("Sample") != nb_samples_in_batch:
                nb_tokens_in_llm_prompt = 0

                try:
                    if config.API_CHOICE == "oai":
                        response = get_response_from_oai(full_prompt)
                    elif config.API_CHOICE == "V1":
                        response = get_response_from_revchatgpt_v1(chatbot, full_prompt)
                    elif config.API_CHOICE == "V3":
                        response = get_response_from_revchatgpt_v3(chatbot, full_prompt)
                    elif config.API_CHOICE == "churchless":
                        response = get_response_from_churchless(full_prompt)

                    nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
                    nb_tokens_in_llm_response = nb_tokens_in_string(response, encoding_name="gpt-3.5-turbo")
                    logger.info(f"Number of tokens in llm prompt: {nb_tokens_in_llm_prompt}")
                    logger.info(f"Number of tokens in llm response: {nb_tokens_in_llm_response}")

                    llm_index_span = trace_tree.Span(name="llm", span_kind = trace_tree.SpanKind.TOOL)
                    wb_setup.root_span.add_child_span(llm_index_span)
                    llm_index_span.add_named_result({"llm input": full_prompt}, {"llm response": response})
                    llm_index_span.attributes = {"prediction format try n°": wrong_formatting_attempts+1}
                    llm_index_span.attributes = {"batch index": batch_index}
                    llm_index_span.attributes = {"nb of samples processed": evaluation_state['nb_samples_processed'] + nb_samples_in_batch}
                    
                    tokens_used = nb_tokens_in_llm_prompt + nb_tokens_in_llm_response
                    llm_index_span.attributes = {"token_usage_one_llm_exchange": tokens_used}

                except (requests.exceptions.HTTPError, revChatGPT.typings.Error) as e:
                    logger.warning(f"Exception caught: {e}")
                    logger.warning("Trying again to request the model API...")

                if not(response.startswith("Sample") and response.count("Sample") != nb_samples_in_batch):
                    if wrong_formatting_attempts >= 5:  # we set the predictions for this batch to a reference class [-1] for all samples in this batch
                        response = "\\n".join([f"Sample {id} reference class: [-1]" for id in range(nb_samples_in_batch)])
                    else:
                        logger.warning(f"Wrongly formatted model response: {response}")
                        logger.warning("Model did not follow the response format. Requesting the model API again...")

                    wrong_formatting_attempts += 1

        else:

            if config.API_CHOICE == "oai":
                response = get_response_from_oai(full_prompt)
            elif config.API_CHOICE == "V1":
                response = get_response_from_revchatgpt_v1(chatbot, full_prompt)
            elif config.API_CHOICE == "V3":
                response = get_response_from_revchatgpt_v3(chatbot, full_prompt)
            elif config.API_CHOICE == "churchless":
                response = get_response_from_churchless(full_prompt)

            nb_tokens_in_llm_prompt = nb_tokens_in_string(full_prompt, encoding_name="gpt-3.5-turbo")
            nb_tokens_in_llm_response = nb_tokens_in_string(response, encoding_name="gpt-3.5-turbo")
            logger.info(f"Number of tokens in llm prompt: {nb_tokens_in_llm_prompt}")
            logger.info(f"Number of tokens in llm response: {nb_tokens_in_llm_response}")
                
            llm_index_span = trace_tree.Span(name="llm", span_kind = trace_tree.SpanKind.TOOL)
            wb_setup.root_span.add_child_span(llm_index_span)
            llm_index_span.add_named_result({"llm input": full_prompt}, {"llm response": response})
        
            llm_index_span.attributes = {"batch index": batch_index}
            llm_index_span.attributes = {"nb of samples processed": evaluation_state['nb_samples_processed'] + nb_samples_in_batch}
                    
            tokens_used = nb_tokens_in_llm_prompt + nb_tokens_in_llm_response
            llm_index_span.attributes = {"token_usage_one_llm_exchange": tokens_used}
            
        logger.debug(f"Raw API response: {response}")

        saved_outputs, evaluation_state = get_predictions_from_api_output(saved_outputs, response, uids, evaluation_state, batch_index, nb_samples_in_batch)
    
        # explicit (linear) wait time to avoid OAI API rate limit
        sleep_time = (batch_index % 20) * 0.5
        time.sleep(sleep_time)
        
        evaluation_state['nb_samples_processed'] += nb_samples_in_batch

        logger.info(f"{evaluation_state['nb_samples_processed']} samples processed out of {total_nb_samples}")
        
    return saved_outputs, evaluation_state