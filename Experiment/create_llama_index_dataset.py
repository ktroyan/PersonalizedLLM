"""
Create a dataset augmented using llama-index. 
The original dataset must be a "reformatted" LaMP dataset (see file reformatted_lamp_dataset.py or folder /lamp_reformatted).

More particularly, additional context is given to input prompts (of the predictive model) by adding examples
of data from the same user profile and similar to the input data for which a prediction has to be made.

This file can be run as a script, or some of its functions are imported and used in the experiment.py script.
"""

import os
import time
import argparse

from config import config, CONSTANTS as C

import utils
from utils import wb_setup

import openai
import revChatGPT
from revChatGPT.V1 import Chatbot as ChatbotV1
from revChatGPT.V3 import Chatbot as ChatbotV3

from oai_api import get_response_from_oai
from revchatgpt_v1_api import get_response_from_revchatgpt_v1
from revchatgpt_v3_api import get_response_from_revchatgpt_v3
from churchless_api import get_response_from_churchless

import pandas as pd    
pd.options.mode.chained_assignment = None  # default='warn'

# import llama_index
# assert llama_index.__version__ >= "0.7.2", "Please ensure you are using llama-index v0.6.21 or higher in order to use WandB prompts."
from llama_index import Prompt, GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, LangchainEmbedding, VectorStoreIndex
from llama_index.llms.base  import CompletionResponse, LLMMetadata
from llama_index.llms.custom  import CustomLLM
from llama_index.indices.list import GPTListIndex, ListIndex
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT

from wandb.sdk.data_types import trace_tree
import wandb
assert wandb.__version__ >= "0.15.3", "Please ensure you are using wandb v0.15.3 or higher in order to use WandB prompts."

from typing import Optional, List, Mapping, Any
from IPython.display import Markdown, display

from loguru import logger
from utils import setup_loguru
setup_loguru(logger)

# define the llm to be used for llama-index; to be used if the official OAI API is not used
class MyLLM(CustomLLM):
    """Custom LLM to access any local model or model API."""

    def __init__(self):
        super().__init__()

        # set context window size
        self.context_window = 4096  # should match the model's
        # set number of output tokens
        self.num_output = 256

        if config.API_CHOICE == "V1":
            api_config={"access_token": config.OAI_ACCESS_TOKEN}
            self.chatbot = ChatbotV1(config=api_config)
        elif config.API_CHOICE == "V3":
            self.chatbot = ChatbotV3(api_key=openai.api_key,
                    engine = "gpt-3.5-turbo",
                    proxy = None,
                    timeout = None,
                    max_tokens = None,
                    temperature = 0.1,
                    top_p = 1.0,
                    presence_penalty = 0.0,
                    frequency_penalty = 0.0,
                    reply_count = 1,
                    system_prompt = "You are a state-of-the-art predictive model. You should only output predictions strictly respecting the output format."   # "You are ChatGPT, a large language model trained by OpenAI. Respond as concisely, straightforwardly and accurately as possible."
                    )
        elif config.API_CHOICE == "churchless":
            pass

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window, num_output=self.num_output
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)

        if config.API_CHOICE == "V1":
            response = get_response_from_revchatgpt_v1(self.chatbot, prompt)
        
        elif config.API_CHOICE == "V3":
            response = get_response_from_revchatgpt_v3(self.chatbot, prompt)

        elif config.API_CHOICE == "churchless":
            response = get_response_from_churchless(prompt)
                    
        return CompletionResponse(text=response)
    
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError()

def index_documents(dataset_name, service_context=None):
    
    # NOTE: here the files read should not contain the target (ground-truth) score of the input samples as it should not be accessible to llama-index (since we are not training anything)
    # input_files = [f"./Experiment/Data/lamp_reformatted/LaMP_3/LaMP_3_dataset_{dataset_name}_no_target.json"]  # here we can give a list of files on which we should create the llama index
    input_files = [f"./Experiment/Data/lamp_reformatted/LaMP_3/LaMP_3_dataset_{dataset_name}_subset_no_target.json"]  # here we can give a list of files on which we should create the llama index
    excluded_files = []

    # https://gpt-index.readthedocs.io/en/stable/getting_started/concepts.html
    # A Reader is a data connector that ingest data from different data sources and data formats into a simple Document representation (i.e., text and metadata)
    # A Document is a generic container around any data source. 
    # A Node is the atomic unit of data in Llamaindex and represents a "chunk" of a source Document. It’s a rich representation that includes metadata and relationships (to other nodes) to enable accurate and expressive retrieval operations
    documents = SimpleDirectoryReader(input_files=input_files, exclude=excluded_files).load_data()     # https://gpt-index.readthedocs.io/en/latest/reference/readers.html#llama_index.readers.SimpleDirectoryReader
    # logger.debug(documents)

    # https://gpt-index.readthedocs.io/en/stable/getting_started/concepts.html
    # Once data were ingested, LlamaIndex will help index the data into a format that’s easy to retrieve. 
    # Under the hood, LlamaIndex parses the raw documents into intermediate representations, calculates vector embeddings, and infers metadata. The most commonly used index is the VectorStoreIndex
    if config.API_CHOICE == "oai":
        documents_indexed = GPTVectorStoreIndex.from_documents(documents, openai_api_key=os.environ['OPENAI_API_KEY'], name="dataset")
    else:
        # documents_indexed = GPTListIndex.from_documents(documents, service_context=service_context)
        # documents_indexed = ListIndex.from_documents(documents, service_context=service_context)
        # documents_indexed = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        documents_indexed = VectorStoreIndex.from_documents(documents, service_context=service_context)

    return documents_indexed

def get_query_engine(documents_indexed):
    query_engine = documents_indexed.as_query_engine(text_qa_template=DEFAULT_TEXT_QA_PROMPT)     # TODO: here template not working?
    return query_engine

def llama_index_input_prompts(data_folder_path, dataset_df, dataset_name):
    # NOTE: llama-index
    # 1. Have "other" samples for each sample (user)
    # 2. Have an id for each sample (with the prefix being that of a user)
    # 3. Create a prompt for the llama-index LLM so that it is asking to find K relevant examples of samples to help make a prediction given the current sample.
    # 4. Make sure that llama-index does not have access to the actual sample's ground-truth score.
    # 5. Parse the outputted sample(s) and update the dataset with the new prompt that the predictive model will receive.

    # TODO: Question: Is it better to:
    # 1) create a llama-index on a newly created dataset file that would be a file containing the profile of a user? 
    # 2) OR create a llama-index on the whole dataset file
    # This choice comes from the fact that the profile of a user is typically large while the number of samples in a dataset is relatively not.

    # logger.info("Dataset: ", dataset_df, "\n")
    # logger.info("Dataset shape: ", dataset_df.shape, "\n")
    # logger.info("Dataset columns: ", dataset_df.columns())

    if config.API_CHOICE in ["V1", "V3", "churchless"]:
        # define the LLM for llama-index
        llamaindex_llm = MyLLM()

        service_context = ServiceContext.from_defaults(
            llm=llamaindex_llm, 
            context_window=llamaindex_llm.context_window, 
            num_output=llamaindex_llm.num_output
        )

        documents_indexed = index_documents(dataset_name, service_context)
    
    else:
        documents_indexed = index_documents(dataset_name)

    query_engine = get_query_engine(documents_indexed)

    for i, sample_prompt in enumerate(dataset_df['prompt']):    # iterate over the samples of the dataset

        logger.debug(f"Llama-index iteration over the sample: {i+1}/{len(dataset_df['prompt'])}")

        user_id = dataset_df['uid'][i]
        
        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Find and output another useful review from the user profile field of the same user to predict the score of the following review: {sample_prompt}
        # """

        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Given the review delimited by triple backticks, find and output a different review in the profile field of the same user.
        # The review you output should be similar to the review delimited by triple backticks as it should help in making a score prediction for it.
        # The review you output should be the review text and its associated score.
        # ```{sample_prompt}```
        # """

        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Given the review delimited by triple backticks, retrieve a different but similar review in the profile field of the same user.
        # The review you output should be similar to the review delimited by triple backticks as it should help in making a score prediction for it.
        # The review you output should be the review text and its associated score.
        # ```{sample_prompt}```
        # """

        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Given the review delimited by triple backticks, retrieve a different but similar review in the profile field of the same user.
        # The review you output should be the selected review text and its associated score.
        # ```{sample_prompt}```
        # """

        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Given the review delimited by triple backticks, retrieve a similar review in the profile field of the same user.
        # If and only if a useful review was found, your output should be the selected review text and its associated score.
        # If no useful review was found, output an empty string and nothing else.
        # ```{sample_prompt}```
        # """

        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Given the review delimited by triple backticks, retrieve a similar review in the profile field of the same user.
        # If a useful review was found, your output should be the selected review text and its associated score.
        # If no useful review was found, output "***" and nothing else.
        # ```{sample_prompt}```
        # """

        llama_prompt = f"""
        The current user has the id {user_id}. 
        Given the review delimited by triple backticks, retrieve a similar but different review in the profile field of the same user.
        If a useful review was found, your output should be the selected review text and its associated score.
        If no useful review was found, output "***" and nothing else.
        ```{sample_prompt}```
        """

        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Given the review delimited by triple backticks, retrieve different but similar reviews in the profile field of the same user.
        # The reviews you output should be similar to the review delimited by triple backticks as it should help in making a score prediction for it.
        # The reviews you output should be the reviews text and each of their associated score.
        # ```{sample_prompt}```
        # """

        # llama_prompt = f"""
        # The current user has the id {user_id}. 
        # Given the review delimited by triple backticks, retrieve a different but similar review in the profile field of the same user.
        # The review you output should be similar to the review delimited by triple backticks as it should help in making a score prediction for it.
        # The review you output should be in the format:
        # 'review_text': <review_text> \n
        # 'score': <score>.
        # ```{sample_prompt}```
        # """


        logger.debug(f"Llama-index prompt: {llama_prompt}")

        response = str(query_engine.query(llama_prompt))

        # logger.debug("[IN llama_index_input_prompts] Llama-index churchless response: \n" + response)

        nb_tokens_in_llama_prompt = utils.nb_tokens_in_string(llama_prompt, encoding_name="gpt-3.5-turbo")
        nb_tokens_in_llama_response = utils.nb_tokens_in_string(response, encoding_name="gpt-3.5-turbo")
        # logger.info("Number of tokens in llama-index prompt: ", nb_tokens_in_llama_prompt)
        # logger.info("Number of tokens in llama-index response: ", nb_tokens_in_llama_response)

        llama_index_span = trace_tree.Span(name="llama-index", span_kind = trace_tree.SpanKind.TOOL)
        wb_setup.root_span.add_child_span(llama_index_span)
        llama_index_span.add_named_result({"input": llama_prompt}, {"response": response})
        
        # add metadata to the span using .attributes
        tokens_used = nb_tokens_in_llama_prompt + nb_tokens_in_llama_response
        # tokens_used = nb_tokens_in_llama_response
        llama_index_span.attributes = {"token_usage_one_llama_exchange": tokens_used}

        if "***" in response:
            # logger.debug("Llama-index response was not useful for this sample. The prompt will be the same as the sample prompt.")
            dataset_df['prompt'][i] = sample_prompt
        else: 
            dataset_df['prompt'][i] = sample_prompt + "\nTo give more context, an example of correct prediction for this user is: " + response
        
        new_input_prompt = dataset_df['prompt'][i]

        # logger.debug("New input prompt: \n" + new_input_prompt)

        # Write in the /results folder in a txt file the prompt and response for each sample
        with open(data_folder_path + 'results/llama_iteration_' + str(i) + '.txt', 'w') as f:
            f.write('Sample user id:\n ' + str(user_id) + '\n\n\n')
            f.write('Sample prompt:\n ' + sample_prompt + '\n\n\n')
            f.write('Llama prompt:\n ' + llama_prompt + '\n\n\n')
            f.write('Llama response:\n ' + response + '\n\n\n')
            f.write('New input prompt:\n ' + new_input_prompt + '\n\n\n')

        # NOTE: Wait to avoid being rate limited by the API
        time.sleep((0.1*i) % 30)

    # logger.debug("\n\n --- END OF LLAMA INDEXING AND PROMPT AUGMENTATION --- \n\n")

    return dataset_df

def get_dataset_split(data_path, dataset_split_name):
    # dataset_split = pd.read_json(data_path + dataset_split_name + ".json", orient='records')
    dataset_split = pd.read_json(data_path + dataset_split_name + "_subset.json", orient='records')
    return dataset_split


class WBSetup(object):

    def __init__(self, use_wandb):
        self.use_wandb = use_wandb

    def create_trace(self,):

        # prepare wandb prompts
        root_span = trace_tree.Span(name="llama-index", span_kind=trace_tree.SpanKind.AGENT)   # root Span: create a span for high level agent
        trace = trace_tree.WBTraceTree(root_span)
        if use_wandb:
            wandb_run = wandb.init(project="wandb-prompts")
            self.wandb_run = wandb_run

        self.root_span = root_span
        self.trace = trace


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()

    args_parser.add_argument("--lamp_dataset_index", default="3", help="The index of the LaMP dataset to use. E.g., 1, 2, 3, etc.")
    args_parser.add_argument("--dataset_split_name", default="val", help="The name of the dataset split. I.e., train, val, test.")
    args_parser.add_argument("--use_wandb", action='store_true', default=False, help="Whether or not to use WandB prompt.")
    
    cmd_args = args_parser.parse_args()
    
    lamp_dataset_index = cmd_args.lamp_dataset_index
    dataset_split_name = cmd_args.dataset_split_name
    use_wandb = cmd_args.use_wandb

    # set the OAI API key in order to use the Store Index
    # NOTE: no better way?
    with open('./Experiment/oai_api_private_key.txt','r') as f:
        openai.api_key = f.read().replace('\n', '')
        config.OAI_API_KEY = openai.api_key
        os.environ['OPENAI_API_KEY'] = config.OAI_API_KEY

    # path to the folder containing the experiment data
    data_folder_path = f"./Experiment/Data/lamp_reformatted/LaMP_{lamp_dataset_index}/"

    if not os.path.exists(data_folder_path + "results"):
        os.makedirs(data_folder_path + "results")

    data_path = data_folder_path + f"LaMP_{lamp_dataset_index}_dataset_"

    # load the dataset
    val_dataset = get_dataset_split(data_path, dataset_split_name)
    
    # setup wandb
    # create an instance of the WBSetup class to be used across the experiment modules
    wb_setup = WBSetup(use_wandb)
    wb_setup.create_trace()

    # rewrite the prompts using llama-index
    llama_indexed_dataset_df = llama_index_input_prompts(data_folder_path, val_dataset, dataset_split_name)

    # save the dataset as json file
    llama_indexed_dataset_df.to_json(data_path + f"{dataset_split_name}_llama_indexed.json", orient='records')

    logger.info(f"New dataset created at {data_path + dataset_split_name}_llama_indexed.json \n")