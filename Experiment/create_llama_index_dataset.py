import os
import time
from config import config, CONSTANTS as C

import argparse

import pandas as pd    
pd.options.mode.chained_assignment = None  # default='warn'

import openai
import revChatGPT
from model_api_free_churchless import churchless_get_response
 
import utils
from utils import print_in_blue, print_in_green, print_in_red, print_in_yellow

# import llama_index
# assert llama_index.__version__ >= "0.7.2", "Please ensure you are using llama-index v0.6.21 or higher in order to use WandB prompts."
from llama_index import Prompt, GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, LangchainEmbedding, VectorStoreIndex
from llama_index.llms.base  import CompletionResponse, LLMMetadata
from llama_index.llms.custom  import CustomLLM
from llama_index.indices.list import GPTListIndex, ListIndex
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT

from utils import wb_setup
from wandb.sdk.data_types import trace_tree
import wandb
assert wandb.__version__ >= "0.15.3", "Please ensure you are using wandb v0.15.3 or higher in order to use WandB prompts."

from typing import Optional, List, Mapping, Any

from IPython.display import Markdown, display

# define the llm to be used for llama-index
class MyLLM(CustomLLM):
    """Custom LLM to access any (free) API."""

    def __init__(self):
        super().__init__()

        # set context window size
        self.context_window = 4096
        # set number of output tokens
        self.num_output = 256

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window, num_output=self.num_output
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)

        print_in_green(f"[IN class MyLLM] Llama-index prompt of length {prompt_length}: \n" + prompt)

        response = churchless_get_response(prompt)

        print_in_green("[IN class MyLLM] Churchless API response raw: \n" + str(response))

        response = response.json()

        if 'choices' not in response:
            print("Response does not contain a 'choices' attribute. Not able to get the output text.")
            print_in_red("[Error] = " + str(response))
            retries = 0
            while retries < 3:
                print("Retrying to get the response, retry ", retries+1, "/3 \n\n")
                response = churchless_get_response(prompt)
                response = response.json()
                if 'choices' in response:
                    break
                retries += 1
                time.sleep(0.5)

        output = response['choices'][0]['message']['content']
        
        print_in_yellow("[IN class MyLLM] Llama-index churchless response FINAL: \n" + output)

        return CompletionResponse(text=output)
    
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError()

def index_documents(dataset_name, service_context):
    
    # NOTE: here the files read should not contain the target (ground-truth) score of the samples as it should not be accessible to llama-index (since we are not training anything)
    # input_files = [f"./Experiment/Data/lamp_reformatted/LaMP_3/LaMP_3_dataset_{dataset_name}_no_target.json"]  # here we can give a list of files on which we should create the llama index
    input_files = [f"./Experiment/Data/lamp_reformatted/LaMP_3/LaMP_3_dataset_{dataset_name}_subset_no_target.json"]  # here we can give a list of files on which we should create the llama index
    excluded_files = []

    # https://gpt-index.readthedocs.io/en/stable/getting_started/concepts.html
    # A Reader is a data connector that ingest data from different data sources and data formats into a simple Document representation (i.e., text and metadata)
    # A Document is a generic container around any data source. 
    # A Node is the atomic unit of data in Llamaindex and represents a "chunk" of a source Document. It’s a rich representation that includes metadata and relationships (to other nodes) to enable accurate and expressive retrieval operations
    documents = SimpleDirectoryReader(input_files=input_files, exclude=excluded_files).load_data()     # https://gpt-index.readthedocs.io/en/latest/reference/readers.html#llama_index.readers.SimpleDirectoryReader
    # print(documents)

    # https://gpt-index.readthedocs.io/en/stable/getting_started/concepts.html
    # Once data were ingested, LlamaIndex will help index the data into a format that’s easy to retrieve. 
    # Under the hood, LlamaIndex parses the raw documents into intermediate representations, calculates vector embeddings, and infers metadata. The most commonly used index is the VectorStoreIndex
    # documents_indexed = GPTListIndex.from_documents(documents, service_context=service_context)
    # documents_indexed = ListIndex.from_documents(documents, service_context=service_context)
    # documents_indexed = GPTVectorStoreIndex.from_documents(documents, openai_api_key=os.environ['OPENAI_API_KEY'], name="dataset")
    # documents_indexed = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    documents_indexed = VectorStoreIndex.from_documents(documents, service_context=service_context)

    return documents_indexed

def get_query_engine(documents_indexed):
    query_engine = documents_indexed.as_query_engine(text_qa_template=DEFAULT_TEXT_QA_PROMPT)     # TODO: here template not working??
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
    # The first option would mean I would have to do it later in the program.
    # The second option means that I can do it here.

    print("Dataset: ", dataset_df, "\n")
    print("Dataset shape: ", dataset_df.shape, "\n")
    # print("Dataset columns: ", dataset_df.columns())

    # define the LLM for llama-index
    llamaindex_llm = MyLLM()

    service_context = ServiceContext.from_defaults(
        llm=llamaindex_llm, 
        context_window=llamaindex_llm.context_window, 
        num_output=llamaindex_llm.num_output
    )

    documents_indexed = index_documents(dataset_name, service_context)
    query_engine = get_query_engine(documents_indexed)

    print_in_blue("\n\n --- LLAMA INDEXING AND PROMPT AUGMENTATION --- \n\n")

    for i, sample_prompt in enumerate(dataset_df['prompt']):    # iterate over the samples of the dataset

        print_in_red(f"ITERATION OVER THE SAMPLES: {i+1}/{len(dataset_df['prompt'])}")

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


        print_in_blue("[IN llama_index_input_prompts] Llama-index prompt: \n" + llama_prompt)

        response = str(query_engine.query(llama_prompt))

        print_in_green("[IN llama_index_input_prompts] Llama-index churchless response: \n" + response)

        nb_tokens_in_llama_prompt = utils.nb_tokens_in_string(llama_prompt, encoding_name="gpt-3.5-turbo")
        nb_tokens_in_llama_response = utils.nb_tokens_in_string(response, encoding_name="gpt-3.5-turbo")
        print("Number of tokens in llama-index prompt: ", nb_tokens_in_llama_prompt)
        print("Number of tokens in llama-index response: ", nb_tokens_in_llama_response)

        llama_index_span = trace_tree.Span(name="llama-index", span_kind = trace_tree.SpanKind.TOOL)
        wb_setup.root_span.add_child_span(llama_index_span)
        llama_index_span.add_named_result({"input": llama_prompt}, {"response": response})
        
        # add metadata to the span using .attributes
        tokens_used = nb_tokens_in_llama_prompt + nb_tokens_in_llama_response
        # tokens_used = nb_tokens_in_llama_response
        llama_index_span.attributes = {"token_usage_one_llama_exchange": tokens_used}

        if "***" in response:
            print_in_red("Llama-index response was not useful for this sample. The prompt will be the same as the sample prompt.")
            dataset_df['prompt'][i] = sample_prompt
        else: 
            dataset_df['prompt'][i] = sample_prompt + "\nTo give more context, an example of correct prediction for this user is: " + response
        
        new_input_prompt = dataset_df['prompt'][i]

        print_in_green("New input prompt: \n" + new_input_prompt)

        # Write in the /results folder in a txt file the prompt and response for each sample
        with open(data_folder_path + 'results/llama_iteration_' + str(i) + '.txt', 'w') as f:
            f.write('Sample user id:\n ' + str(user_id) + '\n\n\n')
            f.write('Sample prompt:\n ' + sample_prompt + '\n\n\n')
            f.write('Llama prompt:\n ' + llama_prompt + '\n\n\n')
            f.write('Llama response:\n ' + response + '\n\n\n')
            f.write('New input prompt:\n ' + new_input_prompt + '\n\n\n')

        # NOTE: Wait to avoid being rate limited?
        time.sleep(0.1*i)

    print_in_blue("\n\n --- END OF LLAMA INDEXING AND PROMPT AUGMENTATION --- \n\n")

    return dataset_df

def get_dataset_split(data_path, dataset_split_name):
    # dataset_split = pd.read_json(data_path + "val.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    dataset_split = pd.read_json(data_path + "val_subset.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    return dataset_split

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()

    args_parser.add_argument("--lamp_dataset_index", default="3", help="The index of the LaMP dataset to use. E.g., 1, 2, 3, etc.")
    args_parser.add_argument("--dataset_split_name", default="val", help="The name of the dataset split. I.e., train, val, test.")
    
    cmd_args = args_parser.parse_args()
    
    lamp_dataset_index = cmd_args.lamp_dataset_index
    dataset_split_name = cmd_args.dataset_split_name

    # path to the folder containing the experiment data
    data_folder_path = f"./Experiment/Data/lamp_reformatted/LaMP_{lamp_dataset_index}/"

    if not os.path.exists(data_folder_path + "results"):
        os.makedirs(data_folder_path + "results")

    data_path = data_folder_path + f"LaMP_{lamp_dataset_index}_dataset_"

    # load the dataset
    val_dataset = get_dataset_split(data_path, dataset_split_name)
    
    # rewrite the prompts using llama-index
    llama_indexed_dataset_df = llama_index_input_prompts(data_folder_path, val_dataset, dataset_split_name)

    # save the dataset as json file
    llama_indexed_dataset_df.to_json(data_path + f"{dataset_split_name}_llama_indexed.json", orient='records')

    print("New dataset created? \n")