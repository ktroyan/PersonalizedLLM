import time
from config import config, CONSTANTS as C

import os
import sys

import pandas as pd    
pd.options.mode.chained_assignment = None  # default='warn'

import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

import openai
import revChatGPT

import utils
import model_api_oai
import model_api_free_v1
import model_api_free_v3
import model_api_free_churchless

# import llama_index
# assert llama_index.__version__ >= "0.7.2", "Please ensure you are using llama-index v0.6.21 or higher in order to use WandB prompts."
from llama_index import Prompt, GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, LangchainEmbedding
from llama_index.llms.base  import CompletionResponse, LLMMetadata
from llama_index.llms.custom  import CustomLLM
from llama_index.indices.list import GPTListIndex, ListIndex

from utils import wb_setup
from wandb.sdk.data_types import trace_tree
import wandb
assert wandb.__version__ >= "0.15.3", "Please ensure you are using wandb v0.15.3 or higher in order to use WandB prompts."

import torch
from transformers import pipeline
from typing import Optional, List, Mapping, Any

from IPython.display import Markdown, display
import requests
from llama_index.prompts.default_prompts import  ( DEFAULT_TEXT_QA_PROMPT )



def get_dataset_splits(data_folder_path, dataset_version):
    data_path = data_folder_path + dataset_version
    # train_dataset = pd.read_json(data_path + "_dataset_" + "train.json", orient='records')
    # val_dataset = pd.read_json(data_path + "_dataset_" + "val.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    val_dataset = pd.read_json(data_path + "_dataset_" + "val_subset.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    test_dataset = pd.read_json(data_path + "_dataset_" + "test.json", orient='records')

    train_dataset = None

    return train_dataset, val_dataset, test_dataset

def train(dataset_df):
    """
    Note that training/fine-tuning is not done for models such as GPT-3.5-turbo.
    """
    raise NotImplementedError

def compute_metrics(gt_targets, predictions):
    
    print("Accuracy score: ", accuracy_score(gt_targets, predictions))
    print("Mean Absolute Error (MAE): ", mean_absolute_error(gt_targets, predictions))
    print("Root Mean Squared Error (RMSE): ", mean_squared_error(gt_targets, predictions, squared=False))
    # print("F1 score: ", f1_score(gt_targets, predictions, average='macro'))
    # print("Precision score: ", precision_score(gt_targets, predictions, average='macro'))
    # print("Recall score: ", recall_score(gt_targets, predictions, average='macro'))

def batch_samples(dataset_df):
    print("Using batches of size ", config.request_batch_size, " for the API requests.")

    # batch the samples to reduce the total number of API requests to perform
    uids = dataset_df['uid']
    input_data_batches = [dataset_df['prompt'][i:i+config.request_batch_size] for i in range(0,len(dataset_df['prompt']), config.request_batch_size)]
    return uids, input_data_batches


def churchless_get_response(prompt):
    print("Calling churchless_get_response...")
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


class MyLLM(CustomLLM):
    """Custom LLM to access any (free) API."""

    def __init__(self):
        super().__init__()

        # set context window size
        self.context_window = 2048
        # set number of output tokens
        self.num_output = 256

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window, num_output=self.num_output
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Wait 3s to avoid being rate limited:
        time.sleep(3)
        prompt_length = len(prompt)
        response = churchless_get_response(prompt)
        response = response.json()
        # print("Churchless API response raw = ", response)

        if 'choices' not in response:
            print("Response does not contain a 'choices' attribute. Not able to get the output text.")
            # print response in red:
            print("\033[91m" + "\n[Error] = " + str(response) + "\033[0m")
            retries = 1
            while retries >= 3:
                print("Retrying to get the response, retry ", retries, "/3")
                response = churchless_get_response(prompt)
                response = response.json()
                if 'choices' in response:
                    break
                retries += 1

        output = response['choices'][0]['message']['content']
        # print Hello in gray:
        print("\033[90m" + "\n[Prompt] = " + prompt + "\033[0m")
        # print("\nPrompt = ", prompt)


        print("\nChurchless API response text = ", output, "\n")

        # Write in the /results folder in a txt file the prompt and response for each sample,
        # and create the folder if it doesnt exist:
        if not os.path.exists('./Experiment/results'):
            os.makedirs('./Experiment/results')
        # write in a new file for each sample, with incrementing number:
        with open('./Experiment/results/response_' + str(prompt_length) + '.txt', 'w') as f:
            f.write('Prompt: ' + prompt + '\n')
            f.write('Response: ' + output + '\n')


        return CompletionResponse(text=output)
    
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError()
    
def llama_index_input_prompts(dataset_df, dataset_name):
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

    # define our LLM
    llamaindex_llm = MyLLM()

    service_context = ServiceContext.from_defaults(
        llm=llamaindex_llm, 
        context_window=llamaindex_llm.context_window, 
        num_output=llamaindex_llm.num_output
    )

    # NOTE: here the files read should not contain the target (ground-truth) score of the samples as it should not be accessible to llama-index (since we are not training anything)
    # input_files = [f"./Experiment/Data/lamp_reformatted/LaMP_3/LaMP_3_dataset_{dataset_name}_no_target.json"]  # here we can give a list of files on which we should create the llama index
    input_files = [f"./Experiment/Data/lamp_reformatted/LaMP_3/LaMP_3_dataset_{dataset_name}_subset_no_target.json"]  # here we can give a list of files on which we should create the llama index
    excluded_files = []
    documents = SimpleDirectoryReader(input_files=input_files, exclude=excluded_files).load_data()     # https://gpt-index.readthedocs.io/en/latest/reference/readers.html#llama_index.readers.SimpleDirectoryReader
    # print(documents)

    # index_documents = GPTListIndex.from_documents(documents, service_context=service_context)
    index_documents = ListIndex.from_documents(documents, service_context=service_context)
    # index_documents = GPTVectorStoreIndex.from_documents(documents, openai_api_key=os.environ['OPENAI_API_KEY'], name="dataset")

    query_engine = index_documents.as_query_engine(text_qa_template=DEFAULT_TEXT_QA_PROMPT)
    # print(query_engine)

    print("Dataset: ", dataset_df, "\n")
    print("Dataset shape: ", dataset_df.shape, "\n")
    # print("Dataset columns: ", dataset_df.columns())

    for i, sample_prompt in enumerate(dataset_df['prompt']):    # iterate over the samples of the dataset
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

        llama_prompt = f"""
        The current user has the id {user_id}. 
        Given the review delimited by triple backticks, retrieve a different but similar review in the profile field of the same user.
        The review you output should be similar to the review delimited by triple backticks as it should help in making a score prediction for it.
        The review you output should be the review text and its associated score.
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

        print("llama-prompt: ", llama_prompt, "\n")

        response = str(query_engine.query(llama_prompt))

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

        print("Initial input prompt: ", sample_prompt, "\n")

        print("Llama-index response: ", response, "\n")
        
        dataset_df['prompt'][i] = sample_prompt + "\nAn example of correct prediction for this user is: " + response
        
        print("New input prompt: ", dataset_df['prompt'][i], "\n")

    return dataset_df

def predict(dataset_df, dataset_name, evaluation_state):
    print(f"We consider the following dataset/dataframe of shape {dataset_df.shape}: ")
    print("Dataset head: ", dataset_df.head())

    total_nb_samples = len(dataset_df['prompt'])

    outputs = []

    # messages = []   # TODO: Prompt buffering needed? What when prompt gets too long (> 4000 tokens)? What is discarded? Concretely, should I append the prediction to "messages" to keep track of the "conversation"? Wouldn't this be like having the model remember the predictions for the previous samples?

    if config.use_llama_index:
        dataset_df = llama_index_input_prompts(dataset_df, dataset_name)

    uids, input_data_batches = batch_samples(dataset_df)

    if config.API_CHOICE == "official":    # Here using the official API: https://platform.openai.com/
        outputs, evaluation_state = model_api_oai.run_api(total_nb_samples, uids, input_data_batches, outputs, evaluation_state)

    elif config.API_CHOICE == "free":  # Here using free API: https://github.com/acheong08/ChatGPT
        if config.FREE_API_VERSION == "V1":
            outputs, evaluation_state = model_api_free_v1.run_api(total_nb_samples, uids, input_data_batches, outputs, evaluation_state)
        elif config.FREE_API_VERSION == "V3":
            outputs, evaluation_state = model_api_free_v3.run_api(total_nb_samples, uids, input_data_batches, outputs, evaluation_state)

        elif config.FREE_API_VERSION == "churchless":
            outputs, evaluation_state = model_api_free_churchless.run_api(total_nb_samples, uids, input_data_batches, outputs, evaluation_state)

    return outputs, evaluation_state
            
def evaluate(dataset_df, dataset_name, test_mode=False):

    evaluation_state = {'nb_samples_processed': 0, 'uid_index': 0, 'predicted_uids': []}

    try:
        predictions, evaluation_state = predict(dataset_df, dataset_name, evaluation_state)

    except (openai.error.RateLimitError, revChatGPT.typings.APIConnectionError) as e:    # OAI API rate limit is reached
        print("Exception: ", e, "\n")
        return None, None

    print("Predictions: ", predictions)
    
    if not test_mode:
        gt_targets = dataset_df['completion'].tolist()
        print("GT targets: ", gt_targets, "\n")

        assert len(gt_targets) == len(predictions), f"The number of ground truth targets ({len(gt_targets)}) and predictions ({len(predictions)}) should be the same."
        print(f"Computing metrics for {len(gt_targets)} samples...")
        compute_metrics(gt_targets, predictions)

    else:
        print("Test mode. Hence, no ground truth targets available (None)\n")
        gt_targets = None
        
    return gt_targets, predictions, evaluation_state

def save_results_to_file(gt_targets, outputs, evaluation_state):

    # convert the lists of integers to lists of strings
    gt_targets = list(map(lambda integer_value: str(integer_value), gt_targets))
    outputs = list(map(lambda integer_value: str(integer_value), outputs))
    predicted_uids = list(map(str, evaluation_state['predicted_uids']))
    
    # save the ground truth targets and the predictions in files
    with open('./Experiment/Data/lamp_reformatted/gt_targets.txt','w') as f:
        f.write('\n'.join(gt_targets))

    with open('./Experiment/Data/lamp_reformatted/predictions.txt','w') as f:
        f.write('\n'.join(outputs))

    with open('./Experiment/Data/lamp_reformatted/predicted_uids.txt','w') as f:
        f.write('\n'.join(predicted_uids))
    

def setup_experiment():

    if config.use_llama_index:
        print(f"Using OAI API key since the flag use_llama_index was set to True (OAI API key is required to use llama-index).")
        config.USE_OAI_API_KEY = True
    
    if config.OAI_ACCESS_TOKEN:
        pass

    elif config.USE_OAI_ACCESS_TOKEN or (config.API_CHOICE == "free" and config.FREE_API_VERSION == "V1"):
        # read the OAI API access token from the text file
        with open('./Experiment/oai_api_access_token.txt','r') as f:
            config.OAI_ACCESS_TOKEN = f.read().replace('\n', '')

    if config.OAI_API_KEY:
        openai.api_key = config.OAI_API_KEY
        os.environ['OPENAI_API_KEY'] = config.OAI_API_KEY

    elif config.USE_OAI_API_KEY:
        # read the OAI API key from the text file
        with open('./Experiment/oai_api_private_key.txt','r') as f:
            openai.api_key = f.read().replace('\n', '')
            config.OAI_API_KEY = openai.api_key
            os.environ['OPENAI_API_KEY'] = config.OAI_API_KEY

    if not config.USE_OAI_ACCESS_TOKEN and not config.USE_OAI_API_KEY and not config.FREE_API_VERSION == "churchless":
        print("Either an API token or API private key must exist. The flag USE_OAI_ACCESS_TOKEN or USE_OAI_API_KEY should be set.")
        print("Exiting the program...")
        quit()

    if config.API_CHOICE == "free":
        print(f"Currently using the {config.API_CHOICE} API version {config.FREE_API_VERSION}.")
    else:
        print(f"Currently using the {config.API_CHOICE} API.")

    if not config.lamp_8_samples_version:
        dataset_version = "LaMP_" + config.lamp_dataset_index
    else:
        dataset_version = "LaMP_" + config.lamp_dataset_index + "_" + config.lamp_8_samples_version

    # path to the folder containing the experiment data
    data_folder_path = f"./Experiment/Data/lamp_reformatted/{dataset_version}/"

    return data_folder_path, dataset_version

if __name__ == '__main__':

    # get the config
    config.parse_args()

    print("\nUsing the following INITIAL config: ", config.__str__(), "\n")

    data_folder_path, dataset_version = setup_experiment()

    print("\nUsing the following UPDATED config: ", config.__str__(), "\n")

    # setup wandb
    wb_setup.create_trace()


    # get the dataset splits
    train_df, val_df, test_df = get_dataset_splits(data_folder_path, dataset_version)

    # train the predictive model
    # trained_model = train(train_df)

    # evaluate the predictive model on the validation set
    gt_targets, outputs, evaluation_state = evaluate(val_df, "val", test_mode=False)

    # save the ground truth targets and the predictions in a file
    save_results_to_file(gt_targets, outputs, evaluation_state)

    # get predictions of the predictive model on the test set
    # outputs, evaluation_state = predict(test_df, "test", evaluation_state)

    if config.use_wandb:
        wb_setup.wandb_run.log({"trace": wb_setup.trace})
        wb_setup.wandb_run.finish()