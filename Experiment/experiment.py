from utils import print_in_blue
from config import config, CONSTANTS as C

import os
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

from utils import wb_setup
import wandb
assert wandb.__version__ >= "0.15.3", "Please ensure you are using wandb v0.15.3 or higher in order to use WandB prompts."

from IPython.display import Markdown, display

from create_llama_index_dataset import llama_index_input_prompts

def get_dataset_splits(data_folder_path, dataset_version):
    data_path = data_folder_path + dataset_version
    
    train_dataset = None
    # train_dataset = pd.read_json(data_path + "_dataset_" + "train.json", orient='records')
    
    # val_dataset = pd.read_json(data_path + "_dataset_" + "val.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    val_dataset = pd.read_json(data_path + "_dataset_" + "val_subset.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    
    test_dataset = pd.read_json(data_path + "_dataset_" + "test.json", orient='records')

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
    input_data_batched = [dataset_df['prompt'][i:i+config.request_batch_size] for i in range(0,len(dataset_df['prompt']), config.request_batch_size)]
    return input_data_batched

def predict(data_folder_path, dataset_df, dataset_name, evaluation_state):
    print(f"We consider the following dataset/dataframe of shape {dataset_df.shape}: ")
    print("Dataset head: ", dataset_df.head())

    total_nb_samples = len(dataset_df['prompt'])

    outputs = []

    # messages = []   # TODO: Prompt buffering needed? What when prompt gets too long (> 4000 tokens)? What is discarded? Concretely, should I append the prediction to "messages" to keep track of the "conversation"? Wouldn't this be like having the model remember the predictions for the previous samples?

    if config.use_llama_index:
        dataset_df = llama_index_input_prompts(data_folder_path, dataset_df, dataset_name)

    uids = dataset_df['uid']

    input_data_batched = batch_samples(dataset_df)

    print_in_blue(f"INFO: \n total_number_samples: {total_nb_samples} \n uids: {uids} \n input_data_batched: {input_data_batched} \n outputs: {outputs} \n evaluation_state: {evaluation_state} \n")

    print_in_blue("\n\n --- CALLING MAIN LLM FOR PREDICTIONS --- \n\n")

    if config.API_CHOICE == "official":    # Here using the official API: https://platform.openai.com/
        outputs, evaluation_state = model_api_oai.run_api(total_nb_samples, uids, input_data_batched, outputs, evaluation_state)

    elif config.API_CHOICE == "free":  
        if config.FREE_API_VERSION == "V1":     # Here using free API: https://github.com/acheong08/ChatGPT
            outputs, evaluation_state = model_api_free_v1.run_api(total_nb_samples, uids, input_data_batched, outputs, evaluation_state)
        elif config.FREE_API_VERSION == "V3":   # Here using free API: https://github.com/acheong08/ChatGPT
            outputs, evaluation_state = model_api_free_v3.run_api(total_nb_samples, uids, input_data_batched, outputs, evaluation_state)
        elif config.FREE_API_VERSION == "churchless":
            outputs, evaluation_state = model_api_free_churchless.run_api(total_nb_samples, uids, input_data_batched, outputs, evaluation_state)

    return outputs, evaluation_state
            
def evaluate(data_folder_path, dataset_df, dataset_name, test_mode=False):

    evaluation_state = {'nb_samples_processed': 0, 'uid_index': 0, 'predicted_uids': []}

    try:
        predictions, evaluation_state = predict(data_folder_path, dataset_df, dataset_name, evaluation_state)

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
    with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/gt_targets.txt','w') as f:
        f.write('\n'.join(gt_targets) + "\n")

    with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/predictions.txt','w') as f:
        f.write('\n'.join(outputs) + "\n")

    with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/predicted_uids.txt','a') as f:
        f.write('\n'.join(predicted_uids) + "\n")
    

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

    # create the /result folder for llama-index if it doesn't exist
    if not os.path.exists(data_folder_path + "results/"):
        os.makedirs(data_folder_path + "results/")

    return data_folder_path, dataset_version

if __name__ == '__main__':

    # get the config
    config.parse_args()

    print_in_blue("\nUsing the following INITIAL config: " + config.__str__() + "\n")

    data_folder_path, dataset_version = setup_experiment()

    print_in_blue("\nUsing the following UPDATED config: " + config.__str__() + "\n")

    # setup wandb
    wb_setup.create_trace()

    # get the dataset splits
    train_df, val_df, test_df = get_dataset_splits(data_folder_path, dataset_version)

    # train the predictive model
    # trained_model = train(train_df)

    # evaluate the predictive model on the validation set
    gt_targets, outputs, evaluation_state = evaluate(data_folder_path, val_df, "val", test_mode=False)

    # save the ground truth targets and the predictions in a file
    save_results_to_file(gt_targets, outputs, evaluation_state)

    # get predictions of the predictive model on the test set
    # outputs, evaluation_state = predict(test_df, "test", evaluation_state)

    if config.use_wandb:
        wb_setup.wandb_run.log({"trace": wb_setup.trace})
        wb_setup.wandb_run.finish()