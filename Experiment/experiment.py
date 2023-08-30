from config import config, CONSTANTS as C
from utils import print_in_blue

import os
import pandas as pd    
pd.options.mode.chained_assignment = None  # default='warn'

import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

import openai
import revChatGPT

import model_api_lamp1
import model_api_lamp2
import model_api_lamp3

import utils
from utils import wb_setup
import wandb
assert wandb.__version__ >= "0.15.3", "Please ensure you are using wandb v0.15.3 or higher in order to use WandB prompts."

from loguru import logger
from utils import setup_loguru
setup_loguru(logger)

from IPython.display import Markdown, display

from create_llama_index_dataset import llama_index_input_prompts

def get_dataset_splits(data_folder_path, dataset_version):
    data_path = data_folder_path + dataset_version
    
    train_dataset = None
    # train_dataset = pd.read_json(data_path + "_dataset_" + "train.json", orient='records')
    
    # val_dataset = pd.read_json(data_path + "_dataset_" + "val.json", orient='records')
    val_dataset = pd.read_json(data_path + "_dataset_" + "val_subset.json", orient='records')    # TODO: change this to "val.json" when the dataset is ready
    
    test_dataset = None
    # test_dataset = pd.read_json(data_path + "_dataset_" + "test.json", orient='records')

    return train_dataset, val_dataset, test_dataset

def train(dataset_df):
    """
    Training/fine-tuning is not done for models such as GPT-3.5-turbo.
    # NOTE: Update 23.08.2023: OAI now supports fine-tuning for GPT-3.5-turbo.

    """
    raise NotImplementedError

def compute_metrics(gt_targets, predictions):
    
    # metrics reported for the official LaMP benchmark

    if config.lamp_dataset_index in ["1", "2"]:
        logger.info(f"Accuracy score: {accuracy_score(gt_targets, predictions)}")

    elif config.lamp_dataset_index == "3":
        logger.info(f"Accuracy score: {accuracy_score(gt_targets, predictions)}")
        logger.info(f"Mean Absolute Error (MAE): {mean_absolute_error(gt_targets, predictions)}")
        logger.info(f"Root Mean Squared Error (RMSE):  {mean_squared_error(gt_targets, predictions, squared=False)}")
        # logger.info(f"F1 score: {f1_score(gt_targets, predictions, average='macro')})
        # logger.info(f"Precision score: {precision_score(gt_targets, predictions, average='macro')})
        # logger.info(f"Recall score: {recall_score(gt_targets, predictions, average='macro')})

def batch_samples(dataset_df):
    logger.info(f"Using batches of size {config.request_batch_size} for the API requests.")

    # batch the samples to reduce the total number of API requests to perform
    input_data_batched = [dataset_df['input'][i:i+config.request_batch_size] for i in range(0,len(dataset_df['input']), config.request_batch_size)]
    return input_data_batched

def predict(dataset_df, dataset_version, evaluation_state):
    logger.info(f"We consider the following dataset/dataframe of shape {dataset_df.shape}.")
    logger.info(f"Dataset head: {dataset_df.head()}")

    total_nb_samples = len(dataset_df['input'])

    outputs = []

    uids = dataset_df['uid']

    input_data_batched = batch_samples(dataset_df)

    logger.info(f"total_number_samples: {total_nb_samples} \n uids: {uids} \n input_data_batched: {input_data_batched} \n outputs: {outputs} \n evaluation_state: {evaluation_state} \n")

    if "LaMP_1" in dataset_version:
        outputs, evaluation_state = model_api_lamp1.run_api(total_nb_samples, uids, input_data_batched, outputs, evaluation_state, retry_formatting=False)
    
    elif "LaMP_2" in dataset_version:
        outputs, evaluation_state = model_api_lamp2.run_api(total_nb_samples, uids, input_data_batched, outputs, evaluation_state, retry_formatting=False)

    elif "LaMP_3" in dataset_version:
        outputs, evaluation_state = model_api_lamp3.run_api(total_nb_samples, uids, input_data_batched, outputs, evaluation_state, retry_formatting=False)

    return outputs, evaluation_state

def save_results_to_file(gt_targets, predictions, evaluation_state, dataset_version):

    if "LaMP_1" in dataset_version:
        # already list of string values (single integer for the class)
        pass
    
    elif "LaMP_2" in dataset_version:
        pass

    elif "LaMP_3" in dataset_version:
        # convert the lists of integers to lists of strings
        gt_targets = list(map(lambda integer_value: str(integer_value), gt_targets))
        predictions = list(map(lambda integer_value: str(integer_value), predictions))
    

    predicted_uids = list(map(str, evaluation_state['predicted_uids']))
    
    # save the ground truth targets and the predictions in files
    with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/gt_targets.txt','w') as f:
        f.write('\n'.join(gt_targets) + "\n")

    with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/predictions.txt','a') as f:
        f.write('\n'.join(predictions) + "\n")

    with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/predicted_uids.txt','a') as f:
        f.write('\n'.join(predicted_uids) + "\n")
    

def evaluate(data_folder_path, dataset_df, dataset_name, dataset_version, test_mode=False):

    evaluation_state = {'nb_samples_processed': 0, 'uid_index': 0, 'predicted_uids': []}

    if config.use_llama_index:
        if config.use_already_indexed_dataset and os.path.exists(f"{data_folder_path}LaMP_{config.lamp_dataset_index}_dataset_{dataset_name}_llama_indexed.json"):
            dataset_df = pd.read_json(f"{data_folder_path}LaMP_{config.lamp_dataset_index}_dataset_{dataset_name}_llama_indexed.json", orient='records')
        else:
            dataset_df = llama_index_input_prompts(data_folder_path, dataset_df, dataset_name)
    
    try:
        predictions, evaluation_state = predict(dataset_df, dataset_version, evaluation_state)

    except (openai.error.RateLimitError, revChatGPT.typings.APIConnectionError) as e:    # OAI API rate limit is reached
        logger.warning(f"Exception: {e} \n")
        return None, None

    logger.info(f"Predictions made: {predictions} \n")
    
    if not test_mode:
        gt_targets = dataset_df['output'].tolist()
        logger.info(f"GT targets: {gt_targets} \n")

        # assert len(gt_targets) == len(predictions), f"The number of ground truth targets ({len(gt_targets)}) and predictions ({len(predictions)}) should be the same."

        if "LaMP_1" in dataset_version:
            # remove the square brackets around the integer predicted
            predictions = [prediction[1:-1] for prediction in predictions]
            gt_targets = [gt_target[1:-1] for gt_target in gt_targets]

        elif "LaMP_2" in dataset_version:
            pass

        elif "LaMP_3" in dataset_version:
            pass

        # save the ground truth targets and the predictions in a file
        save_results_to_file(gt_targets, predictions, evaluation_state, dataset_version)

        # read the values in the files
        with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/gt_targets.txt','r') as f:
            gt_targets = f.read().splitlines()
        
        with open(f'./Experiment/Data/lamp_reformatted/{dataset_version}/results/predictions.txt','r') as f:
            predictions = f.read().splitlines()

        logger.info(f"Predictions until now: {predictions}")
        logger.info(f"Ground truth targets until now: {gt_targets}")

        if "LaMP_1" in dataset_version:
            predictions = [int(prediction) for prediction in predictions]
            gt_targets = [int(gt_target) for gt_target in gt_targets]
        
        elif "LaMP_2" in dataset_version:
            # processing needed before computing the metrics?
            pass

        elif "LaMP_3" in dataset_version:
            predictions = [int(prediction) for prediction in predictions if prediction != '']
            gt_targets = [int(gt_target) for gt_target in gt_targets]
        
        
        logger.info(f"All predictions: {predictions}")
        logger.info(f"All ground-truth targets: {gt_targets}")

        logger.info(f"Computing metrics for a total of {len(gt_targets)} samples...")
        compute_metrics(gt_targets, predictions)

    else:
        logger.info("Test mode. Hence, no ground truth targets available (None)\n")
        gt_targets = None
        
    return gt_targets, predictions, evaluation_state

def setup_experiment():

    if config.USE_OAI_ACCESS_TOKEN or config.API_CHOICE == "V1":
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

    if not config.USE_OAI_ACCESS_TOKEN and not config.USE_OAI_API_KEY and not config.API_CHOICE == "churchless":
        logger.warning("Either an API token or API private key must exist. The flag USE_OAI_ACCESS_TOKEN or USE_OAI_API_KEY should be set.")
        logger.warning("Exiting the program...")
        quit()

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

    logger.info("\nUsing the following INITIAL config: " + config.__str__() + "\n")

    data_folder_path, dataset_version = setup_experiment()

    logger.info("\nUsing the following UPDATED config: " + config.__str__() + "\n")

    # setup wandb
    wb_setup.create_trace()

    # get the dataset splits
    train_df, val_df, test_df = get_dataset_splits(data_folder_path, dataset_version)

    # train the predictive model
    # trained_model = train(train_df)

    # evaluate the predictive model on the validation set
    gt_targets, outputs, evaluation_state = evaluate(data_folder_path, val_df, "val", dataset_version, test_mode=False)

    # get predictions of the predictive model on the test set
    # outputs, evaluation_state = predict(test_df, "test", evaluation_state)

    if config.use_wandb:
        wb_setup.wandb_run.log({"trace": wb_setup.trace})
        wb_setup.wandb_run.finish()