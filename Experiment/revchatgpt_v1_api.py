import os
from config import config, CONSTANTS as C

import time

import revChatGPT
from revChatGPT.V1 import Chatbot as ChatbotV1

from utils import setup_loguru
from loguru import logger
setup_loguru(logger)


def call_revchatgpt_v1(chatbot, prompt):
    output_data_full = chatbot.ask(prompt)
    logger.debug(f"Response from RevChatGPT V1 API: {list(output_data_full)}")
    for data in output_data_full:
        output = data['message']

    return output


def get_response_from_revchatgpt_v1(chatbot, prompt):
    retries = 0

    try:
        response = call_revchatgpt_v1(chatbot, prompt)

    except (revChatGPT.typings.Error) as e:
        logger.warning(f"Exception caught: {e}")
        logger.warning("Trying again to request the model API...")
        retries += 1
        time.sleep(0.5)
        if retries < 3:
            response = call_revchatgpt_v1(chatbot, prompt)
        else:
            raise e

    output = response

    return output


config.OAI_ACCESS_TOKEN = ''

if os.path.exists('./Experiment/oai_api_access_token.txt'):
    # read the OAI API access token from the text file only if it exists:
    with open('./Experiment/oai_api_access_token.txt', 'r') as f:
        config.OAI_ACCESS_TOKEN = f.read().replace('\n', '')
else:
    logger.warning(
        "oai_api_access_token file doesn't exist.")

api_config = {"access_token": config.OAI_ACCESS_TOKEN}
chatbot = ChatbotV1(config=api_config)
