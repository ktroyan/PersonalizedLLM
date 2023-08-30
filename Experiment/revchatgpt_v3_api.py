from config import config, CONSTANTS as C

import os
import time
import requests

import openai
import revChatGPT
from revChatGPT.V3 import Chatbot as ChatbotV3

from utils import setup_loguru
from loguru import logger
setup_loguru(logger)


def call_revchatgpt_v3(chatbot, prompt):

    output_data_full = chatbot.ask(prompt)
    for data in output_data_full:
        output = data['message']

    return output


def get_response_from_revchatgpt_v3(chatbot, prompt):
    retries = 0

    try:
        response = call_revchatgpt_v3(chatbot, prompt)

    except (requests.exceptions.HTTPError, revChatGPT.typings.Error) as e:
        logger.warning(f"Exception caught: {e}")
        logger.warning("Trying again to request the model API...")
        retries += 1
        time.sleep(0.5)
        if retries < 3:
            response = call_revchatgpt_v3(chatbot, prompt)
        else:
            raise e

    output = response
    return output


config.OAI_API_KEY = ''
if os.path.exists('./Experiment/oai_api_private_key.txt'):
    # read the OAI API access token from the text file
    with open('./Experiment/oai_api_private_key.txt', 'r') as f:
        config.OAI_API_KEY = f.read().replace('\n', '')

openai.api_key = config.OAI_API_KEY
os.environ['OPENAI_API_KEY'] = config.OAI_API_KEY

chatbot = ChatbotV3(api_key=openai.api_key,
                    engine="gpt-3.5-turbo",
                    proxy=None,
                    timeout=None,
                    max_tokens=None,
                    temperature=0.1,
                    top_p=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    reply_count=1,
                    # "You are ChatGPT, a large language model trained by OpenAI. Respond as concisely, straightforwardly and accurately as possible."
                    system_prompt="You are a state-of-the-art predictive model. You should only output predictions strictly respecting the output format."
                    )
