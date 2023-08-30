from config import config, CONSTANTS as C

import os
import time

import openai

from utils import setup_loguru
from loguru import logger
setup_loguru(logger)


def call_oai(task, messages, prompt):

    messages.append(task)
    prompt_request = {"role": "user", "content": prompt}
    messages.append(prompt_request)

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages,
                                            temperature=.1,
                                            max_tokens=2048,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                            )

    return response


def get_response_from_oai(prompt):
    task_context = "This is a score prediction task. Score predictions are 1, 2, 3, 4 or 5."
    task = {"role": "assistant", "content": task_context}
    messages = []   # remove this line if prompt buffering is needed

    response = call_oai(task, messages, prompt)

    if 'choices' not in response:
        logger.debug(
            "Response does not contain a 'choices' attribute. Not able to get the output text.")
        retries = 0
        while retries < 3:
            response = call_oai(task, messages, prompt)
            response = response.json()
            if 'choices' in response:
                break
            retries += 1
            time.sleep(0.5)

    output = response.choices[0].message.content

    return output


if os.path.exists('./Experiment/oai_api_private_key.txt'):
    # read the OAI API access token from the text file
    with open('./Experiment/oai_api_private_key.txt', 'r') as f:
        config.OAI_API_KEY = f.read().replace('\n', '')

    openai.api_key = config.OAI_API_KEY
    os.environ['OPENAI_API_KEY'] = config.OAI_API_KEY
else:
    logger.warning(
        "oai_api_private_key file doesn't exist.")
