from config import config, CONSTANTS as C

import time
import requests

from utils import setup_loguru
from loguru import logger
setup_loguru(logger)

def call_churchless(prompt):
    url = 'https://free.churchless.tech/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ChatGPT-Hackers'
    }

    messages = [{"role": "user", "content": prompt}]

    data = {"model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.1, }
    
    response = requests.post(url, headers=headers, json=data)

    return response

def get_response_from_churchless(prompt):
    
    response = call_churchless(prompt)
    response = response.json()

    if 'choices' not in response:
        logger.debug("Response does not contain a 'choices' attribute. Not able to get the output text.")
        retries = 0
        while retries < 3:
            response = call_churchless(prompt)
            response = response.json()
            if 'choices' in response:
                break
            retries += 1
            time.sleep(0.5)
    
    try:
        output = response['choices'][0]['message']['content']
    except KeyError:
        logger.debug("""Response does not contain a 'choices' attribute. 
                     An error occurred. 
                     Not able to get the output text. 
                     You should wait until the API is available again.
                     Exiting now...
                     """)
        quit()
        
    return output