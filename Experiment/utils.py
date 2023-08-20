"""
Utility functions for the experiment files.
"""

import sys
from config import config, CONSTANTS as C

from wandb.sdk.data_types import trace_tree
import wandb
assert wandb.__version__ >= "0.15.3", "Please ensure you are using wandb v0.15.3 or higher in order to use WandB prompts."

import tiktoken

from datetime import datetime

def print_in_red(s):
    print("\n\n \033[91m" + s + "\033[0m \n\n")
          
def print_in_green(s):
    print("\n\n \033[92m" + s + "\033[0m \n\n")

def print_in_yellow(s):
    print("\n\n \033[93m" + s + "\033[0m \n\n")

def print_in_blue(s):
    print("\n\n \033[94m" + s + "\033[0m \n\n")

def setup_loguru(logger):
    def trace_only(record):
        return record["level"].name == "TRACE"
    def debug_only(record):
        return record["level"].name == "DEBUG"
    def info_only(record):
        return record["level"].name == "INFO"
    def success_only(record):
        return record["level"].name == "SUCCESS"
    def warning_only(record):
        return record["level"].name == "WARNING"
    def error_only(record):
        return record["level"].name == "ERROR"

    logger.remove()
    current_date = datetime.now()
    formatted_date = current_date.strftime("%m-%d-%H")
    log_filename = "./Experiment/logs/experiment_" + formatted_date + "H" + ".log"
    fmt ='\n<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <lvl>{level: <8}</> | <cyan>{name}:{function}:{line}</> | \n <lvl>{message}</>\n'
    logger.add(log_filename, level="TRACE", format=fmt)
    logger.add(sys.stdout, format=fmt, filter=debug_only)

def nb_tokens_in_string(string, encoding_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(encoding_name)
    nb_tokens = len(encoding.encode(string))
    return nb_tokens

class WBSetup(object):

    def __init__(self, ):
        pass

    def create_trace(self,):

        # prepare wandb prompts
        root_span = trace_tree.Span(name="PLLM", span_kind=trace_tree.SpanKind.AGENT)   # root Span: create a span for high level agent
        trace = trace_tree.WBTraceTree(root_span)
        if config.use_wandb:
            wandb_run = wandb.init(project="wandb-prompts")
            self.wandb_run = wandb_run

        self.root_span = root_span
        self.trace = trace

# create an instance of the WBSetup class to be used across the experiment modules
wb_setup = WBSetup()