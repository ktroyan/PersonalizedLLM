from config import config, CONSTANTS as C

from wandb.sdk.data_types import trace_tree
import wandb
assert wandb.__version__ >= "0.15.3", "Please ensure you are using wandb v0.15.3 or higher in order to use WandB prompts."

import tiktoken

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