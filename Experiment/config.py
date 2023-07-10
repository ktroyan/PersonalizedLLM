import argparse
import json
import pprint

class Constants(object):
    """
    This is a singleton (only a single instance of the class should exist in the program).
    """
    class __Constants:
        def __init__(self):
            pass

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


class Configuration(object):
    """Configuration parameters given via the commandline."""

    def __init__(self, ):
        pass

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    def parse_args(self,):
        args_parser = argparse.ArgumentParser()

        # General
        args_parser.add_argument("--USE_OAI_ACCESS_TOKEN", action='store_true', default=False, help="If there is a valid access token to OpenAI API in a local .txt file. Works for V1")
        args_parser.add_argument("--USE_OAI_API_KEY", action='store_true', default=False, help="If there is a valid OpenAI API key in a local .txt file.")
        args_parser.add_argument("--API_CHOICE", type=str, default="free", help="The API to use. I.e.: official or free.")
        args_parser.add_argument("--FREE_API_VERSION", type=str, default="churchless", help="The version of the free API to use. I.e.: V1 or V3.")
        args_parser.add_argument("--OAI_ACCESS_TOKEN", type=str, default=None, help="A valid access token to OpenAI API. Works for V1.")
        args_parser.add_argument("--OAI_API_KEY", type=str, default=None, help="A valid OpenAI API key.")

        # Setup
        args_parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
        args_parser.add_argument('--seed', type=int, default=121997, help='Seed for reproducibility, set randomness.')
        args_parser.add_argument("--use_wandb", action='store_true', default=False, help="Whether or not to use WandB during the experiment run.")

        # Experiment run
        args_parser.add_argument("--use_llama_index", action='store_true', default=False, help="Whether or not to use llama-index with the predictive model.")

        # Data
        args_parser.add_argument("--lamp_dataset_index", type=str, required=True, help="LaMP dataset index. E.g., 1, 2, 3, etc.")
        args_parser.add_argument("--lamp_8_samples_version", type=str, default=None, help="Rounded down in thousands number of samples for LaMP_8 dataset. E.g., 3K, 10K, etc.")

        # Model
        args_parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Name of the predictive model: gpt-3.5-turbo")
        args_parser.add_argument("--request_batch_size", type=int, default=1, help="Size of the batch of samples to send to the API for prediction.")


        config = args_parser.parse_args()
        self.__dict__.update(vars(config))


    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)


# create an instance of the constants class
CONSTANTS = Constants()

# create an instance of the configuration class
config = Configuration()