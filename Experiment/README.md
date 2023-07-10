# General Information
This folder contains the files used to perform the experiments such as the evaluation of models (e.g., GPT-3.5-turbo with Llama-index) on different datasets (e.g., [LaMP benchmark](https://lamp-benchmark.github.io/index.html)). 

The file create_oai_dataset.py can be used to create a dataset formatted as per the [OAI recommendations](www.openai.com) from a LaMP dataset.

The script can be run as
```
python create_oai_dataset.py
```

The file reformat_lamp_dataset.py can be used to reformat a lamp dataset into a dataset that can be used by the main file `experiment.py`.

The script can be run as
```
python reformat_lamp_dataset.py
```

In order to use the GPT-3.5-turbo model, an access token (for free V1) or API private key (for free V3 or official OAI API) is required. A private key is necessary to use Llama-index with one of the models. The access token and private can be stored in a oai_api_access_token.txt and oai_api_private_key.txt file respectively.

The main file to run to perform an experiment is the file experiment.py. It evaluates the given model on the given dataset according to the parameters passed and outputs results in files in addition to saving some information in wandb. In case the dataset is a test set, it simply outputs predictions and, obviously, no evaluation is performed. 

The experiment script can be run as
```
python experiment.py
```

# Data
The data folder contains the datasets as well as the experiment results.

# Notes
