# coding: utf-8

FROM_REMOTE=True
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import logging
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Example script to parse command-line parameters.')
parser.add_argument('-n', type=int, default=0, help='Number of times to process the file')
parser.add_argument('-c', type=bool, default=False, help='Zero-shot COT Prompting')

# Parse the command-line arguments
args = parser.parse_args()

# Suppress Warnings during inference
logging.getLogger("transformers").setLevel(logging.ERROR)

# fpb_datasets = load_dataset("financial_phrasebank", "sentences_50agree")
# sent_data = pd.read_csv('financial_data_cleaned.csv')
# with open('samples.txt', 'r') as f:
#     sample_data = f.readlines()

# sample_data = [sample[:-1] for sample in sample_data]
# social_media_dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')

dataset = pd.read_csv('samples.txt')
num_samples = 10
demo_tasks = ['Financial Sentiment Analysis'] * num_samples
# demo_inputs = fpb_datasets['train']['sentence'][-20:]
# demo_inputs = sent_data['Sentence'][-num_samples:]
# demo_inputs = sample_data
# demo_inputs = social_media_dataset['validation']['text'][:num_samples]
demo_inputs = dataset['text'][:num_samples]
demo_instructions = ['What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'] * num_samples
'''
demo_tasks = [
    'Financial Sentiment Analysis',
    'Financial Relation Extraction',
    'Financial Headline Classification',
    'Financial Named Entity Recognition',
]
demo_inputs = [
    "Glaxo's ViiV Healthcare Signs China Manufacturing Deal With Desano",
    "Wednesday, July 8, 2015 10:30AM IST (5:00AM GMT) Rimini Street Comment on Oracle Litigation Las Vegas, United States Rimini Street, Inc., the leading independent provider of enterprise software support for SAP AG’s (NYSE:SAP) Business Suite and BusinessObjects software and Oracle Corporation’s (NYSE:ORCL) Siebel , PeopleSoft , JD Edwards , E-Business Suite , Oracle Database , Hyperion and Oracle Retail software, today issued a statement on the Oracle litigation.",
    'april gold down 20 cents to settle at $1,116.10/oz',
    'Subject to the terms and conditions of this Agreement , Bank agrees to lend to Borrower , from time to time prior to the Commitment Termination Date , equipment advances ( each an " Equipment Advance " and collectively the " Equipment Advances ").',
]
demo_instructions = [
    'What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.',
    'Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.',
    'Does the news headline talk about price in the past? Please choose an answer from {Yes/No}.',
    'Please extract entities and their types from the input sentence, entity types should be chosen from {person/organization/location}.',
]
'''

one_shot = """Example:
ManTech downgraded ahead of difficult comps. // negative\n"""

three_shot = """Example:
ManTech downgraded ahead of difficult comps. // negative
BMO Capital Markets ups to Outperform. // postive
Arex Capital ramps up pressure on Zagg. // neutral\n"""

five_shot = """Example:\n
ManTech downgraded ahead of difficult comps. // negative
Bank of Ireland cuts key profit target as low rates take toll. // negative
Fed's Rosengren says he is 'optimistic' on economy. // positive
BMO Capital Markets ups to Outperform. // postive
Arex Capital ramps up pressure on Zagg. // neutral\n"""

ten_shot = """Example:
ManTech downgraded ahead of difficult comps. // negative
Bank of Ireland cuts key profit target as low rates take toll. // negative
$300,000 Pilot Jobs Drying Up in China After Boeing Grounding. // negative
Apple forecasts 100M+ 5G iPhone sales. // postive
Fed's Rosengren says he is 'optimistic' on economy. // positive
BMO Capital Markets ups to Outperform. // postive
Arex Capital ramps up pressure on Zagg. // neutral
Brazil's central bank stepped in to prop up the currency. // neutral
Alibaba, Manchester United sign streaming pact. // neutral
Credit Suisse power struggle mounts - Bloomberg // neutral\n"""

def load_model(base_model, peft_model, from_remote=False):
    
    model_name = parse_model_name(base_model, from_remote)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, 
        device_map="auto",
    )
    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.padding_side = "left"
    if base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    return model, tokenizer


def test_demo(model, tokenizer):

    idx = 0
    for task_name, input, instruction in zip(demo_tasks, demo_inputs, demo_instructions):
        
        if args.c:
            prompt = 'Instruction: {instruction}\nInput: {input}\nAnswer: Tell me about your decision-making process and the final result'.format(
                input=input, 
                instruction=instruction
            )
        else:
            prompt = 'Instruction: {instruction}\nInput: {input}\nAnswer: '.format(
                input=input, 
                instruction=instruction
            )
        
        # enable different types few shot
        if args.n == 1:
            prompt = one_shot+prompt
        elif args.n == 3:
            prompt = three_shot+prompt
        elif args.n == 5:
            prompt = five_shot+prompt
        elif args.n == 10:
            prompt = ten_shot+prompt
        
        inputs = tokenizer(
            prompt, return_tensors='pt',
            padding=True, max_length=512,
            return_token_type_ids=False
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        res = model.generate(
            **inputs, max_length=512, do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        output = tokenizer.decode(res[0], skip_special_tokens=True)
        print(f"\n==== {task_name} ====\n")
        print(output)
        # print(output, social_media_dataset['validation']['label'][idx])
        idx += 1
    
import os
import datasets

# A dictionary to store various prompt templates.
template_dict = {
    'default': 'Instruction: {instruction}\nInput: {input}\nAnswer: '
}

# A dictionary to store the LoRA module mapping for different models.
lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'falcon': ['query_key_value'],
    'bloom': ['query_key_value'],
    'internlm': ['q_proj', 'k_proj', 'v_proj'],
    'llama2': ['q_proj', 'k_proj', 'v_proj'],
    'llama2-13b': ['q_proj', 'k_proj', 'v_proj'],
    'llama2-13b-nr': ['q_proj', 'k_proj', 'v_proj'],
    'qwen': ["c_attn"],
    'mpt': ['Wqkv'],
    'baichuan': ['q_proj', 'k_proj', 'v_proj'],
}


def get_prompt(template, instruction, input_text):
    """
    Generates a prompt based on a predefined template, instruction, and input.

    Args:
    template (str): The key to select the prompt template from the predefined dictionary.
    instruction (str): The instruction text to be included in the prompt.
    input_text (str): The input text to be included in the prompt.

    Returns:
    str: The generated prompt.

    Raises:
    KeyError: If the provided template key is not found in the template dictionary.
    """
    if not instruction:
        return input_text

    if template not in template_dict:
        raise KeyError(f"Template '{template}' not found. Available templates: {', '.join(template_dict.keys())}")

    return template_dict[template].format(instruction=instruction, input=input_text)


def test_mapping(args, feature):
    """
    Generate a mapping for testing purposes by constructing a prompt based on given instructions and input.

    Args:
    args (Namespace): A namespace object that holds various configurations, including the instruction template.
    feature (dict): A dictionary containing 'instruction' and 'input' fields used to construct the prompt.

    Returns:
    dict: A dictionary containing the generated prompt.

    Raises:
    ValueError: If 'instruction' or 'input' are not provided in the feature dictionary.
    """
    # Ensure 'instruction' and 'input' are present in the feature dictionary.
    if 'instruction' not in feature or 'input' not in feature:
        raise ValueError("Both 'instruction' and 'input' need to be provided in the feature dictionary.")

    # Construct the prompt using the provided instruction and input.
    prompt = get_prompt(
        args.instruct_template,
        feature['instruction'],
        feature['input']
    )

    return {
        "prompt": prompt,
    }

def tokenize(args, tokenizer, feature):
    """
    Tokenizes the input prompt and target/output for model training or evaluation.

    Args:
    args (Namespace): A namespace object containing various settings and configurations.
    tokenizer (Tokenizer): A tokenizer object used to convert text into tokens.
    feature (dict): A dictionary containing 'input', 'instruction', and 'output' fields.

    Returns:
    dict: A dictionary containing tokenized 'input_ids', 'labels', and a flag 'exceed_max_length'.
    """
    # Generate the prompt.
    prompt = get_prompt(
        args.instruct_template,
        feature['instruction'],
        feature['input']
    )
    # Tokenize the prompt.
    prompt_ids = tokenizer(
        prompt,
        padding=False,
        max_length=args.max_length,
        truncation=True
    )['input_ids']

    # Tokenize the target/output.
    target_ids = tokenizer(
        feature['output'].strip(),
        padding=False,
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=False
    )['input_ids']

    # Combine tokenized prompt and target output.
    input_ids = prompt_ids + target_ids

    # Check if the combined length exceeds the maximum allowed length.
    exceed_max_length = len(input_ids) >= args.max_length

    # Add an end-of-sequence (EOS) token if it's not already present
    # and if the sequence length is within the limit.
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)

    # Create label IDs for training.
    # The labels should start from where the prompt ends, and be padded for the prompt portion.
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def parse_model_name(name, from_remote=False):
    """
    Parse the model name and return the appropriate path based on whether
    the model is to be fetched from a remote source or from a local source.

    Args:
    - name (str): Name of the model.
    - from_remote (bool): If True, return the remote path, else return the local path.

    Returns:
    - str: The appropriate path for the given model name.
    """
    model_paths = {
        'chatglm2': ('THUDM/chatglm2-6b', 'base_models/chatglm2-6b'),
        'llama2': ('meta-llama/Llama-2-7b-hf', 'base_models/Llama-2-7b-hf'),
        'llama2-13b': ('meta-llama/Llama-2-13b-hf', 'base_models/Llama-2-13b-hf'),
        'llama2-13b-nr': ('NousResearch/Llama-2-13b-hf', 'base_models/Llama-2-13b-hf'),
        'falcon': ('tiiuae/falcon-7b', 'base_models/falcon-7b'),
        'internlm': ('internlm/internlm-7b', 'base_models/internlm-7b'),
        'qwen': ('Qwen/Qwen-7B', 'base_models/Qwen-7B'),
        'baichuan': ('baichuan-inc/Baichuan2-7B-Base', 'base_models/Baichuan2-7B-Base'),
        'mpt': ('cekal/mpt-7b-peft-compatible', 'base_models/mpt-7b-peft-compatible'),
        'bloom': ('bigscience/bloom-7b1', 'base_models/bloom-7b1')
    }

    if name in model_paths:
        return model_paths[name][0] if from_remote else model_paths[name][1]
    else:
        valid_model_names = ', '.join(model_paths.keys())
        raise ValueError(f"Undefined base model '{name}'. Valid model names are: {valid_model_names}")


def load_dataset(names, from_remote=False):
    """
    Load one or multiple datasets based on the provided names and source location.

    Args:
    names (str): A comma-separated list of dataset names. Each name can be followed by '*n' to indicate replication.
    from_remote (bool): If True, load the dataset from Hugging Face's model hub. Otherwise, load it from a local disk.

    Returns:
    List[Dataset]: A list of loaded datasets. Each dataset is possibly replicated based on the input names.
    """
    # Split the dataset names by commas for handling multiple datasets
    dataset_names = names.split(',')
    dataset_list = []

    for name in dataset_names:
        # Initialize replication factor to 1
        replication_factor = 1
        dataset_name = name

        # Check if the dataset name includes a replication factor
        if '*' in name:
            dataset_name, replication_factor = name.split('*')
            replication_factor = int(replication_factor)
            if replication_factor < 1:
                raise ValueError("Replication factor must be a positive integer.")

        # Construct the correct dataset path or name based on the source location
        dataset_path_or_name = ('FinGPT/fingpt-' if from_remote else 'data/fingpt-') + dataset_name
        if not os.path.exists(dataset_path_or_name) and not from_remote:
            raise FileNotFoundError(f"The dataset path {dataset_path_or_name} does not exist.")

        # Load the dataset
        try:
            tmp_dataset = datasets.load_dataset(dataset_path_or_name) if from_remote else datasets.load_from_disk(
                dataset_path_or_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load the dataset: {str(e)}")

        # Check for 'test' split and create it from 'train' if necessary
        if 'test' not in tmp_dataset:
            if 'train' in tmp_dataset:
                tmp_dataset = tmp_dataset['train']
                tmp_dataset = tmp_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
            else:
                raise ValueError("The dataset must contain a 'train' or 'test' split.")

        # Append the possibly replicated dataset to the list
        dataset_list.extend([tmp_dataset] * replication_factor)

    return dataset_list
    
base_model = 'llama2'
peft_model = 'FinGPT/fingpt-mt_llama2-7b_lora' if FROM_REMOTE else 'finetuned_models/MT-llama2-linear_202309241345'

model, tokenizer = load_model(base_model, peft_model, FROM_REMOTE)
test_demo(model, tokenizer)
