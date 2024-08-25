from fastchat.conversation import get_conv_template
import numpy as np
import torch
import pandas as pd
import random
from datasets import load_dataset


llama3_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_message }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
llama_system_prompt = "You are a helpful and harmless assistant"
mistral_system_prompt = ("Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid "
                         "harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and "
                         "positivity.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_conv(model_name, goal):
    if model_name in ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "Llama-2-70b-chat-hf"]:
        conv = get_conv_template("llama-2")
        conv.set_system_message(llama_system_prompt)
    elif model_name in ["Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct"]:
        conv = get_conv_template("llama-3")
        conv.set_system_message(llama_system_prompt)
    elif model_name in ["vicuna-7b-v1.5", "vicuna-13b-v1.5",
                        "vicuna-7b-v1.5-16k", "vicuna-13b-v1.5-16k", "vicuna-7b-v1.5-32k"]:
        conv = get_conv_template("vicuna_v1.1")
    elif model_name in ["Mistral-7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.2"]:
        conv = get_conv_template("mistral")
        conv.set_system_message(mistral_system_prompt)
    elif model_name in ["Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf",
                        "Llama-3-8B", "Llama-3-70B", "Mistral-7B-v0.1"]:
        return f"{goal}"
    elif model_name in ['falcon-7b', 'falcon-7b-instruct']:
        conv = get_conv_template("falcon")
        conv.set_system_message(llama_system_prompt)
    else:
        raise ValueError("Your model is not correct")
    conv.append_message(conv.roles[0], goal)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def get_data(file_path, shuffle_seed=None, use_conv=False):
    data_df = pd.read_csv(file_path)
    data_list = []
    for i, r in data_df.iterrows():
        if r['goal'][-1] != "." and r['goal'][-1] != "?":
            data_list.append(r['goal'] + ".")
        else:
            data_list.append(r['goal'])
    if shuffle_seed:
        set_seed(shuffle_seed)
        random.shuffle(data_list)

    return data_list


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset("/root/wanda/datasets", data_files={'train': 'c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset("/root/wanda/datasets", data_files={'validation': 'c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)