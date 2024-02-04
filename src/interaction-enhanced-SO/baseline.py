import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os
import torch
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
import datasets
from tqdm import tqdm
import fire
import numpy as np

def encode_prompt(question_anwer):
    question_anwer = question_anwer.replace('Q:','question:').replace('A:','answer:')
    prompt = \
    f'''Please determine if the following question's answer is correct. If it is correct, reply with 'Yes'. If it is incorrect, reply with 'No'.
    Input: {question_anwer}
    Output:
    '''
    return prompt

def get_output(txt,tokenizer,model):
    prompt = encode_prompt(txt)
    raw_text, _ = make_context(
        tokenizer,
        prompt,
        system="You are a helpful assistant.",
        chat_format = 'chatml'
    )
    inputs = tokenizer(raw_text, return_tensors='pt')
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs)
    outputs = tokenizer.decode(outputs.cpu()[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return outputs


def calculate_accuracy(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    accuracy = np.sum(ground_truth == predictions) / len(ground_truth)
    return accuracy

def main(model_path, test_data_path):
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    pad_token='<|extra_0|>',
    eos_token='<|endoftext|>',
    padding_side='left',
    trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
    model.generation_config.max_new_tokens=10
    # load data
    test_data = datasets.load_from_disk(test_data_path)

    gt_labels = []
    predict_labels = []
    for idx in tqdm(range(len(test_data))):
        line = test_data[idx]
        process_response = get_output(line['txt'],tokenizer,model).split('\n')[0]
        if 'Yes' in process_response:
            response = 1
        else:
            response = 0
        gt_labels.append(line['hard_label'])
        predict_labels.append(response)

        print(f"Accuracy: {calculate_accuracy(gt_labels, predict_labels)}") 

if __name__ == "__main__":
    fire.Fire(main)