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

def encode_prompt(question_anwer, knowledge):
    Input = question_anwer.replace("Q:","question:").replace("A:","answer:")
    if knowledge.strip() == '':
        prompt = f'''Please determine if the following question's answer is correct. If it is correct, reply with 'Yes'. If it is incorrect, reply with 'No'.
        Input: {question_anwer}
        Output:'''
    else:
        prompt =f'''Please determine if the following question's answer is correct based on the context. If it is correct, output "Yes". If it is incorrect, output "No".
        Context: {knowledge}
        Input: {Input}
        Output:'''
    return prompt

def calculate_accuracy(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    accuracy = np.sum(ground_truth == predictions) / len(ground_truth)
    return accuracy

def get_output(txt,knowledge,tokenizer,model):
    prompt = encode_prompt(txt, knowledge)
    prompt, _ = make_context(
        tokenizer,
        prompt,
        system="You are a helpful assistant.",
        chat_format = 'chatml'
    )
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs)
    outputs = tokenizer.decode(outputs.cpu()[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return outputs

def main(model_path, data_path, data_type):
    # load data 
    data = datasets.load_from_disk(
        os.path.join(data_path, data_type))

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    pad_token='<|extra_0|>',
    eos_token='<|endoftext|>',
    padding_side='left',
    trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=tokenizer.pad_token_id)
    model.generation_config.max_new_tokens=10

    knowledge_data = json.load(open(f'background_knowledge_{data_type}.json','r',encoding='utf-8'))

    # compare the output with the ground truth
    gt_labels = []
    predict_labels = []
    for idx in tqdm(range(len(data))):
        line = data[idx]
        knowledge = knowledge_data[idx]['knowledge']
        response = get_output(line['txt'], knowledge,tokenizer,model) 
        process_response = response.split('\n')[0]
        if 'Yes' in process_response:
            response = 1
        else:
            response = 0

        gt_labels.append(line['hard_label'])
        predict_labels.append(response)
        print(f"Accuracy: {calculate_accuracy(gt_labels, predict_labels)}") 

if __name__ == "__main__":  
    fire.Fire(main)