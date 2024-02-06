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

def calculate_accuracy(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    accuracy = np.sum(ground_truth == predictions) / len(ground_truth)
    return accuracy

def generate_result_and_prob(txt,tokenizer,model):
    prompt = encode_prompt(txt)
    prompt, _ = make_context(
        tokenizer,
        prompt,
        system="You are a helpful assistant.",
        chat_format = 'chatml'
    )
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, return_dict_in_generate=True,  output_scores=True)
    probs = get_yes_or_no_probs(outputs,tokenizer )
    pred = outputs['sequences']
    return tokenizer.decode(pred.cpu()[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True), probs

def get_yes_or_no_probs(outputs,tokenizer):
    # prob with yes or no
    probs_yes_or_no = dict();yes_or_no_list = set(["Yes", "No"])
    word2label = {"1":1,"0":0,"yes":1,"no":0,"Yes":1,"No":0}
    wordline = outputs['scores'][0]
    prob_yes = wordline[0][tokenizer("Yes")["input_ids"][0]]
    prob_no = wordline[0][tokenizer("No")["input_ids"][0]]
    probs_yes_or_no = {1:prob_yes,0:prob_no} 
    return probs_yes_or_no



def main(model_path, data_path, data_type):
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

    data = datasets.load_from_disk(os.path.join(data_path, data_type))

    write_data = []
    gt_labels = []
    predict_labels = []

    for idx in tqdm(range(len(data))):
        line = data[idx]
        result,prob = generate_result_and_prob(line['txt'],tokenizer,model)
        process_response = result.split('\n')[0]
        log_prob =  torch.tensor([prob[0],prob[1]])
        log_prob = log_prob.softmax(0)
        if log_prob[1] > log_prob[0]:
            response = 1
        else:
            response = 0
    
        write_data.append({
            'txt':line['txt'],
            'label':line['hard_label'],
            'response':response,
            'soft_label':log_prob.tolist(),
        })
        gt_labels.append(line['hard_label'])
        predict_labels.append(response)

        json.dump(write_data, open(f"baseline_{data_type}_logits.json", "w", encoding="utf-8"), ensure_ascii=False,indent=4)

        print(f"Accuracy: {calculate_accuracy(gt_labels, predict_labels)}") 


if __name__ == "__main__":
    fire.Fire(main)
