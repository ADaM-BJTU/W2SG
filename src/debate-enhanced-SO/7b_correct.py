import json
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("model_name", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model_name", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("model_name", trust_remote_code=True)
model.generation_config.max_new_tokens = 32

def encode_prompt(test_line, examples):
    examples = random.sample(examples, 2)
    format_examples = [
        f"Input:Question: {line['question']}\nAnswer: {line['answer']}\nExplanation from another agent:{line['explanation_from_other']}\nOutput:{line['new_explanation']}"
        for line in examples
    ]
    format_examples = "\n\n".join(format_examples)
    prompt = f'''There is a question followed by an answer. Another agent think the answer is incorrect, and its explanation is given below. Please use its explanation as additional information to update your explanation.
    Examples are given below.
    {format_examples}

    Input:Question: {test_line['question']}\nAnswer: {test_line['answer']}\nExplanation from another agent:{test_line['incorrect_explanation']}
    Output:
    '''
    return prompt

def generate_result(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    return tokenizer.decode(pred.cpu()[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

def post_process_response(response):
    response = response.split('Question:')[0].strip()
    response = response.split('Input:')[0].strip()
    return response

if __name__ == '__main__':
    with open('test_data_path', 'r', encoding='utf-8') as test_file:
        test_data = json.load(test_file)
        
    
    with open('demo_data_path', 'r', encoding='utf-8') as file1:
        demo_data = json.load(file1)
       
    write_data = []
    for line in tqdm(test_data):
        prompt = encode_prompt(line, demo_data)
        result = ""
        while result == "":
            raw_result = generate_result(prompt)
            result = post_process_response(raw_result)
        write_data.append({
            "question": line['question'],
            "answer":line['answer'],
            "explanation": result,
            "hard_label":line['hard_label']
        })
        print("raw response: " + raw_result)
        print('*******************')
        print("explanation: " + result)
        print('-------------------')

    json.dump(write_data,
                open("output_json_path","w", encoding="utf-8"),
                indent=4,
                ensure_ascii=False)
