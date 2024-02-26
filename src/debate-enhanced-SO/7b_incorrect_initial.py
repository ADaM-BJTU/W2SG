import json
import os
import random
from tqdm import tqdm
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("model_name", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model_name", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("model_name", trust_remote_code=True)
model.generation_config.max_new_tokens = 32

# Encode prompt
def encode_prompt(question_answer, examples):
    # ICL demonstrations
    format_examples = []
    examples = random.sample(examples, 2)
    for line in examples:
        input = line['txt']
        format_examples.append(f"Input:{input}\nExplanation:{line['response']}")
    format_examples = "\n\n".join(format_examples)
    question_answer = question_answer.replace("Q:", "question:").replace("A:", "answer:")
    prompt = f'''There is a question followed by an answer. Assuming the answer is incorrect, please give your explanation.\nExamples are given below.\n
    {format_examples}\n\nInput: {question_answer}\nExplanation:
    '''
    return prompt

# Generate result
def generate_result(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    return tokenizer.decode(pred.cpu()[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

# Post-process 
def post_process_response(response):
    response = response.split('Question:')[0].strip()
    response = response.split('Input:')[0].strip()
    return response

# Main function
if __name__ == '__main__':
    # Load test data
    test_data = []
    with open('test_data_path', 'r', encoding='utf-8') as test_file:
        for line in test_file.readlines():
            json_line = json.loads(line)
            test_data.append(json_line)

    # Load demo data
    demo_data = json.load(open('demo_data_path', 'r', encoding='utf-8'))

    write_data = []
    for line in tqdm(test_data):
        prompt = encode_prompt(line['txt'], demo_data)
        result = ""
        while result == "":
            raw_result = generate_result(prompt)
            result = post_process_response(raw_result)
        write_data.append({
            "question": line['question'],
            "answer": line['txt'].split("A:")[-1],
            "explanation": result,
            "hard_label": line['hard_label']
        })
        print("raw response: " + raw_result)
        print('*******************')
        print("explanation: " + result)
        print('-------------------')

    # Write data to JSON file
    json.dump(write_data,
              open("output_json_path", "w", encoding="utf-8"),
              indent=4,
              ensure_ascii=False)