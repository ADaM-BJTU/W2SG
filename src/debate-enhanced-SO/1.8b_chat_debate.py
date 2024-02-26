import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from qwen_generation_utils import make_context

tokenizer = AutoTokenizer.from_pretrained("model_name", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("model_name", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("model_name", pad_token_id=tokenizer.pad_token_id)
model.generation_config.max_new_tokens = 10

def encode_prompt(test_line):
    prompt = f'''Please read the context first and then determine if the following question's answer is correct. 
    If it is correct, reply with solely 'Yes'. If it is incorrect, reply with solely 'No'.
    Context:One person think it is correct for the reason {test_line['correct_explanation']}
            Anothre person think it is incorrect for the reason {test_line['incorrect_explanation']}
    Question:{test_line['question']}
    Answer:{test_line['answer']}.

    Output:
    '''
    return prompt

def get_output(txt):
    prompt = encode_prompt(txt)
    raw_text, _ = make_context(
        tokenizer,
        prompt,
        system="You are a helpful assistant.",
        chat_format='chatml'
    )
    inputs = tokenizer(raw_text, return_tensors='pt')
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs)
    outputs = tokenizer.decode(outputs.cpu()[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return outputs

if __name__ == '__main__':
    with open('test_data_path', 'r', encoding='utf-8') as file:
        test_data = json.load(file)
        print(len(test_data))

    write_data = []
    count = 0
    label_map = {1:'Yes',0:'No'}
    for idx, line in enumerate(test_data):
        process_response = get_output(line)
        response = 1 if 'Yes' in process_response or 'yes' in process_response else 0
        if response == line['hard_label']:
            count += 1
        write_data.append({
            "question": line["question"],
            "answer": line["answer"],
            "correct_explanation": line["correct_explanation"],
            "incorrect_explanation": line["incorrect_explanation"],
            'hard_label':line['hard_label'],
            'process_response':process_response,
            'response':response
        })
        print(line)
        print(process_response)
        print(response)
        print(line['hard_label'])
        print(1.0*count/(idx+1))
        print('-------------------')
    print((count + 0.0)/len(test_data))
    json.dump(write_data, open("output_json_path", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
