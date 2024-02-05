import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os
import torch
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
from tqdm import tqdm
import datasets
import random
import fire

def encode_prompt(question_anwer, examples, context_example_num=2):
    format_examples = []
    examples = random.sample(examples, context_example_num)
    for line in examples:
        input = line['txt'].split("A:")[0].replace("Q:","")
        format_examples.append(f"Question: {input}\nKnowledge: {line['response']}")
    format_examples = "\n\n".join(format_examples)
    question_anwer = question_anwer.split("A:")[0].replace("Q:","")
    prompt = \
    f'''{format_examples}\n\nQuestion: {question_anwer}\nKnowledge:
    '''
    return prompt

def generate_result(prompt,tokenizer,model):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    return tokenizer.decode(pred.cpu()[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)



# each data type should generate a background knowledge file
def main(model_path, data_path, demo_data_path, data_type='train2',context_example_num=2):
    # load data
    data = datasets.load_from_disk(os.path.join(data_path, data_type))
    demo_data = json.load(open(demo_data_path,'r',encoding='utf-8'))
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.generation_config.max_new_tokens = 64

    # generate backgroud knowledge
    write_data = []
    for line in tqdm(data):
        prompt = encode_prompt(line['txt'], demo_data, context_example_num=2)
        result = ""
        while result == "":
            raw_result = generate_result(prompt,tokenizer,model)
            result = raw_result.split('Question:')[0].strip()
        write_data.append({
            "txt": line['txt'],
            "knowledge": result,
        })
        print("txt: " + line['txt'])
        print("knowledge: " + result)
        print('===============================')
        json.dump(write_data,
                    open(f"background_knowledge_{data_type}.json","w", encoding="utf-8"),
                    indent=4,
                    ensure_ascii=False)
        
if __name__ == "__main__":
    fire.Fire(main)


    