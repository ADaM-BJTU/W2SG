import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os
import torch
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
import datasets
from tqdm import tqdm
import fire
import random

def encode_prompt(question_anwer, examples):
    question = question_anwer.split("A:")[0].replace("Q:","")
    prompt = \
    f'''Please provide the background knowledge to answer the
following question. Limit your reply to 30 words.
    Input: {question}
    Output:
    '''
    return prompt

def get_output(txt,tokenizer,model):
    prompt = encode_prompt(txt, examples=[])
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
    
def main(model_path, data_path, example_set_num):
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
    model.generation_config.max_new_tokens=32
    # load data
    data = datasets.load_from_disk(data_path)
    sample_index = random.sample(list(range(0,len(data))), example_set_num)
    # select a random sample of data
    sample_data = data.select(sample_index)
  
    write_data = []
    for idx in tqdm(range(len(sample_data))):
        line = sample_data[idx]
        response = "a"
        # ensure the response is an entire sentence
        while response[len(response)-1] != '.':
            response = get_output(line['txt'],tokenizer,model)
        write_data.append({
            'txt':line['txt'],
            'response':response
        })
        json.dump(write_data, open("demos.json", "w", encoding="utf-8"), ensure_ascii=False,indent=4)

if __name__ == "__main__":
    fire.Fire(main)
    
    
