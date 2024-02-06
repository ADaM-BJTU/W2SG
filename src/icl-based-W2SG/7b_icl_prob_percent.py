import json
import os
import re 
import pandas as pd 
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig 
import torch 
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from openicl import DatasetReader 
import random 
import json 
from datasets import DatasetDict 
from weak_to_strong.common import clear_mem
from weak_to_strong.icl_retriever import load_retriever, BASE
from weak_to_strong.icl_gen_inferencer import GenInferencer
from weak_to_strong.template import load_template
# Load the modelx
def maybe_load_model(model_name):
    if 1:
        # assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_name}" #change
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            num_labels=2,
            device_map="auto", 
            trust_remote_code=True
        ) 
    else:
        minibatch_size_per_device=1;batch_size=2
        model = AutoModelForCausalLM.from_pretrained(
           model_name, num_labels=2 ).to("cuda")  
        # data parallel:  currently not supported with model parallel
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), 2)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )
    return model 

def encode_prompt(test_line, train_ds, examples,error_num=0): 
    ctx_template = "{txt}\n{hard_label} (Confidence: {prob:.2f})"   
    error_id = random.sample(range(len(examples)), error_num) if error_num else [] 
    examples = "\n\n".join([ctx_template.format(txt=train_ds[exp]['txt'], hard_label=train_ds[exp]['hard_label']^1,prob = train_ds[exp]['prob'] ) if ii in error_id \
                            else ctx_template.format(txt=train_ds[exp]['txt'], hard_label=train_ds[exp]['hard_label'],prob = train_ds[exp]['prob']) for ii,exp in enumerate(examples) ])
    prompt = f'''There is a science knowledge question, followed by an answer. Respond with 1 if the answer is correct, and with 0 otherwise. Note that there may be errors in the answers to the contextual examples.

{examples}

{test_line['txt']}
''' 
    return prompt, error_id
def get_yes_or_no_probs(outputs,tokenizer  ):
    # prob with yes or no
    probs_yes_or_no = dict();yes_or_no_list = set(["1", "0"])
    word2label = {"1":1,"0":0,"yes":1,"no":0,"Yes":1,"No":0}
    wordline = outputs['scores'][0] 
    wordline[0]  = torch.nn.functional.softmax(wordline[0], dim=-1)
    sortprob = wordline[0].argsort(descending=True)
    num=0
    for id_ in sortprob:
        if tokenizer.decode([id_]).strip()  in yes_or_no_list: 
            label_name = word2label[tokenizer.decode([id_]).strip().lower()]
            probs_yes_or_no.setdefault(label_name,0)
            probs_yes_or_no[label_name] = max(wordline[0][id_].item(),probs_yes_or_no[label_name ])
            num+=1
        if num ==len(yes_or_no_list):
            break 
    return probs_yes_or_no
def generate_result_and_prob(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, return_dict_in_generate=True,  output_scores=True )
    probs = get_yes_or_no_probs(outputs,tokenizer )
    pred = outputs['sequences']
    
    return tokenizer.decode(pred.cpu()[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True), probs
 
def post_process_qwen_response(prompt,response): 
    response = response.split(prompt)[-1].strip()
    # 匹配label，也就是第一个匹配到的'0'或'1'
    label = re.search(r'0|1', response)
    if label is not None:
        label = label.group() 
        label = eval(label)
    else:
        label = ''

    return label, response

def main(
        batch_size: int = 32, 
        weak_model_size: str = "../hub/qwen-1.8B-chat/", 
        strong_model_size: str = "../hub/qwen-7B/",     
        icl_process_batch_size: int = 4,
        results_folder: str = "./tmp/results_icl_new",    
        retriever_name: str = "stochastic_random",  #change stochastic_random
        ice_num: int = 10, 
        Weak2Strong: bool = True, 
        Strong: bool = False,
        weak_ds_path: str = 'dataset/sciq/1.8b_eval_naive_logits.json',
        weak_support_ds_path: str = 'dataset/sciq/1.8b_eval_q_support_logits.json',
        test_ds_path: str = 'dataset/sciq/test',
        train2_ds_path: str = 'dataset/sciq/train2',
        weak_type_list: str="native,support",
        error_num: int=0,
        icl_random_seed: int = 43,  #random随机采样的seed 是np.random.seed 默认43 change
        main_func_seed: int = 43, # main函数的seed 是random.seed，主要是用于strong那列时，随机挑选错误示例的 默认0(icl10之后默认是=icl_random_seed)
        test_num: int =2000,
        max_every_class: dict = None, # {0: 3, 1:7}
        percent: str="100" #默认使用全部数据
    ):
    # 输出所有main的参数
    print("==== "*20)
    print("main args: ", locals())
    # max_every_class = eval(max_every_class) if max_every_class else None
    every_class_path_name = "_"+"_".join([f"{k}({v})"+"_" for k,v in max_every_class.items()]) if max_every_class else ""
    """load model"""
    tokenizer = AutoTokenizer.from_pretrained(strong_model_size, trust_remote_code=True)
    model = maybe_load_model(strong_model_size ).eval()
    model.generation_config = GenerationConfig.from_pretrained(strong_model_size, trust_remote_code=True)
    model.generation_config.max_new_tokens = 64
 
    def icl_model( 
        model_name: str,
        train_ds: list, # context_ds
        test_ds: list, # test_ds  
        subpath: str,      
        retriever_name: str,
        ice_num: int = 5, 
        type_: str in ["native", "support", "strong"] = "native", 
        error_num: int = 0,
        ):
        save_path = os.path.join(results_folder, subpath)
        os.makedirs(save_path, exist_ok=True) 

        """icl context choice"""
        retriever = load_retriever(f"{BASE}/sciq/",retriever_name=retriever_name) 
        inferencer = GenInferencer(model_name=model_name,model = model,tokenizer=tokenizer,batch_size=icl_process_batch_size,generation_kwargs={}) 
        template = load_template(f"{BASE}/sciq/")

        print("context example:", train_ds[0]['txt']) 
        print("test example:", test_ds[0]['txt'])
        dataset_dict = DatasetDict({"train": Dataset.from_pandas(pd.DataFrame(train_ds)), "test": Dataset.from_pandas(pd.DataFrame(test_ds)) })
        # opencil_dataset = DatasetReader(dataset_dict, input_columns=['txt',"hard_label"], output_column="hard_label"  )
        opencil_dataset = DatasetReader(dataset_dict, input_columns=['txt',"hard_label"], output_column="hard_label",input_template=template.context_template  )
        if "qwen" in retriever_name :
            retriever = retriever.create_context_retriever(opencil_dataset, ice_num=ice_num, batch_size=batch_size,seed= icl_random_seed, sentence_transformers_model=model, tokenizer=tokenizer)
        elif 'so' in retriever_name:
            weak_model = maybe_load_model(weak_model_size).eval()
            weak_model.generation_config = GenerationConfig.from_pretrained(weak_model_size, pad_token_id=tokenizer.pad_token_id)
            weak_model.generation_config.max_new_tokens=256
            weak_model.generation_config.top_p=0.8
            wea_tokenizer =  AutoTokenizer.from_pretrained(
                                weak_model_size ,
                                pad_token='<|extra_0|>',
                                eos_token='<|endoftext|>',
                                padding_side='left',
                                trust_remote_code=True
                            )
            retriever = retriever.create_context_retriever(opencil_dataset, ice_num=ice_num, batch_size=batch_size,seed= icl_random_seed, sentence_transformers_model=weak_model, tokenizer=wea_tokenizer)
        else:
            retriever = retriever.create_context_retriever(opencil_dataset, ice_num=ice_num, batch_size=batch_size,seed= icl_random_seed, )
        
        """得到每个test的icl_context_idx"""
        if 'stochastic' in retriever_name :
            ice_idx_list = inferencer.inference(retriever=retriever, output_json_filepath=save_path,output_json_filename="icl_results",return_ice_idx=True, ice_template=template.context_template,label_column="hard_label",max_every_class=max_every_class )
        else:
            ice_idx_list = inferencer.inference(retriever=retriever, output_json_filepath=save_path,output_json_filename="icl_results",return_ice_idx=True, ice_template=template.context_template  )
        try:
            write_data = json.load(open(f"{save_path}/prob3_{percent}%_{type_}.json","r", encoding="utf-8"))
        except:
            write_data = []
        ex = encode_prompt(test_data[0], train_ds, ice_idx_list[0], error_num)
        print("prompt\n", ex[0],"\n",ex[1] )
        
        ice_idx_list=ice_idx_list[len(write_data):]   
        for idx,line in enumerate(tqdm(test_data[len(write_data):])): 
            prompt, error_id = encode_prompt(line, train_ds, ice_idx_list[idx],error_num=error_num)
            result = ""   
            if 1:
                raw_result, prob = generate_result_and_prob(prompt, tokenizer, model) 
                result, response = post_process_qwen_response(prompt,raw_result) 

            write_data.append({
                "question":line['question'],
                "txt": line['txt'],
                "hard_acc": line['hard_label']==result,
                "soft_acc": line['hard_label']==max(prob, key=prob.get),
                "label": line['hard_label'],
                "hard_pred": result,
                "soft_pred":prob,
                "error_id": error_id,
                "response": response,
                "context": [train_ds[i] for i in ice_idx_list[idx]] 
            })
            
            json.dump(write_data,
                        open(f"{save_path}/prob3_{percent}%_{type_}.json","w", encoding="utf-8"),
                        indent=4,
                        ensure_ascii=False)
        hard_accs = [line['hard_acc'] for line in  write_data]
        soft_accs = [line['soft_acc'] for line in  write_data]
        # try to clean up memory
        clear_mem()
        return sum(hard_accs)/len(hard_accs), sum(soft_accs)/len(soft_accs)
 
    model_name = strong_model_size.replace("/","_")
    type_list=weak_type_list.split(",") if type(weak_type_list)==str else weak_type_list
    # # load dataset
    weak_native_ds = [{**line,'hard_label':line['response'],'prob': line["soft_label"][line['response']]} for line in json.load(open(weak_ds_path))]
    weak_support_ds = [{**line,'hard_label':line['response'],'prob': line["soft_label"][line['response']]  } for line in json.load(open(weak_support_ds_path))]
    test_data = [line for line in load_from_disk(test_ds_path)][:test_num]
    train2_ds = [line for line in load_from_disk(train2_ds_path)] 
    random.seed(main_func_seed)
    torch.manual_seed(main_func_seed) 
    print("train2 size: ",len(weak_native_ds), "test size: ",len(test_data)) 

    # 1. native Weak2Strong
    if Weak2Strong:
        if "native" in type_list:
            subpath=os.path.join(retriever_name,f"icl{ice_num}-seed{icl_random_seed}_main-func-seed{main_func_seed}",
                            "strong_model_transfer",
                            f"{weak_model_size.replace('/', '_')}_{strong_model_size.replace('/', '_')}{every_class_path_name}",
                        ) 
            hard_accs, soft_accs=icl_model(model_name, weak_native_ds, test_data, subpath, ice_num=ice_num, retriever_name=retriever_name,type_="native")
            print("--"*20)
            print("native hard acc: ", hard_accs, "\nnative soft acc: ", soft_accs)
        if "support" in type_list:
            subpath=os.path.join(retriever_name,f"icl{ice_num}-seed{icl_random_seed}_main-func-seed{main_func_seed}",
                            "strong_model_transfer",
                            f"{weak_model_size.replace('/', '_')}_{strong_model_size.replace('/', '_')}{every_class_path_name}",
                        ) 
            hard_accs, soft_accs=icl_model(model_name, weak_support_ds, test_data, subpath, ice_num=ice_num, retriever_name=retriever_name,type_="support")
            print("--"*20)
            print("support hard acc: ", hard_accs, "\nsupport soft acc: ", soft_accs)
    # 2. Strong
    if Strong:
        if error_num!=0:
            subpath=os.path.join(retriever_name,f"icl{ice_num}-seed{icl_random_seed}_main-func-seed{main_func_seed}",
                        "strong_model",
                        f"{weak_model_size.replace('/', '_')}_{strong_model_size.replace('/', '_')}_error{error_num}{every_class_path_name}", 
                    )
        else:
            subpath=os.path.join(retriever_name,f"icl{ice_num}-seed{icl_random_seed}_main-func-seed{main_func_seed}",
                        "strong_model",
                        f"{weak_model_size.replace('/', '_')}_{strong_model_size.replace('/', '_')}{every_class_path_name}",
                    ) 
        hard_accs, soft_accs=icl_model(model_name, train2_ds, test_data, subpath, ice_num=ice_num, retriever_name=retriever_name,type_="strong",error_num=error_num)
        print("--"*20)
        print(f"error{error_num} strong hard acc: ", hard_accs, f"\nerror{error_num} strong soft acc: ", soft_accs)
    # 清除缓存
if __name__ == '__main__':
    fire.Fire(main)
    