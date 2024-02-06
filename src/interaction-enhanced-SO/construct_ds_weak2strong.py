from datasets import load_dataset, load_from_disk
import datasets
from random import Random
import json
from tqdm import tqdm
from collections import Counter
import random
import numpy as np
import fire
import os
seed = 42
rng = Random(seed)

def process_dataset(ds, weak_label_ds, sample_num_perclass):
    new_ds = []
    ds_0 = []
    ds_1 = []

    for idx in tqdm(range(len(ds))):
        line = ds[idx]
        response = weak_label_ds[idx]['response']
        if response == 0:
            new_line = {
                'question': line['question'],
                'correct_answer': line['correct_answer'],
                'distractor1': line['distractor1'],
                'distractor2': line['distractor2'],
                'distractor3': line['distractor3'],
                'support': line['support'],
                'txt': line['txt'],
                'hard_label': 0,
                'soft_label': weak_label_ds[idx]['soft_label'],
                'true_label': line['hard_label']
            }
            ds_0.append(new_line)
      
        else:
            new_line = {
                'question': line['question'],
                'correct_answer': line['correct_answer'],
                'distractor1': line['distractor1'],
                'distractor2': line['distractor2'],
                'distractor3': line['distractor3'],
                'support': line['support'],
                'txt': line['txt'],
                'hard_label': 1,
                'soft_label': weak_label_ds[idx]['soft_label'],
                'true_label': line['hard_label']
            }
            ds_1.append(new_line)
          
  
    select_indices_0 = random.sample([i for i in range(len(ds_0))], sample_num_perclass)
    select_indices_1 = random.sample([i for i in range(len(ds_1))], sample_num_perclass)

    for index in select_indices_0:
        new_ds.append(ds_0[index])
 
    for index in select_indices_1:
        new_ds.append(ds_1[index])

    new_ds = datasets.Dataset.from_list(new_ds)
    new_ds = new_ds.shuffle()
    return new_ds


def main(data_path,data_type,weak_label_path,sample_num_perclass):
    ds = load_from_disk(os.path.join(data_path, data_type))
    weak_label = json.load(open(weak_label_path,'r',encoding='utf-8'))
    new_ds = process_dataset(ds,weak_label,sample_num_perclass)
    ds.save_to_disk(f'./sciq_weak/{date_type}/')

if __name__ == '__main__':
    fire.Fire(main)