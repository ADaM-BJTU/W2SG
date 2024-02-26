from datasets import load_dataset, load_from_disk
import datasets
from random import Random
import random
from tqdm import tqdm
seed = 42
rng = Random(seed)

def process_dataset(ds,sample_num=5000,sample_seed=42):

    def sample_list(input_list, num_samples, random_seed, sampling_method = "random"):
        if sampling_method == "random":
            random.seed(random_seed) # random sampling
        elif sampling_method == "bootstrap":
            random.choices(random_seed) # bootstrap sampling
        return random.sample(input_list, num_samples)
    
    sampled_list = sample_list(ds,sample_num, sample_seed)
    valid_list =[]

    sampled_ds = datasets.Dataset.from_list(sampled_list)
    dataset_ds=datasets.Dataset.from_list(ds)
    for i in tqdm(dataset_ds):
        if i["question"] not in sampled_ds["question"]:
            valid_list.append(i)
                
    valid_set=datasets.Dataset.from_list(valid_list)
    new_ds = []

    for line in sampled_ds:
        false_answer = rng.choice([line["distractor1"], line["distractor2"], line["distractor3"]])
        false_txt = f"Q: {line['question']} A: {false_answer}"
        false_line = {
            'question': line['question'],
            'correct_answer': line['correct_answer'],
            'distractor1': line['distractor1'],
            'distractor2': line['distractor2'],
            'distractor3': line['distractor3'],
            'support': line['support'],
            'txt': false_txt,
            'hard_label': 0,
            'soft_label': [1.0, 0.0]
        }
        new_ds.append(line)
        new_ds.append(false_line)
    new_valid_list=[]
    for line in valid_set:
        false_answer = rng.choice([line["distractor1"], line["distractor2"], line["distractor3"]])
        false_txt = f"Q: {line['question']} A: {false_answer}"
        false_line = {
            'question': line['question'],
            'correct_answer': line['correct_answer'],
            'distractor1': line['distractor1'],
            'distractor2': line['distractor2'],
            'distractor3': line['distractor3'],
            'support': line['support'],
            'txt': false_txt,
            'hard_label': 0,
            'soft_label': [1.0, 0.0]
        }
        new_valid_list.append(line)
        new_valid_list.append(false_line)
    new_ds = datasets.Dataset.from_list(new_ds)
    new_valid_ds=datasets.Dataset.from_list(new_valid_list)
    new_ds = new_ds.shuffle()
    print("new_ds:",len(new_ds)," new_valid_ds:",len(new_valid_ds))
    return new_ds,new_valid_ds

neighbor_num=20
sample_seed=[random.randint(1,10000) for _ in range(neighbor_num)]

dataset=load_from_disk("./sciq/train1")
true_data=[]
for data in dataset:
    if data["hard_label"]==1:
        true_data.append(data)

SAMPLE_NUM=[5000]

sampled_dataset=[]
for sample_num in SAMPLE_NUM:
    for i in range(neighbor_num):
        tmp_train,valid=process_dataset(true_data,sample_num,sample_seed[i])
        tmp_train.save_to_disk("./sciq/num_{}/train_1{}/train".format(str(sample_num),i))
        valid.save_to_disk("./sciq/num_{}/train_1{}/valid".format(str(sample_num),i))
