from datasets import load_dataset, load_from_disk
import datasets
from random import Random
seed = 42
rng = Random(seed)
# dataset = load_dataset('boolq',cache_dir="./boolq")

# train_dataset, test_ds = dataset["train"], dataset["test"]
# split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
# train1_ds, train2_ds = split_data["train"], split_data["test"]
# print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))

# train1_ds.save_to_disk('./sciq/my/train1')
# train2_ds.save_to_disk('./sciq/my/train2')
# test_ds.save_to_disk('./sciq/my/test')

# dataset = load_dataset('sciq',cache_dir="./sciq")
dataset = load_from_disk('/data2/zhangj/weak2strong/dataset/sciq')
trainset = dataset['train']
testset = dataset['test']

question = []
our_trainset = []
all_train_num = 10000
for line in trainset:
    if line['question'] not in question:
        our_trainset.append(line)
        question.append(line)
    if len(our_trainset) == all_train_num:
        break
our_trainset = datasets.Dataset.from_list(our_trainset)
split_data = our_trainset.train_test_split(test_size=0.5, seed=seed)
train1_ds, train2_ds = split_data["train"], split_data["test"]

def process_dataset(ds):
    new_ds = []
    for line in ds:
        true_txt = f"Q: {line['question']} A: {line['correct_answer']}"
        true_line = {
            'question': line['question'],
            'correct_answer': line['correct_answer'],
            'distractor1': line['distractor1'],
            'distractor2': line['distractor2'],
            'distractor3': line['distractor3'],
            'support': line['support'],
            'txt': true_txt,
            'hard_label': 1,
            'soft_label': [0.0, 1.0]
        }

       
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
        new_ds.append(true_line)
        new_ds.append(false_line)
    new_ds = datasets.Dataset.from_list(new_ds)
    new_ds = new_ds.shuffle()
    return new_ds

train1_ds = process_dataset(train1_ds)
train2_ds = process_dataset(train2_ds)
testset = process_dataset(testset)
print(len(train1_ds), len(train2_ds), len(testset))
train1_ds.save_to_disk('./sciq/train1')
train2_ds.save_to_disk('./sciq/train2')
testset.save_to_disk('./sciq/test')