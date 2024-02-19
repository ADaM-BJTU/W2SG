import pickle
import numpy
from tqdm import tqdm
from collections import Counter
import json
import datasets
import matplotlib.pyplot as plt
import random
import os
from datasets import load_dataset, load_from_disk

result=[]
def get_gt_labels(result_file,key):
    # result_file = "result/gpt2_origin/gpt2/weak_model_gt/gpt2/results.pkl"
    result_file =result_file
    result = pickle.load(open(result_file, 'rb'))
    inference_results = result[key]
    gt_labels = []
    for line in inference_results:
        gt_labels.append(line['gt_label'])
    return gt_labels

def get_trainset_results(result_file):
    TRAIN_SET = ["train_10", "train_11", "train_12", "train_13", "train_14", "train_15", "train_16", "train_17",
                 "train_18", "train_19", "train_110", "train_111", "train_112", "train_113", "train_114",
                 "train_115", "train_116", "train_117", "train_118", "train_119"]
    # TRAIN_SET = [i for i in range(20)]
    # TRAIN_SET = [i for i in range(10, 25)]
    results = []
    for layer in TRAIN_SET:
        result_file=result_file
        result = pickle.load(open(os.path.join(result_file,"weak_model_gt/gpt2/results.pkl"), 'rb'))
        results.append(result)
    return results
def process_result(results,trainset_size=6,key='inference_results'):
    # random.seed(42)
    # sampled_results = random.sample(results, trainset_size)
    test_size =len(results[0][key])
    # sampled_results = results[-trainset_size:]
    sampled_results = results[:trainset_size]
    predicted_labels_soft = []
    predicted_labels_hard = []
    all_soft_labels = []
    # for idx in tqdm(range(test_size)):
    for idx in range(test_size):
        soft_label = []
        hard_label = []
        for index in range(trainset_size):
            temp_inference_results = sampled_results[index][key]
            soft_label.append(temp_inference_results[idx]['soft_label'])
            hard_label.append(temp_inference_results[idx]['hard_label'])
        soft_label = numpy.array(soft_label)
        soft_label = numpy.mean(soft_label, axis=0)
        if soft_label[0] > soft_label[1]:
            predicted_labels_soft.append(0)
        else:
            predicted_labels_soft.append(1)
        collection_hard_labels = Counter(hard_label)
        predicted_labels_hard.append(collection_hard_labels.most_common(1)[0][0])
        all_soft_labels.append(soft_label.tolist())

    return predicted_labels_hard, predicted_labels_soft, all_soft_labels

def calculate_acc(gt_labels, predicted_labels):
    acc = 0
    for idx in range(len(gt_labels)):
        if gt_labels[idx] == predicted_labels[idx]:
            acc += 1
    acc = acc / len(gt_labels)
    return acc

def weak_to_strong_data(predicted_labels_hard=None,predicted_labels_soft=None,all_soft_labels=None,gt_labels=None,save_path=None):

    result_file = "./result/gpt2_origin/gpt2/weak_model_gt/_data2_yuhang_huggingface_hub_gpt2_/results.pkl"
    result = pickle.load(open(result_file, 'rb'))
    inference_result = result['inference_results']
    results = []
    for idx in tqdm(range(len(inference_result))):
        results.extend(
            [
                dict(
                    txt=inference_result[idx]['txt'],
                    input_ids=inference_result[idx]['input_ids'],
                    gt_label=inference_result[idx]['gt_label'],
                    hard_label=predicted_labels_hard[idx],
                    acc=len(gt_labels) == predicted_labels_soft[idx],
                    soft_label=all_soft_labels[idx]
                )
            ]
        )
    ds = datasets.Dataset.from_list(results)
    ds.save_to_disk(save_path)

if __name__ == '__main__':
    results_file= "your result folder"
    save_path="your save path"
    results = get_trainset_results(results_file)
    gt_labels = get_gt_labels("test_results")
    soft_acc_results=[]
    hard_acc_results=[] 
    for i in tqdm(range(1,21)):
        predicted_labels_hard,predicted_labels_soft,all_soft_labels = process_result(results,trainset_size=i,key="test_results")
        weak_to_strong_data(predicted_labels_soft=predicted_labels_soft, predicted_labels_hard=predicted_labels_hard,
                            all_soft_labels=all_soft_labels, gt_labels=gt_labels, save_path=os.path.join(save_path,f"{i}"))
        soft_acc=calculate_acc(gt_labels,predicted_labels_soft)
        hard_acc=calculate_acc(gt_labels,predicted_labels_hard)
        soft_acc_results.append(soft_acc)
        hard_acc_results.append(hard_acc)
    print("gpt2_soft_ : ",soft_acc_results)
    print("gpt2_hard_ : ",hard_acc_results)
