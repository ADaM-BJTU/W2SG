import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from bagging import get_gt_labels,get_trainset_results,process_result,calculate_acc,weak_to_strong_data
SAMPLE_NUM=[2000]

MODEL=["gpt2","gpt2-medium"]
Train_set=["train_10","train_11","train_12","train_13","train_14","train_15","train_16","train_17","train_18","train_19",
           "train_110","train_111","train_112","train_113","train_114","train_115","train_116","train_117","train_118","train_119"]

for model in MODEL:
    for sample_num in SAMPLE_NUM:
        for train_set in Train_set:
            print("training on {}".format(train_set))
            dataset = '{}_20'.format(str(sample_num))
            commend = f'''python train_weak.py \
                        --weak_model_size /path_to_the_models/ \
                        --train1_name  ./sciq/num_{sample_num}/{train_set}/train/ \
                        --train2_name  ./sciq/train2/ \
                        --eval_name  ./sciq/num_{sample_num}/{train_set}/valid/ \
                        --infer_valid_name ./sciq/val/ \
                        --test_name  ./sciq/test/ \
                        --results_folder  ./bagging_result/{model}/{dataset}/{train_set} \
                        --gt_epochs 5 \
                        --eval_every 300 '''
            os.system(commend)

SAMPLE_NUM=[2000,2500,3000,3500,4000]
MODEL=["gpt2"]
for model in MODEL:
    for sample_num in SAMPLE_NUM:
        for train_set in Train_set:
            print("training on {}".format(train_set))
            dataset = '{}_20'.format(str(sample_num))
            commend = f'''python train_weak.py \
                        --weak_model_size /path_to_the_models/ \
                        --train1_name  ./sciq/num_{sample_num}/{train_set}/train/ \
                        --train2_name  ./sciq/train2/ \
                        --eval_name  ./sciq/num_{sample_num}/{train_set}/valid/ \
                        --infer_valid_name ./sciq/val/ \
                        --test_name  ./sciq/test/ \
                        --results_folder  ./bagging_result/{model}/{dataset}/{train_set} \
                        --gt_epochs 5 \
                        --eval_every 300 '''
            os.system(commend)