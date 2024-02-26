import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from bagging import get_gt_labels,get_trainset_results,process_result,calculate_acc,weak_to_strong_data



if __name__ == '__main__':
    SAMPLE_NUM = [2000,2500,3000,3500,4000,5000]
    MODEL = ["gpt2", "gpt2-medium"]
    Train_set = ["train_10", "train_11", "train_12", "train_13", "train_14", "train_15", "train_16", "train_17",
                 "train_18", "train_19",
                 "train_110", "train_111", "train_112", "train_113", "train_114", "train_115", "train_116", "train_117",
                 "train_118", "train_119"]
    for model in MODEL:
        for sample_num in SAMPLE_NUM:
            if model=="gpt2-medium" and sample_num!=5000:
                continue
            results = get_trainset_results(Train_set, f"./bagging_result/{model}/{sample_num}_20")
            # print(results[0])
            gt_labels = get_gt_labels("test_results",
                                      f"./bagging_result/{model}/{sample_num}_20/train_10/weak_model_gt/eval_best_model/results.pkl")
            # print(gt_label)
            soft_acc_results = []
            hard_acc_results = []
            for i in tqdm(range(1, 20)):
                predicted_labels_hard, predicted_labels_soft, all_soft_labels = process_result(results, trainset_size=i,
                                                                                               key="test_results")
                soft_acc = calculate_acc(gt_labels, predicted_labels_soft)
                hard_acc = calculate_acc(gt_labels, predicted_labels_hard)
                soft_acc_results.append(soft_acc)
                hard_acc_results.append(hard_acc)
            print(f"{model}_soft_{sample_num} : ", soft_acc_results)
            print(f"{model}_hard_{sample_num} : ", hard_acc_results)
            plt.plot(soft_acc_results, label='soft')
            plt.plot(hard_acc_results, label='hard')
            plt.legend()
            if not os.path.exists("./bagging_result/plot"):
                os.makedirs("./bagging_result/plot")
            plt.savefig(f"./bagging_result/plot/{model}_data.png")
            # plt.show()
            max_index= soft_acc_results.index(max(soft_acc_results))
            predicted_labels_hard, predicted_labels_soft, all_soft_labels = process_result(results, trainset_size=max_index+1,
                                                                                               key="infer_valid_results")
            gt_labels = get_gt_labels("infer_valid_results",
                                      f"./bagging_result/{model}/{sample_num}_20/train_10/weak_model_gt/eval_best_model/results.pkl")
            weak_to_strong_data(f"./bagging_result/{model}/{sample_num}_20/train_10/weak_model_gt/eval_best_model/results.pkl",
                                predicted_labels_hard=predicted_labels_hard,
                                predicted_labels_soft=predicted_labels_soft,
                                all_soft_labels=all_soft_labels,
                                gt_labels=gt_labels,
                                save_path=f"./sciq/data_sample/{sample_num}/valid")
            predicted_labels_hard, predicted_labels_soft, all_soft_labels = process_result(results,
                                                                                           trainset_size=max_index + 1,
                                                                                           key="inference_results")
            gt_labels = get_gt_labels("inference_results",
                                      f"./bagging_result/{model}/{sample_num}_20/train_10/weak_model_gt/eval_best_model/results.pkl")
            weak_to_strong_data(
                f"./bagging_result/{model}/{sample_num}_20/train_10/weak_model_gt/eval_best_model/results.pkl",
                predicted_labels_hard=predicted_labels_hard,
                predicted_labels_soft=predicted_labels_soft,
                all_soft_labels=all_soft_labels,
                gt_labels=gt_labels,
                save_path=f"./sciq/data_sample/{sample_num}/train")