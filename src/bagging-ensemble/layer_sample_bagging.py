import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from bagging import get_gt_labels,get_trainset_results,process_result,calculate_acc,weak_to_strong_data

if __name__ == '__main__':
    MODEL = ["gpt2-medium"]
    Train_set =[i for i in range(10, 25)]
    for model in MODEL:
        results = get_trainset_results(Train_set, f"./bagging_result/{model}_layer/24")
        # print(results[0])
        gt_labels = get_gt_labels("test_results",
                                  f"./bagging_result/{model}_layer/24/weak_model_gt/eval_best_model/results.pkl")
        # print(gt_label)
        soft_acc_results = []
        hard_acc_results = []
        for i in tqdm(range(1, len(Train_set)+1)):
            predicted_labels_hard, predicted_labels_soft, all_soft_labels = process_result(results, trainset_size=i,
                                                                                           key="test_results",type="layer")
            soft_acc = calculate_acc(gt_labels, predicted_labels_soft)
            hard_acc = calculate_acc(gt_labels, predicted_labels_hard)
            soft_acc_results.append(soft_acc)
            hard_acc_results.append(hard_acc)
        print(f"{model}_soft : ", soft_acc_results)
        print(f"{model}_hard : ", hard_acc_results)
        plt.plot(soft_acc_results, label='soft')
        plt.plot(hard_acc_results, label='hard')
        plt.legend()
        if not os.path.exists("./bagging_result/plot"):
            os.makedirs("./bagging_result/plot")
        plt.savefig(f"./bagging_result/plot/{model}_layer.png")
        # plt.show()
        max_index= soft_acc_results.index(max(soft_acc_results))
        predicted_labels_hard, predicted_labels_soft, all_soft_labels = process_result(results, trainset_size=max_index+1,
                                                                                           key="eval_results")
        gt_labels = get_gt_labels("inference_results",
                                  f"./bagging_result/{model}_layer/24/weak_model_gt/eval_best_model/results.pkl")
        weak_to_strong_data(
            f"./bagging_result/{model}_layer/24/weak_model_gt/eval_best_model/results.pkl",
            predicted_labels_hard=predicted_labels_hard,
            predicted_labels_soft=predicted_labels_soft,
            all_soft_labels=all_soft_labels,
            gt_labels=gt_labels,
            save_path=f"./sciq/layer_sample/valid")
        predicted_labels_hard, predicted_labels_soft, all_soft_labels = process_result(results,
                                                                                       trainset_size=max_index + 1,
                                                                                       key="inference_results")
        gt_labels = get_gt_labels("inference_results",
                                  f"./bagging_result/{model}_layer/24/weak_model_gt/eval_best_model/results.pkl")
        weak_to_strong_data(
            f"./bagging_result/{model}_layer/24/weak_model_gt/eval_best_model/results.pkl",
            predicted_labels_hard=predicted_labels_hard,
            predicted_labels_soft=predicted_labels_soft,
            all_soft_labels=all_soft_labels,
            gt_labels=gt_labels,
            save_path=f"./sciq/layer_sample/train")