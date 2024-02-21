import os
MODEL=["gpt2-medium"]
for model in MODEL:
    for layer in range(10,26):
        commend = f'''CUDA_VISIBLE_DEVICES=1 python train_weak.py \
                                --weak_model_size /path_to_the_models/ \
                                --train1_name  ./sciq/train1 \
                                --train2_name  ./sciq/train2/ \
                                --eval_name  ./sciq/validation/ \
                                --test_name  ./sciq/test/ \
                                --results_folder  bagging_result/{model}_layer/{layer} \
                                --hidden_layer {layer} \
                                --gt_epochs 4 \
                                --eval_every 300 '''
        os.system(commend)