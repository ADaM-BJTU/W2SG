import os
import random
random.seed(42)
MODEL = ["gpt2-medium"]
RATE = [0.1, 0.3, 0.5, 0.7]
RANDOM_SEED = random.sample(range(43, 1000), 20)
for model in MODEL:
    for rate in RATE:
        for index, random_seed in enumerate(RANDOM_SEED):
            commend = f'''CUDA_VISIBLE_DEVICES=1 python train_weak.py \
                                    --weak_model_size /data2/yuhang/huggingface/hub/{model}/ \
                                    --train1_name  ./sciq/train1/ \
                                    --train2_name  ./sciq/train2/ \
                                    --eval_name  ./sciq/validation/ \
                                    --test_name  ./sciq/test/ \
                                    --results_folder  ./bagging_result/{model}_feature/{rate}/{index} \
                                    --sample_num {int(rate * 1024)} \
                                    --random_seed {random_seed} \
                                    --gt_epochs 4 \
                                    --eval_every 300 '''
            os.system(commend)