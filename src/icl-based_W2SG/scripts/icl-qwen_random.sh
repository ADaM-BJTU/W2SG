
prob="100"
echo "【ICL 5-shot stochastic_random】 " 
CUDA_VISIBLE_DEVICES=2 python 7b_icl_noprob_percent.py --retriever_name "stochastic_random" --strong_model_size  "/data2/zhangj/model/qwen-7B/" --ice_num 5 --weak_type_list "native,support" --Weak2Strong True --Strong True  --weak_ds_path "dataset/sciq/1.8b_eval_naive_logits.json" --weak_support_ds_path "dataset/sciq/1.8b_eval_q_support_logits.json"    --percent $prob    

echo "【ICL 5 shot gt error】"
for error_num in 1 2 3 4 5
do
 CUDA_VISIBLE_DEVICES=2 python 7b_icl_noprob_percent.py --retriever_name "random" --strong_model_size   "/data2/zhangj/model/qwen-7B/" --ice_num 5 --weak_type_list "" --Weak2Strong True --Strong True  --error_num $error_num
done

# nohup bash scripts/icl-qwen_random.sh > new_logs/random/sciq-qwen7b-random-icl5nsS-icl10error-icl5error.log