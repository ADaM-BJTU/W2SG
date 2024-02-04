# prob="100"
# echo "【ICL so25 5-shot】"
# CUDA_VISIBLE_DEVICES=2 python 7b_icl_noprob_percent.py --retriever_name "so25"  --ice_num 5 --type_list "native,support" --Weak2Strong False --Strong True --weak_ds_path "dataset/sciq/1.8b_eval_naive_logits.json" --weak_support_ds_path "dataset/sciq/1.8b_eval_q_support_logits.json"  --percent $prob 

prob="100"
echo "【ICL so25 5-shot】"
CUDA_VISIBLE_DEVICES=3 python 7b_icl_noprob_percent.py --retriever_name "so25"  --ice_num 5 --type_list "native,support" --Weak2Strong True --Strong False --weak_ds_path "dataset/sciq/1.8b_eval_naive_logits.json" --weak_support_ds_path "dataset/sciq/1.8b_eval_q_support_logits.json"  --percent $prob 


#  nohup bash scripts/icl-qwen_so25.sh >> new_logs/so25/sciq-qwen7b-so25-icl10nsS-icl5nsS-icl10error-icl5error.log

prob="100"
echo "【ICL so25 5-shot with prob3】"
CUDA_VISIBLE_DEVICES=3 python 7b_icl_prob_3_percent.py --retriever_name "so25" --strong_model_size  "/data2/yuhang/huggingface/hub/qwen-7B/" --ice_num 5 --weak_type_list "native,support" --Weak2Strong True --Strong False --main_func_seed 43 --icl_random_seed 43 --weak_ds_path "dataset/sciq/1.8b_eval_naive_logits.json" --weak_support_ds_path "dataset/sciq/1.8b_eval_q_support_logits.json"  --percent $prob
#  nohup bash scripts/icl-qwen_so25copy.sh > new_logs/so25/copy_sciq-qwen7b-so25-icl10nsS-icl5nsS-icl10error-icl5error.log
