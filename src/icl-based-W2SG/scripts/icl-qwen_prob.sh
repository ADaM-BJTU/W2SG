
echo "【ICL 5-shot stochastic_random with prob】"
percent="100"
CUDA_VISIBLE_DEVICES=0 python 7b_icl_prob_percent.py --retriever_name "stochastic_random" --strong_model_size  "your_qwen-7B_path/" --ice_num 5 --weak_type_list "native,support" --Weak2Strong True --Strong True --main_func_seed 43 --icl_random_seed 43  
 


