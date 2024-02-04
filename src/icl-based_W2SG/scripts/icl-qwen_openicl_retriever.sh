echo "【ICL 5-shot openicl_bm25】"
prob="100" 
CUDA_VISIBLE_DEVICES=2 python 7b_icl_noprob_percent.py --retriever_name "openicl_bm25"  --ice_num 5 --type_list "native,support" --Weak2Strong True --Strong True --weak_ds_path "dataset/sciq/1.8b_eval_naive_logits.json" --weak_support_ds_path "dataset/sciq/1.8b_eval_q_support_logits.json"  --percent $prob    
echo "【ICL 5-shot topk】"
prob="100" 
CUDA_VISIBLE_DEVICES=2 python 7b_icl_noprob_percent.py --retriever_name "topk"  --ice_num 5 --type_list "native,support" --Weak2Strong True --Strong True --weak_ds_path "dataset/sciq/1.8b_eval_naive_logits.json" --weak_support_ds_path "dataset/sciq/1.8b_eval_q_support_logits.json"  --percent $prob   
echo "【ICL 5-shot vote】"
prob="100" 
CUDA_VISIBLE_DEVICES=2 python 7b_icl_noprob_percent.py --retriever_name "vote"  --ice_num 5 --type_list "native,support" --Weak2Strong True --Strong True --weak_ds_path "dataset/sciq/1.8b_eval_naive_logits.json" --weak_support_ds_path "dataset/sciq/1.8b_eval_q_support_logits.json"  --percent $prob   
 