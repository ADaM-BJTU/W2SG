# ICL-based Weak-to-Strong Generalization

This part of the code explains the ICL-based W2SG section.

### Getting Started 

#### Installation

You need to install Python on your machine. To install dependencies, you can use a package manager like `pip`:

```
pip install -r requirements.txt.
```
If you fail to install OpenICL, it is recommended to install it according to the method [here](https://github.com/Shark-NLP/OpenICL/tree/main).

#### Running the Script

- If you want to try the basic W2SG setting under random in Sec 5.1, you can run the following command here `scripts/icl-qwen_random.sh`:
```
bash scripts/icl-qwen_random.sh
```
- If you want to try the confidence-based W2SG setting under random in Sec 5.1, you can run the following command here `icl-qwen_prob.sh`:
```
bash scripts/icl-qwen_prob.sh
```
- If you want to try the confidence-based W2SG setting under different retrievers in Sec 5.2, you can run the following command here `icl-qwen_openicl_retriever.sh`:
```
bash scripts/icl-qwen_openicl_retriever.sh
```
#### Parameter Description 
- `weak_model_size`: String, the path of the weak model, default is "model_path_qwen-1.8B-chat".
- `strong_model_size`: String, the path of the strong model, default is "model_path_qwen-7B".
- `icl_process_batch_size`: Integer, the batch size for ICL processing, default is 4.
- `results_folder`: String, the path of the results folder, default is "./tmp/results_icl_new".
- `retriever_name`: String, the name of the retriever, default is 'stochastic_random', you can check the details in `weak_to_strong/icl_retriever.py`.
- `ice_num`: Integer, the number of context examples, default is 5.
- `Weak2Strong`: Boolean, whether to perform weak to strong ICL, default is True.
- `Strong`: Boolean, whether to perform strong ICL, default is True.
- `weak_ds_path`: String, the path of the weak dataset before SO, default is 'dataset/sciq/1.8b_eval_naive_logits.json'.
- `weak_support_ds_path`: String, the path of the weak dataset after SO, default is 'dataset/sciq/1.8b_eval_q_support_logits.json'.
- `test_ds_path`: String, the path of the test dataset, default is 'dataset/sciq/test'.
- `train2_ds_path`: String, the path of the real supervised training dataset, default is 'dataset/sciq/train2'.
- `weak_type_list`: String, the list of weak types, default is "native,support".
- `error_num`: Integer, the default number of error labels under real supervision, default is 0.
- `icl_random_seed`: Integer, the random seed for ICL random example selection, default is 43.
- `main_func_seed`: Integer, the seed of the main function, default is 43.
- `test_num`: Integer, the number of tests, default is 2000.
- `max_every_class`: Dictionary, the maximum number of each class in the example, default is None.
- `percent`: String, percentage, default is "100".

#### Expected results

The results will be output in the `tmp/results_icl_new` directory