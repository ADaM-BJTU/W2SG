 
# icl-based Weak-to-strong generalization

 
这一部分的代码说明了icl-based w2sg部分。

### Getting Started 

#### Installation

您需要在您的机器上安装 Python。要安装依赖项，您可以使用类似`pip`的包管理器:

```
pip install -r requirements.txt.
```
如果您安装OpenICL失败，建议您按照[这里](https://github.com/Shark-NLP/OpenICL/tree/main)的方法安装。

#### Running the Script

- 如果您想尝试Sec 5.1中随机下的basic w2sg设置，在这里`scripts/icl-qwen_random.sh`.可以通过运行以下命令：
```
bash scripts/icl-qwen_random.sh
```
- 如果您想尝试Sec 5.1中随机下的基于confidence的w2sg设置，在这里`icl-qwen_prob.sh`，可以通过运行以下命令：
```
bash scripts/icl-qwen_prob.sh
```
- 如果您想尝试Sec 5.2中不同retriever下的基于confidence的w2sg设置，在这里`icl-qwen_openicl_retriever.sh`，可以通过运行以下命令：
```
bash scripts/icl-qwen_openicl_retriever.sh
```
#### 参数说明 
- `weak_model_size`: 字符串，弱模型的路径，默认为 "model_path_qwen-1.8B-chat"。
- `strong_model_size`: 字符串，强模型的路径，默认为 "model_path_qwen-7B"。
- `icl_process_batch_size`: 整数，ICL 处理的批处理大小，默认为 4。
- `results_folder`: 字符串，结果文件夹的路径，默认为 "./tmp/results_icl_new"。
- `retriever_name`: 字符串，检索器的名称，默认为 'stochastic_random'，在`weak_to_strong/icl_retriever.py`可以查看细节。
- `ice_num`: 整数，上下文示例的数量，默认为 5。
- `Weak2Strong`: 布尔值，是否执行弱到强icl，默认为 True。
- `Strong`: 布尔值，是否执行强icl，默认为 True。
- `weak_ds_path`: 字符串，SO前弱数据集的路径，默认为 'dataset/sciq/1.8b_eval_naive_logits.json'。
- `weak_support_ds_path`: 字符串，SO后弱数据集的路径，默认为 'dataset/sciq/1.8b_eval_q_support_logits.json'。
- `test_ds_path`: 字符串，测试数据集的路径，默认为 'dataset/sciq/test'。
- `train2_ds_path`: 字符串，真实监督训练数据集的路径，默认为 'dataset/sciq/train2'。
- `weak_type_list`: 字符串，弱类型列表，默认为 "native,support"。
- `error_num`: 整数，真实监督下默认的错误标签数量，默认为 0。
- `icl_random_seed`: 整数，ICL random挑选示例的随机种子，默认为 43。
- `main_func_seed`: 整数，主函数的种子，默认为 43。
- `test_num`: 整数，测试数量，默认为 2000。
- `max_every_class`: 字典，示例中每个类的最大数量，默认为 None。
- `percent`: 字符串，百分比，默认为 "100"。

#### Expected results

结果将输出在`tmp/results_icl_new`目录下


