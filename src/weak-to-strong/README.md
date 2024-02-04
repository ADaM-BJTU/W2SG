# Improving Weak-to-Strong Generalization with Scalable Oversight and Ensemble Learning

This project contains code for implementing our [paper on Improving Weak-to-Strong Generalization with Scalable Oversight and Ensemble Learning](https://arxiv.org/pdf/2402.00667.pdf).

The primary codebase contains a re-implementation of our weak-to-strong learning setup for binary classification tasks.  The codebase contains code for training strong models using the labels from weak models.


### Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Installation

You need to have Python installed on your machine. The project uses `pyproject.toml` to manage dependencies. To install the dependencies, you can use a package manager like `pip`:

```
pip install .
```

#### Running the Script

The main script of the project is train_strong.py. It can be run from the command line using the following command:
```
python train_strong.py
```

The script accepts several command-line arguments to customize the training process. Here are some examples:

```
python train_strong.py \
    --train_name weak-to-strong/datasets/boosting/ada_train2 \
    --valid_name  weak-to-strong/datasets/boosting/ada_dev \
    --test_name  weak-to-strong/datasets/test \
    --results_folder  ./results_w2s \
    --weak_model_size  /data2/yuhang/huggingface/hub/gpt2-medium/ \
    --loss_type logconf
```

#### Expected results
The detailed training process and results will be stored in the results_folder. Additionally, the code will output "weak_acc" and "best_valid_test_acc". The former is the result of the model on the test set at the end of training, and the latter is the result of the model with the best performance on the validation set on the test set. We take the "best_valid_test_acc" as our result.


