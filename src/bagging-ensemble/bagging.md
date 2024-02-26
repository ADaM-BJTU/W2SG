##Bagging-enhanced Weak-to-Strong Generalization

###Step 1: Dataset segmentation

####1) Training set sampling

Run `data_sample.py` to process the dataset.

```python
def sample_list(input_list, num_samples, random_seed, sampling_method):
    if sampling_method == "random":
        random.seed(random_seed) # random sampling
    elif sampling_method == "bootstrap":
        random.choices(random_seed) # bootstrap sampling
    return random.sample(input_list, num_samples)
```
In the above code snippet, `num_samples` denotes for the number of sampled items (each item represents a pair of samples, one with correct label and another with incorrect label). 

`Sampling method` represents the chosen method between `random sampling` and `bootstrap sampling`.

####2) Feature layer sampling

###Step 2: Model training

The main script of this step is `train_shell.py`.

`train_shell.py` takes in almost all of the arguments that `train_weak.py` takes (e.g. `--weak_model_size`, `--train1_name`, `--train2_name` etc., see `train_weak.py` for a full list). These arguments are simply forwarded to `train_weak.py`.

If needed, you can also run `train_weak.py` directly.

###Step 3: Model integration