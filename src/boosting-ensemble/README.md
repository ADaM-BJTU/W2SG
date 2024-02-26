The following are running examples of the code.

# Runnig the Script

## Adaboost

### train
```bash
bash ada_train.sh
```

### predict
```bash
bash ada_predict.sh
```

## Gradient Boost

### train
```bash
bash gradient_train.sh
```

### predict
```bash
bash gradient_predict.sh
```

# File Description
- `ada_train_weak.py/gradient_train_weak.py`: Train the first model
- `ada_generate_weight.py/gradient_generate_weight.py`: Generate weighted samples with adaboost/gradient boost method.
- `ada_train_weak.py/gradient_train_weak.py`: Train single weak model with weighted samples
- `ada_predict.py/gradient_predict.py`: Test the results of boosting ensemble methods   
