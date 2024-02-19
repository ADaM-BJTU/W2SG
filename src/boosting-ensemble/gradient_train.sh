python "./gradient_train_weak.py"
for Epoch in 1 2 3 4
do
    for filename in "./gradient_generate_weight.py" "gradient_train_weak_weight.py"
    do
        python $filename $Epoch
    done
done