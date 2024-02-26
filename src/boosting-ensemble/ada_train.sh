python "./ada_train_weak.py"
for Epoch in 1 2 3 4
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        python $filename $Epoch
    done
done