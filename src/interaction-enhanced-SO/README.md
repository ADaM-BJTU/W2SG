## baseline
python baseline.py --model_path [model path] --test_data_path [test data path]

## In-context-Examples-Generation
python In-context-Examples-Generation.py --model_path [model path] --data_path [data path] -- example_set_num m

## ICL-based-Knowledge-Extraction
python ICL-based-Knowledge-Extraction.py --model_path [model path] --data_path [data path] --demo_data_path ./demos.json --data_type train2 --context_example_num n

## Interation-enhanced-SO
python Interation-enhanced-SO.py --model_path [model path] --data_path [data path] --data_type train2 
