The following are running examples of the code.

## Baseline
python baseline.py --model_path [model path] --test_data_path [test data path]

## In-context-Examples-Generation
python In_context_Examples_Generation.py --model_path [model path] --data_path [data path] -- example_set_num [ m ]

## ICL-based-Knowledge-Extraction
python ICL_based_Knowledge_Extraction.py --model_path [model path] --data_path [data path] --demo_data_path ./demos.json --data_type train2 --context_example_num [ n ]

## Interation-enhanced-SO
python Interation_enhanced_SO.py --model_path [model path] --data_path [data path] --data_type train2 
