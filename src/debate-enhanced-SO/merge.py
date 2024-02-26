import json

with open('correct_explanation_path', 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)
with open('correct_explanation_path', 'r', encoding='utf-8') as file2:
    data2 = json.load(file2)


merged_data = []

for idx, line in enumerate(data1):
    merged_item = {
        "question": line["question"],
        "answer": line["answer"],
        "correct_explanation": line["explanation"],
        "incorrect_explanation": data2[idx]["explanation"],
        "hard_label": line['hard_label']
    }
    merged_data.append(merged_item)

with open('output_json_path', 'w', encoding='utf-8') as outfile:
    json.dump(merged_data, outfile, ensure_ascii=False, indent=2)
