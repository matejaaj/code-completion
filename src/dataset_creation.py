import os
import json
import random

input_folder = 'data'
output_file = os.path.join(input_folder, 'dataset.json')

def process_file(file_path, sample_id):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    cursor_position = random.randint(0, len(content) - 1)
    prefix = content[:cursor_position]
    suffix_start = content.find('\n', cursor_position)


    middle = content[cursor_position:suffix_start]
    suffix = content[suffix_start + 1:]

    data = {
        "id": sample_id,
        "prefix": prefix,
        "middle": middle,
        "suffix": suffix,
        "prediction": "",
        "file": os.path.basename(file_path)
    }

    return data

def main():
    py_files = [f for f in os.listdir(input_folder) if f.endswith('.py')]
    all_samples = []
    sample_id = 1
    
    for file_name in py_files:
        file_path = os.path.join(input_folder, file_name)
        
        for _ in range(5):
            sample = process_file(file_path, sample_id)
            all_samples.append(sample)
            sample_id += 1
    
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(all_samples, json_file, indent=4, ensure_ascii=False)
    
    print(f"All samples saved to:  '{output_file}'")

if __name__ == '__main__':
    main()
