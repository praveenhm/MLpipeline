import json

# File paths
file1_path = "/Users/praveenm/dev/tanh/data/20240228/inference/clean_combined_test.jsonl"
file2_path = "/Users/praveenm/dev/tanh/data/20240228/train/normalize/clean_balanced_combined_train.jsonl"
output_file2_path = "/Users/praveenm/dev/tanh/data/20240228/train/normalize/unique_entries.jsonl"

def read_texts_from_jsonl(file_path):
    texts = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if 'text' in data:
                texts.add(data['text'])
    return texts

def save_difference_and_count(file1_texts, file2_path, output_path):
    difference_count = 0
    with open(file2_path, 'r', encoding='utf-8') as file, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in file:
            data = json.loads(line)
            if 'text' in data and data['text'] not in file1_texts:
                json.dump(data, outfile)
                outfile.write('\n')
                difference_count += 1
    return difference_count

# Extract the 'text' field values from file1
texts_file1 = read_texts_from_jsonl(file1_path)

# Save the difference (unique 'text' entries from file2) to the output file and get the count
difference_entries_count = save_difference_and_count(texts_file1, file2_path, output_file2_path)

print(f"Entries unique to file2 have been saved to: {output_file2_path}")
print(f"Total number of unique entries in file2 compared to file1: {difference_entries_count}")
