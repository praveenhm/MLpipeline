import pandas as pd
import json

def parquet_to_jsonl(parquet_file_path, jsonl_file_path):
    # Step 1: Read the Parquet file
    df = pd.read_parquet(parquet_file_path)
    
    # Step 2: Convert DataFrame to JSON Lines format
    # We iterate through each row, convert it to a dictionary, then to a JSON string
    jsonl_str = (df.apply(lambda x: json.dumps(x.to_dict()), axis=1)).tolist()
    
    # Step 3: Write JSON Lines to a file
    with open(jsonl_file_path, 'w') as outfile:
        for entry in jsonl_str:
            outfile.write(entry + '\n')

# Example usage
parquet_file_path = 'path/to/your/input.parquet'
jsonl_file_path = 'path/to/your/output.jsonl'
parquet_to_jsonl(parquet_file_path, jsonl_file_path)

print(f"Converted {parquet_file_path} to {jsonl_file_path}")
