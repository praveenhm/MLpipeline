import pandas as pd

def dataframe_to_jsonl(df, output_file_path):
    """
    Convert a pandas DataFrame to a JSONL file.

    Parameters:
    - df: pandas.DataFrame to be converted.
    - output_file_path: String. The path where the JSONL file will be saved.
    """
    # Convert the DataFrame to JSONL string and save to file
    df.to_json(output_file_path, orient='records', lines=True, force_ascii=False)

    print(f"DataFrame successfully saved to {output_file_path}")

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {'text': ['Example text 1', 'Example text 2'],
            'label_text': ['Label 1', 'Label 2'],
            'source_label': ['manual', 'manual']}
    df = pd.DataFrame(data)
    
    # Specify the output JSONL file path
    output_jsonl_path = '/mnt/data/output.jsonl'
    
    # Call the function with the DataFrame and output file path
    dataframe_to_jsonl(df, output_jsonl_path)
