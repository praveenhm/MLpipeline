import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import tempfile
import os
import pandas as pd

def load_jsonl(input_file):
    """Load data from a JSONL file."""
    data = []
    with open(input_file, "r") as file:
        for line in tqdm(file, desc="Loading JSONL"):
            data.append(json.loads(line.strip()))
    return data

def deduplicate(data,subject="text"):
    """Remove exact duplicates based on the `text` field."""
    unique_data = list({d[subject]: d for d in tqdm(data, desc="Deduplicating")}.values())
    return unique_data

def remove_similar_sentences(data, subject, threshold=0.2):
    """Remove sentences that are very similar based on cosine similarity."""
    texts = [d[subject] for d in data]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    to_remove = set()
    for i in tqdm(range(len(cosine_sim)), desc="Calculating Similarity"):
        for j in range(i + 1, len(cosine_sim)):
            if cosine_sim[i, j] > threshold:
                to_remove.add(j)

    reduced_data = [data[i] for i in tqdm(range(len(data)), desc="Removing Similar Sentences") if i not in to_remove]
    return reduced_data

def save_jsonl(data, output_file):
    """Save data back to a JSONL file."""
    with open(output_file, "w") as outfile:
        for entry in tqdm(data, desc="Saving JSONL"):
            json.dump(entry, outfile)
            outfile.write("\n")


def read_jsonl_in_chunks(input_file, chunk_size=10000):
    """Generator to lazily load JSONL data in chunks."""
    chunk = []
    with open(input_file, "r") as file:
        for line in tqdm(file, desc="Reading in Batches"):
            chunk.append(json.loads(line.strip()))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:  
            yield chunk

def process_chunk_and_save(chunk, intermediate_file,subject="text"):
    """Process a single chunk of data and append to an intermediate file."""
    # Process chunk to remove similar sentences
    processed_chunk = remove_similar_sentences(chunk,subject)
    
    # Append processed chunk to the intermediate file
    with open(intermediate_file, "a") as outfile:
        for entry in tqdm(processed_chunk, desc="Appending to Intermediate File"):
            json.dump(entry, outfile)
            outfile.write("\n")

def final_deduplication_and_save(intermediate_file, output_file):
    """Perform final deduplication on the intermediate file and save to the output file."""
    data = load_jsonl(intermediate_file)  # Load the intermediate data
    deduplicated_data = deduplicate(data)  # Deduplicate
    save_jsonl(deduplicated_data, output_file)  # Save the final deduplicated data

def jsonl_to_dataframe(output_file):
    """Convert a JSONL file to a pandas DataFrame."""
    return pd.read_json(output_file, lines=True)

def clean_large_data(input_file, output_file, subject="text", chunk_size=10000):
    """Process a large JSONL file in chunks and save the cleaned data. Optionally return a DataFrame of the cleaned data."""
    # Create a temporary file to store intermediate results
    intermediate_file = tempfile.mkstemp()[1]
    
    # Check if the file is smaller than the batch size
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    if total_lines <= chunk_size:
        # If the file is smaller than the batch size, process it directly without chunking
        data = load_jsonl(input_file)
        processed_data = remove_similar_sentences(data,subject)
        deduplicated_data = deduplicate(processed_data,subject)
        save_jsonl(deduplicated_data, output_file)
    else:
        # Process each chunk if the file is larger than the batch size
        for chunk in read_jsonl_in_chunks(input_file, chunk_size):
            process_chunk_and_save(chunk, intermediate_file,subject)
        # Perform final deduplication on the collected data and save
        final_deduplication_and_save(intermediate_file, output_file)
    
    # Clean up the intermediate file
    os.remove(intermediate_file)
       
    
    return jsonl_to_dataframe(output_file)

# Example usage
if __name__ == "__main__":
    input_file = "data/20240308/train_data_norm_test.jsonl"
    output_file = "data/filtered/train_data_filt3.jsonl"
    subject = "text"
    dx = clean_large_data(input_file, output_file, subject)
    print(dx.head())
    print(dx.info())
    print(len(dx))