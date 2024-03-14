import json

from sentence_transformers import SentenceTransformer, util


def clean_data(data, threshold=0.8):
    """Cleans and deduplicates JSON data."""
    seen_texts = set()
    unique_data = []
    for item in data:
        if item["deberta_labels"][0] not in seen_texts:
            seen_texts.add(item["deberta_labels"][0])
            unique_data.append(item)

    # Cosine similarity check
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        list(d["deberta_labels"][0] for d in unique_data),
        convert_to_tensor=True,
    )

    for i in range(len(unique_data) - 1):
        for j in range(i + 1, len(unique_data)):
            cos_sim = util.cos_sim(embeddings[i], embeddings[j])[0][0].item()
            if cos_sim > threshold:
                del unique_data[j]
                break

    return unique_data


def process_jsonl_file(input_file, output_file, threshold=0.8):
    """Reads JSONL file, cleans data, and writes to a new JSONL file."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        data = [json.loads(line) for line in infile]
        cleaned_data = clean_data(data, threshold)

        for item in cleaned_data:
            json.dump(item, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    # Configuration
    input_file = (
        "data/clean-data/batchchunker.combined.deberta.correct_data.jsonl"
    )
    output_file = "data/filtered/deberta_subset.jsonl"

    # Process the file
    process_jsonl_file(input_file, output_file)
    