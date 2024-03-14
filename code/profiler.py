from libdocs.utils.jsonl.jsonl import JSONL

j = JSONL()
j.from_files(
    "data/20240307",
    "train_data_norm2.jsonl",
)
j.profile(
    "data/20240307/norm2.html",
    "text",
    "label",
)
