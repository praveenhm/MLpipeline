import json
import logging
import os
import time
from typing import Any, Tuple

import numpy as np
import rich
from libdocs.classifiers.knn.knn_classifier import KnnEmbeddingClassifier
from libdocs.embedder.embedder import Embedder
from libdocs.embedder.visualize import (create_and_plot_2d_umap,
                                        create_and_save_3d_csv)
from libdocs.faissindexer.faissindexer import FaissIndexer
from libdocs.types.types import LabeledChunk
from libdocs.utils.banner.banner import banner
from libdocs.utils.jsonl.jsonl import JSONL
from libdocs.utils.training.training import load_data
from tqdm import tqdm


def create_embeddings(model, index, labeled_chunks):

    banner([f"Create embeddings: {index}"])

    # Initialize the models
    embedder = Embedder(model)
    faiss_index = FaissIndexer(embedder.get_dimensions())

    # Encoding and indexing train chunks
    labeled_chunk_ids = [0] * len(labeled_chunks)
    for id, chunk in enumerate(labeled_chunks):
        labeled_chunk_ids[id] = chunk.id
    embeddings = embedder.create_embedding(labeled_chunks)

    # Assert validate basic sanity
    assert len(labeled_chunks) == len(
        embeddings
    ), f"chunks {len(labeled_chunks)} embeddings {len(embeddings)}"
    assert len(labeled_chunks) == len(
        labeled_chunk_ids
    ), f"chunks {len(labeled_chunks)} ids {len(labeled_chunk_ids)}"

    # Add to faiss
    faiss_index.add(embeddings, labeled_chunk_ids)
    faiss_index.save_index(index)

    # Save the embeddings
    embedder.save_embeddings(index)
    return embeddings


# Filtering function
def dense_filter(
    embedding_dimensions,
    subjects: list[str],
    chunk_id_map,
    chunk_id_to_flat_id_map,
    chunk_embeddings,
    max_per_subject,
    d=0.1,
) -> Tuple[list[LabeledChunk], list[Any]]:

    balanced_chunks: list[LabeledChunk] = []
    balanced_embeddings = []

    # A FAISS per subject
    faiss_balancer = {}
    for subject in subjects:
        faiss_balancer[subject] = FaissIndexer(dimension=embedding_dimensions)
        # TODO Insert Centroid

    # Look at all chunks
    for chunk_id, chunk in chunk_id_map.items():

        # Get the subject faiss instance
        faiss_instance = faiss_balancer[chunk.subject]

        # Get matches for this chunk.
        flat_id = chunk_id_to_flat_id_map[chunk_id]
        chunk_embedding = chunk_embeddings[flat_id]
        # Add the current chunk
        faiss_instance.bare_index.train(np.array([chunk_embedding]))

    for subject in subjects:
        faiss_instance = faiss_balancer[subject]
        c = faiss_instance.bare_index.reconstruct_n(
            0, faiss_instance.bare_index.nlist
        )
        rich.print(c)

    return balanced_chunks, balanced_embeddings


def sparse_filter(
    embedding_dimensions,
    subjects: list[str],
    chunk_id_map,
    chunk_id_to_flat_id_map,
    chunk_embeddings,
    max_per_subject,
    d=0.1,
) -> Tuple[list[LabeledChunk], list[Any]]:

    balanced_chunks: list[LabeledChunk] = []
    balanced_embeddings = []

    # A FAISS per subject
    faiss_balancer = {}
    for subject in subjects:
        faiss_balancer[subject] = FaissIndexer(dimension=embedding_dimensions)

    # Look at all chunks
    for chunk_id, chunk in tqdm(chunk_id_map.items()):

        # Get the subject faiss insta   nce
        faiss_instance = faiss_balancer[chunk.subject]

        # If the faiss is already overpopulated skip more entries. However we need to do UMAP analysis
        # and make sure this is indeed a sparse rep of the original subject chunk.
        if faiss_instance.index.ntotal > max_per_subject:
            continue

        # Check if this chunk is too close to a pre-existing chunk. If it is, lets skip it.
        flat_id = chunk_id_to_flat_id_map[chunk_id]
        chunk_embedding = chunk_embeddings[flat_id]
        matches, indices = faiss_instance.index.search(
            np.array([chunk_embedding]), 5
        )
        assert len(matches[0]) == len(
            indices[0]
        ), f"mismatch len matches: {len(matches[0])} indices: {len(indices[0])}"

        is_eligible_entry = True
        # across all matches find the min and the max distances
        min_distance = 1000
        max_distance = 0
        for i, index in enumerate(indices[0]):
            # index is negative for a non-match
            if index < 0:
                continue
            match = matches[0][i]
            if match < min_distance:
                min_distance = match
            if match > max_distance:
                max_distance = match

            # We are doing sparse so if we are too close to a matche, ignore the current entry
            if min_distance < d:
                is_eligible_entry = False

        if is_eligible_entry:
            # Add the current chunk
            faiss_instance.add(np.array([chunk_embedding]), [chunk_id])
            balanced_chunks.append(chunk)
            balanced_embeddings.append(chunk_embedding)

    return balanced_chunks, balanced_embeddings


def filter(
    model,
    index,
    input_dir,
    input_file,
    text_label,
    subject_label,
    output_dir,
    distances,
    sparse=False,
):
    banner(["Filtering to an balanced data"])

    start_time = time.time()

    # This is the 80% random and shuffled data with a cluster per subject.
    # The objective is to take every cluster and not have more than max_frequency_count entries per cluster.
    knn_classifier = KnnEmbeddingClassifier(index)
    chunk_embeddings = knn_classifier.embedder.embeddings
    end_time = time.time()
    total_time = end_time - start_time
    rich.print(f"Loaded classifier in time {total_time}")

    # Invalid subjects populated for those where the frequency is very low.
    invalid_subjects = ["business_ethics", "business_development"]  # TODO

    # Freq Table, Reduced Chunks
    subjects = {}
    reduced_chunk_id_map = {}
    reduced_chunk_id_to_flat_id_map = {}
    chunk_texts = []
    chunk_embeddings = []

    for id, chunk in enumerate(knn_classifier.chunk_id_map.values()):

        if chunk.subject in invalid_subjects:
            continue
        if chunk.subject not in subjects:
            subjects[chunk.subject] = 0
        subjects[chunk.subject] += 1
        embedding = knn_classifier.embedder.embeddings[id]
        reduced_chunk_id_map[id] = chunk
        reduced_chunk_id_to_flat_id_map[id] = len(chunk_texts)
        chunk_texts.append(chunk.text)
        chunk_embeddings.append(embedding)

    # Create embeddings for all chunks (maybe this exists somewhere)
    embedder = Embedder(model)
    assert len(chunk_embeddings) == len(
        chunk_texts
    ), f"{len(chunk_embeddings)} != {len(chunk_texts)}"

    # Get Frequency counts so we can compare visually
    min_frequency_count = 10000000000
    max_frequency_count = 0
    for subject, count in subjects.items():
        if count < min_frequency_count:
            min_frequency_count = count
        if count > max_frequency_count:
            max_frequency_count = count

    banner(
        [
            f"subjects: {subjects}",
            f"     min_frequency_count: {min_frequency_count}",
            f"     max_frequency_count: {max_frequency_count}",
        ]
    )

    max_frequency_count = 2 * min_frequency_count
    banner(
        [
            "new min_frequency_count and max_frequency_count",
            f"     min_frequency_count: {min_frequency_count}",
            f"     max_frequency_count: {max_frequency_count}",
        ]
    )

    banner(["Balancing"])
    for d in distances:
        # Now lets create a subset of data such that no cluster has more than max_frequency_allowed
        balanced_chunks: list[LabeledChunk] = []
        if sparse:
            balanced_chunks, balanced_embeddings = sparse_filter(
                embedding_dimensions=embedder.get_dimensions(),
                subjects=subjects,
                chunk_id_map=reduced_chunk_id_map,
                chunk_id_to_flat_id_map=reduced_chunk_id_to_flat_id_map,
                chunk_embeddings=chunk_embeddings,
                max_per_subject=max_frequency_count,
                d=d,
            )
        else:
            balanced_chunks, balanced_embeddings = dense_filter(
                embedding_dimensions=embedder.get_dimensions(),
                subjects=subjects,
                chunk_id_map=reduced_chunk_id_map,
                chunk_id_to_flat_id_map=reduced_chunk_id_to_flat_id_map,
                chunk_embeddings=chunk_embeddings,
                max_per_subject=max_frequency_count,
                d=d,
            )

        output_filename = input_file.replace(".jsonl", "")
        if sparse:
            output_filename = output_filename + f"_sparse_{d}"
        else:
            output_filename = output_filename + f"_dense_{d}"

        rich.print(f"Creating Sparse Dataset: {output_filename}.jsonl")
        import pandas as pd

        df = pd.DataFrame(columns=["text", "label"])
        for chunk in balanced_chunks:
            df.loc[len(df.index)] = [chunk.text, chunk.subject]
        j = JSONL(df)
        j.to_file(output_dir=input_dir, output_filename=output_filename)

        rich.print("Creating UMAP: {output_filename}.png")
        # Create umap
        create_umap(
            None,
            balanced_chunks,
            balanced_embeddings,
            os.path.join(input_dir, output_filename + ".png"),
        )


def test_examples(
    index, input_dir, input_file, text_label, subject_label, output_dir
):
    banner(["Testing examples"])
    start_time = time.time()
    knn_classifier = KnnEmbeddingClassifier(index)
    end_time = time.time()
    total_time = end_time - start_time
    rich.print(f"Loaded classifier in time {total_time}")

    # Read examples from file
    example_df = JSONL().from_files(input_dir, input_file)
    texts = list(example_df[text_label])
    input_subjects = list(example_df[subject_label])
    labled_chunks = []
    for _, row in example_df.iterrows():
        labled_chunks.append(
            LabeledChunk(text=row[text_label], subject=row[subject_label])
        )

    accuracies = []
    for topk in range(1, 4):
        # Get the predictions, write to a file for analysis
        predictions = knn_classifier.predict(texts, topk)
        y_pred_list = [p.y_pred for p in predictions[0]]
        y_proba_list = [p.y_proba for p in predictions[0]]

        output_file = f"predictions_k{topk}.jsonl"
        with open(os.path.join(output_dir, output_file), "w") as f:
            for text, input_subject, y_pred, y_proba in zip(
                texts, input_subjects, y_pred_list, y_proba_list
            ):
                data = {
                    "text": text,
                    "subject": input_subject,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                }
                f.write(json.dumps(data) + "\n")

        # Get the summary evaluation and matrix
        start_time = time.time()
        accuracy = knn_classifier.summary_evaluation(labled_chunks, topk)
        end_time = time.time()
        total_time = end_time - start_time
        accuracies.append([accuracy, total_time])

    accuracy_prints = ["Accuracies:"]
    for idx, a in enumerate(accuracies):
        accuracy_prints.append(
            f"Top {idx+1}: {a[0]*100:.4f}% in {a[1]:.3f} seconds"
        )
    banner(accuracy_prints, skip_intermediate=True)


def predict_single_text(index, text):
    """Predicts the subject for a single text input."""
    knn_classifier = KnnEmbeddingClassifier(index)

    for index in range(1, 4):
        start_time = time.time()
        predictions = knn_classifier.predict([text], index)
        end_time = time.time()
        total_time = end_time - start_time
        output = {
            "text": text,
            "predictions": predictions,
            "time_takem": total_time,
        }
        rich.print(output)


def test_and_print_accuracies(index, testing):
    # Display results with test chunks
    banner([f"Testing with {len(testing)} chunks"])
    start_time = time.time()
    knn_classifier = KnnEmbeddingClassifier(index)
    end_time = time.time()
    total_time = end_time - start_time
    rich.print(f"Loaded classifier in time {total_time}")

    accuracies = []
    for index in range(1, 4):
        start_time = time.time()
        accuracy = knn_classifier.summary_evaluation(testing, index)
        end_time = time.time()
        total_time = end_time - start_time
        accuracies.append([accuracy, total_time])

    accuracy_prints = ["Accuracies:"]
    for idx, a in enumerate(accuracies):
        accuracy_prints.append(
            f"Top {idx+1}: {a[0]*100:.4f}% in {a[1]:.3f} seconds"
        )
    banner(accuracy_prints, skip_intermediate=True)


def create_umap(
    load_df, chunks, embeddings, filepath, num_rows_to_process: int = 25000
):

    csvpath = filepath.replace(".png", ".csv")
    tsv_axes_path = filepath.replace(".png", ".axes.tsv")
    tsv_meta_path = filepath.replace(".png", ".meta.tsv")

    banner(
        [
            "Plot UMAP:",
            f"    2D:       umap - {filepath}",
            f"    3D:        3gs - {csvpath}",
            f"    3D: tensorflow - {tsv_axes_path} and {tsv_meta_path}",
        ],
        skip_intermediate=True,
    )

    create_and_plot_2d_umap(
        load_df, chunks, embeddings, num_rows_to_process, True, filepath
    )

    create_and_save_3d_csv(
        chunks, embeddings, -1, csvpath, tsv_axes_path, tsv_meta_path
    )


if __name__ == "__main__":
    import argparse

    """KNN training, testing and other things useful."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--input-dir",
        default="data/",
        help="input directory",
    )
    parser.add_argument(
        "-f",
        "--input-file",
        default="zephyr_combined.jsonl",
        help="input jsonl file. wildcards also work.",
    )
    parser.add_argument(
        "-i",
        "--index-file",
        default="index.faiss",
        help="faiss index file",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="distiluse-base-multilingual-cased-v2",
        help="embedding model to use",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="skip test.",
    )
    parser.add_argument(
        "-o",
        "--skip-umap-generation",
        action="store_true",
        help="skip umap generation.",
    )
    parser.add_argument(
        "-s",
        "--subject-label",
        default="label",
        help="column name in input document of subject label",
    )
    parser.add_argument(
        "-t",
        "--text-label",
        default="text",
        help="column name in input document of text",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="filter and create a subset.",
    )
    parser.add_argument(
        "--filter-sparse",
        action="store_true",
        help="filter and create sparse subset.",
    )
    parser.add_argument(
        "--filter-dense",
        action="store_true",
        help="filter and create dense subset.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="produce examples outputs.",
    )
    parser.add_argument(
        "--conversation",
        action="store_true",
        help="Enable prediction for a single text input.",
    )
    parser.add_argument(
        "-c",
        "--conversation_text",
        default="What is a good sales strategy?",
        help="Text for single conversation prediction.",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        default="data/",
        help="output directory",
    )
    args = parser.parse_args()

    if args.index_file == "index.faiss":
        args.index_file = os.path.join(args.input_dir, args.index_file)

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.filter or args.filter_dense or args.filter_sparse:

        start_time = time.time()
        sparse_distances = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.3,
            1.9,
            2.5,
            3.0,
            4.0,
            5.0,
        ]
        dense_distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        run_types = []
        if args.filter or args.filter_sparse:
            run_types.append(True)
        if args.filter or args.filter_dense:
            run_types.append(False)

        for sparse in run_types:
            if sparse:
                distances = sparse_distances
            else:
                distances = dense_distances
            filter(
                args.model,
                args.index_file,
                args.input_dir,
                args.input_file,
                args.text_label,
                args.subject_label,
                args.output_dir,
                distances,
                sparse,
            )
            end_time = time.time()
            total = end_time - start_time
            rich.print(f"Total Time: {total}")
    # Example
    elif args.examples:
        test_examples(
            args.index_file,
            args.input_dir,
            args.input_file,
            args.text_label,
            args.subject_label,
            args.output_dir,
        )
    elif args.conversation:
        predict_single_text(args.index_file, args.conversation_text)
    else:
        # Load data
        train_chunks, test_chunks, train_df, test_df = load_data(
            args.input_dir, args.input_file, args.text_label, args.subject_label
        )
        banner([f"Training with {len(train_chunks)} chunks"])
        # Create embeddings.
        # NOTE: As load_data randomizes and shuffles we MUST regenerate embeddings from train set.
        train_embeddings = create_embeddings(
            args.model, args.index_file, train_chunks
        )

        # Skip testing
        if not args.skip_test:
            # Test with the test_chunks and print accuracies
            test_and_print_accuracies(args.index_file, test_chunks)

        # Create umap
        if not args.skip_umap_generation:
            filepath = "combined-embedding-plot.png"
            if args.input_file.find("*") < 0:
                filepath = args.input_file + ".png"
            create_umap(
                train_df,
                train_chunks,
                train_embeddings,
                os.path.join(args.input_dir, filepath),
            )

    banner(["Done !"])
