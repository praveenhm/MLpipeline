import matplotlib.pyplot as plt
import pandas as pd
import umap


def create_umap(
    chunks,
    chunk_embeddings,
    n_components=2,
):
    assert len(chunks) == len(
        chunk_embeddings
    ), f"chunks: {len(chunks)} embeddings: {len(chunk_embeddings)}"

    # Extract text and labels
    labels = [chunk.subject for chunk in chunks]

    # Create UMAP projection
    # reducer = umap.UMAP(random_state=1, n_jobs=1, n_components=n_components)
    reducer = umap.UMAP(
        n_components=n_components, densmap=False, random_state=42
    )
    emb_reduced = reducer.fit_transform(chunk_embeddings)
    return labels, emb_reduced


def create_and_save_3d_csv(
    chunks,
    chunk_embeddings,
    num_rows_to_process,
    filepath,
    tsv_axes_path=None,
    tsv_meta_path=None,
):
    """
    This function takes labeled chunks, creates a 3d umap projection,
    and optionally saves it as a file.

    Args:
      chunks: Chunk Text.
      chunk_embeddings: Chunk Embeddings.
      num_rows_to_process: Number of rows to process from the file.
      save_plot: Whether to save the generated plot (default: True).
      plot_file: File for saving the plot (default: "embedding_plot.png").
    """

    # Limit chunks if num_rows_to_process is specified
    if num_rows_to_process > 0:
        chunks = chunks[:num_rows_to_process]
        chunk_embeddings = chunk_embeddings[:num_rows_to_process]

    # Get UMAP
    labels, emb_reduced = create_umap(
        chunks,
        chunk_embeddings,
        n_components=3,
    )

    texts = []
    ids = []
    category = []
    x = []
    y = []
    z = []
    for index, chunk in enumerate(chunks):
        texts.append(chunk.text)
        ids.append(chunk.id)
        category.append(chunk.subject)
        x.append(emb_reduced[index][0])
        y.append(emb_reduced[index][1])
        z.append(emb_reduced[index][2])

    # Generate CSV for 3GS
    df = pd.DataFrame(columns=["x", "y", "z", "category", "text", "id"])
    df["x"] = x
    df["y"] = y
    df["z"] = z
    df["category"] = category
    df["text"] = texts
    df["id"] = ids
    df.to_csv(filepath, index=False)

    # Generate TSV files for tensorflow
    if tsv_axes_path is not None:
        df = pd.DataFrame(columns=["x", "y", "z"])
        df["x"] = x
        df["y"] = y
        df["z"] = z
        df.to_csv(tsv_axes_path, sep="\t", index=False)

    if tsv_meta_path is not None:
        df = pd.DataFrame(columns=["id", "category"])
        df["category"] = category
        df["id"] = ids
        df.to_csv(tsv_meta_path, sep="\t", index=False)


def create_and_plot_2d_umap(
    chunks_df,
    chunks,
    chunk_embeddings,
    num_rows_to_process=25000,
    save_plot=True,
    plot_file="embedding_plot.png",
):
    """
    This function takes labeled chunks, creates a 2d umap projection,
    generates a visualization and optionally saves it to a file.

    Args:
      chunks: Chunk Text.
      chunk_embeddings: Chunk Embeddings.
      num_rows_to_process: Number of rows to process from the file.
      save_plot: Whether to save the generated plot (default: True).
      plot_file: File for saving the plot (default: "embedding_plot.png").
    """

    # Limit chunks if num_rows_to_process is specified
    if num_rows_to_process > 0:
        chunks = chunks[:num_rows_to_process]
        chunk_embeddings = chunk_embeddings[:num_rows_to_process]
        if chunks_df is not None:
            chunks_df = chunks_df.iloc[:num_rows_to_process]

    assert len(chunks) == len(
        chunk_embeddings
    ), f"chunks: {len(chunks)}  embeddings: {len(chunk_embeddings)}"
    if chunks_df is not None:
        assert len(chunks) == len(
            chunks_df
        ), f"chunks: {len(chunks)}  chunks_df: {len(chunks_df)}"

    # Get UMAP
    labels, emb_reduced = create_umap(
        chunks,
        chunk_embeddings,
        n_components=2,
    )

    assert len(chunks) == len(
        emb_reduced
    ), f"chunks: {len(chunks)}  emb_reduced: {len(emb_reduced)}"

    # Viz
    plt.figure(figsize=(20, 15))

    # Group text under subjects
    unique_labels = list(set(labels))
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            emb_reduced[indices, 0],
            emb_reduced[indices, 1],
            label=label,
            s=2,
            alpha=1,
        )

    # Add to chunks_df
    all_indices = list(range(len(emb_reduced)))

    if chunks_df is not None:
        chunks_df["reduced_embedding_x"] = emb_reduced[all_indices, 0]
        chunks_df["reduced_embedding_y"] = emb_reduced[all_indices, 1]

    # Add subjects as annotations
    for i, label in enumerate(labels):
        plt.annotate(
            "",
            (emb_reduced[i, 0], emb_reduced[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.legend(loc="lower center", bbox_to_anchor=(1, 1), fontsize=20)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_file)

    if save_plot and chunks_df is not None:
        import plotly.express as px
        from plotly.subplots import make_subplots

        scatter = px.scatter(
            chunks_df,
            x="reduced_embedding_x",
            y="reduced_embedding_y",
            color="label",
            hover_data=[
                "id",
                "text",
                "label",
                # "zephyr_verdict",
                # "mistral_verdict",
                # "deberta_verdict",
                # "entity_id",
            ],
        )

        figure = make_subplots(rows=1, cols=1)
        figure.add_traces(data=scatter["data"], rows=1, cols=1)
        figure.write_html(plot_file + ".scatter.html")

    plt.show()
