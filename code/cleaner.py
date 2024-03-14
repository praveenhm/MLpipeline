import os

from libdocs.llmchecker.llmchecker import LLMChecker

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="zephyr",
        help="model to validate against. options are deberta|mistral|zephyr",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/combined/run3/",
        help="input directory",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="pdfs_chunks_batchchunker.combined.jsonl",
        help="input file",
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
        "--dry-run",
        action="store_true",
        help="dry run",
    )
    args = parser.parse_args()

    # intermediate directory
    intermediate_dir = os.path.join(args.input_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    checker = LLMChecker()
    checker.add_model_verdict(
        args.model,
        args.input_dir,
        args.input_file,
        args.text_label,
        args.subject_label,
        intermediate_dir,
        args.dry_run,
    )
