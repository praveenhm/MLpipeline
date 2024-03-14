import logging
import time

import torch
from libdocs.classifiers.deberta.deberta import DebertaZeroShot
from libdocs.finetune.finetune import finetune
from libdocs.finetune.run import run
from libdocs.utils.banner.banner import banner
from libdocs.utils.jsonl.jsonl import JSONL
from libdocs.utils.training.training import df_to_train_df

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def finetune_model(
    hf_access_token: str = None,
    wandb_access_token: str = None,
    model_for_training_finetune: str = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
    input_hf_dataset: str = None,
    hf_model_name: str = None,
    downsample: bool = False,
    wandb_output: str = "./output",
):
    start_time = time.time()
    # we will use the second finetune model
    finetune(
        hf_access_token,
        wandb_access_token,
        model_for_training_finetune,
        input_hf_dataset,
        hf_model_name,
        downsample,
    )
    end_time = time.time()
    total_time = end_time - start_time
    banner([f"Finetuning completed in time {total_time}"])


def test(
    test_dataset,
    hf_model_name,
    hf_access_token,
    output_classify_file,
    flush_interval,
    cumulative_score,
    text_column,
    subj_column,
):
    banner(["Classifier started"])
    db_zero_shot = DebertaZeroShot(
        hf_model_name, cumulative_score, hf_access_token
    )
    start_time = time.time()
    run(
        zero_shot_model=db_zero_shot,
        test_df=test_dataset,
        output_classify_file=output_classify_file,
        # flush_interval=flush_interval,
        batch_size=500,
        text_column=text_column,
        subj_column=subj_column,
    )
    end_time = time.time()
    total_time = end_time - start_time
    banner([f"Classifier completed in time {total_time}"])
    logging.info(
        "Classification process completed. Results are saved to %s",
        output_classify_file,
    )


if __name__ == "__main__":
    import argparse

    """Model training, testing and other things useful."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="produce examples outputs.",
    )

    # Add arguments corresponding to the environment variables with direct defaults and short options
    parser.add_argument(
        "-t", "--hf-access-token", help="Hugging Face Access Token"
    )

    parser.add_argument(
        "-w", "--wandb-access-token", help="Weights & Biases Access Token"
    )

    parser.add_argument(
        "-m",
        "--model-for-training-finetune",
        default="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        help="Base Model for Training Finetune",
    )

    parser.add_argument(
        "-f", "--flush-interval", type=int, default=100, help="Flush Interval"
    )

    # Modify these to your needs with command-line arguments and short options
    parser.add_argument(
        "-i", "--input-dir", default="data/combined", help="Input Directory"
    )

    parser.add_argument(
        "-s",
        "--suffix",
        default="*deberta_combined_data_correct.jsonl",
        help="File Suffix",
    )

    parser.add_argument(
        "-a" "--dataset-name", default="zephyr", help="Dataset Name"
    )

    parser.add_argument(
        "-hf",
        "--input-hf-dataset",
        default="penma/zephyr_f2_feb14_01",
        help="Input Hugging Face Dataset",
    )

    parser.add_argument(
        "-o",
        "--hf-model-name",
        default="penma/zephyr_f2_feb14_01",
        help="Hugging Face Model Name for Training Finetune",
    )

    parser.add_argument(
        "-l",
        "--output-classify-file",
        default="data/combined/zephyr_f2.csv",
        help="Output Classify File",
    )

    parser.add_argument(
        "--subject-label",
        default="label",
        help="column name in input document of subject label",
    )
    parser.add_argument(
        "--text",
        default="text",
        help="column name in input document of text",
    )

    parser.add_argument(
        "--cumulative-score",
        default=0.8,
        help="threshold for classification cut-off cumulative score",
    )

    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Speed up the process for testing, if set true",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="train the model, if set true",
    )

    parser.add_argument(
        "--filename",
        default="mixtral_subset.jsonl",
        help="file name for the input examples file",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="file name for the input examples file",
    )

    parser.add_argument(
        "--wandb-output",
        default="./output",
        help="wandb output directory, where the hmtl files are saved",
    )

    args = parser.parse_args()
    # args = parser.parse_args(["--examples"])

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    # check cuda availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
        logging.info(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
        logging.info(torch.cuda.current_device())
        logging.info(torch.cuda.device(0))
        logging.info(torch.cuda.device_count())
        logging.info(torch.cuda.is_initialized())
    else:
        logging.info("No GPU available, exiting.")
        # device = torch.device("cpu")
        exit()

    if args.examples:
        """Test agianst given model."""
        # from testdata.test_examples import examples_list

        banner(["Test agianst given model"])
        df = JSONL().from_files(args.input_dir, args.filename)
        logging.info(df.head())
        test_df = df_to_train_df(df, args.text, args.subject_label)
        logging.info(test_df.head())

    else:
        if args.train:
            # step downloading the dataset
            from datasets import load_dataset

            dataset_finetune = load_dataset(
                args.input_hf_dataset, token=args.hf_access_token
            )
            logging.info(dataset_finetune)
            test_df = dataset_finetune["test"].to_pandas()
            logging.info(test_df.head())

            # Step 4: Finetune the model
            banner(["Finetuning started"])
            finetune_model(
                hf_access_token=args.hf_access_token,
                wandb_access_token=args.wandb_access_token,
                model_for_training_finetune=args.model_for_training_finetune,
                input_hf_dataset=args.input_hf_dataset,
                hf_model_name=args.hf_model_name,
                downsample=args.downsample,
                wandb_output=args.wandb_output,
            )

    # Step 5: Run the classification process on the finetuned model
    test(
        test_df,
        args.hf_model_name,
        args.hf_access_token,
        args.output_classify_file,
        args.flush_interval,
        args.cumulative_score,
        args.text,
        args.subject_label,
    )
