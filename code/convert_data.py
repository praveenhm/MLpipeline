import argparse
import hashlib
import logging
import os

from libdocs.utils.jsonl.jsonl import JSONL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        dest="inputFile",
        type=str,
        default="",
        help="Name of the file to run through the data converter",
    )
    parser.add_argument(
        "--output-dir",
        dest="outputDir",
        type=str,
        default=os.getcwd(),
        help="Path to the output directory where the file is going to get created. The naming is determined based on the data prefix and the chunker name.",
    )
    parser.add_argument(
        "--data-prefix",
        dest="prefix",
        type=str,
        default="default",
        help="The prefix name for the input data set. This determines the produced output file name.",
    )
    parser.add_argument(
        "--chunker-name",
        dest="chunkerName",
        type=str,
        default="none",
        help="The name of the chunker which was applied to the input data set. This defaults to 'none'. This determines the produced output file name.",
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.logLevel))

    if not os.path.isfile(args.inputFile):
        raise Exception(f"{args.inputFile} does not exist or is not a file")
    if not os.path.isdir(args.outputDir):
        raise Exception(
            f"{args.outputDir} does not exist or is not a directory"
        )

    # read input file
    dir = os.getcwd()
    if args.inputFile.startswith("/"):
        dir = os.path.dirname(args.inputFile)
    file = os.path.basename(args.inputFile)
    jl = JSONL()
    jl.from_files(dir, file)

    # rename columns
    jl.df.rename(
        inplace=True,
        columns={
            "body": "text",
            "chunk": "text",
            "prompt": "text",
            "human_prompt": "text",
            "llm_labels": "label",
            "input_subject": "label",
            "subject": "label",
        },
    )

    # drop the superfluous label column if it exists
    if "label" in jl.df.columns:
        jl.df.drop(inplace=True, columns=["label"])
    if "id" in jl.df.columns:
        jl.df.drop(inplace=True, columns=["id"])
    if "title" in jl.df.columns:
        jl.df.drop(inplace=True, columns=["title"])
    if "act" in jl.df.columns:
        jl.df.drop(inplace=True, columns=["act"])
    if "chatgpt_response" in jl.df.columns:
        jl.df.drop(inplace=True, columns=["chatgpt_response"])

    # add input source column
    jl.df["input_src"] = file

    # add input subject column if it doesn't exist
    if "label" not in jl.df.columns:
        jl.df["label"] = "unlabeled"

    # generate entity_id and potentially drop line
    for i, row in jl.df.iterrows():
        if row["text"] == "":
            jl.df.drop(i, inplace=True)
            continue
        jl.df.at[i, "entity_id"] = hashlib.sha256(
            row["text"].encode("utf-8")
        ).hexdigest()
        subj = row["label"]
        if isinstance(subj, list):
            logging.warning(
                f"input row[{i}]: 'label' has {len(subj)} entries. Flattening to the first entry."
            )
            jl.df.at[i, "label"] = subj[0]
            subj = subj[0]
        if " " in subj:
            logging.info(
                f"input row[{i}]: {{'label': '{subj}'}}: contains spaces. Replacing spaces with '_' characters."
            )
            jl.df.at[i, "label"] = subj.strip().replace(" ", "_")

    jl.to_file(args.outputDir, f"{args.prefix}.{args.chunkerName}.input")


if __name__ == "__main__":
    main()
