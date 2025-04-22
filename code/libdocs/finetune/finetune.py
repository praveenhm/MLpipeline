from math import log


def finetune(
    hf_access_token: str,
    wandb_access_token: str,
    model_for_training_finetune: str,  # e.g. "microsoft/deberta-v3-base"
    input_hf_dataset: str,
    hf_model_name: str,
    downsample: bool = False,
    wandb_output: str = "./output",
):
    import gc
    import logging
    import warnings
    from datetime import datetime, timedelta, timezone

    # from random import sample
    import datasets
    import numpy as np
    import pandas as pd
    import torch
    import wandb
    from accelerate.utils import release_memory
    from datasets import load_dataset
    from libdocs.wandb.wandb_report import WandbMetricsReport
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
    )
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(
        "\n"
        "===========================================\n"
        f"model_for_training_finetune : {model_for_training_finetune} \n"
        f"input_hf_dataset : {input_hf_dataset} \n"
        f"hf_model_name : {hf_model_name} \n"
        f"downsample : {downsample} \n"
        "==========================================="
    )

    # Convert current date to Pacific Time Zone
    pacific_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-8)))
    formatted_date = pacific_time.strftime("%Y%m%d")

    # set random seed for reproducibility
    DATE = formatted_date
    SEED_GLOBAL = 42
    WANDB_REPORT_DIR = wandb_output
    WANDB_REPORT_FILE_NAME = "wandb-report.html"

    np.random.seed(SEED_GLOBAL)

    dataset_finetune = load_dataset(input_hf_dataset, token=hf_access_token)
    logging.info(f"Dataset: {input_hf_dataset} \n")
    logging.info(dataset_finetune)

    df_train = dataset_finetune["train"].to_pandas()
    df_test = dataset_finetune["test"].to_pandas()

    # Create label maps *before* redefining DataFrames
    # Use training set labels to define the map
    unique_labels_train = np.sort(df_train.label.unique()).tolist()
    label2id = dict(
        zip(
            unique_labels_train,
            np.sort(pd.factorize(unique_labels_train, sort=True)[0]).tolist(),
        )
    )
    id2label = dict(
        zip(
            np.sort(pd.factorize(unique_labels_train, sort=True)[0]).tolist(),
            unique_labels_train,
        )
    )
    logging.info(
        f"Derived label maps from training data: \n label2id: {label2id} \n id2label: {id2label}"
    )

    # Transform the label to a numeric value using the map
    df_train = pd.DataFrame(
        {
            "text": df_train["text"],
            "label_text": df_train["label"],
            "label": df_train["label"].map(label2id),
        }
    )
    df_test = pd.DataFrame(
        {
            "text": df_test["text"],
            "label_text": df_test["label"],
            "label": df_test["label"].map(label2id),
        }
    )
    logging.info("..........After transforming........")
    logging.info(df_train.head())
    logging.info(df_test.head())

    logging.info(
        "Length of training and test sets: \n"
        " train: "
        f" {len(df_train)} \n"
        " test: "
        f"{len(df_test)}"
    )

    # optional: use training data sample size of e.g. 1000 for faster testing
    if downsample is True:
        sample_size = 1000
        df_train = df_train.sample(
            n=min(sample_size * 5, len(df_train)), random_state=SEED_GLOBAL
        ).copy(deep=True)
        df_test = df_test.sample(
            n=min(sample_size * 2, len(df_test)), random_state=SEED_GLOBAL
        ).copy(deep=True)
        logging.info(
            "Length of training and test sets after sampling: \n"
            " train:  "
            f"{len(df_train)}\n"
            " test:  "
            f"{len(df_test)}"
        )

    # inspect the data
    # label distribution train set
    logging.info(
        f"Train set label distribution: {df_train.label_text.value_counts()} \n"
        f"Train count: {len(df_train)}"
    )
    # label distribution test set
    logging.info(
        f"Test set label distribution: {df_test.label_text.value_counts()} \n"
        f"Test count: {len(df_test)}"
    )

    # Data preprocessing
    df_train["text"] = df_train.text.fillna("")
    df_test["text"] = df_test.text.fillna("")

    #  keep 3 columns: label,label_text,text  remove other columns
    df_train = df_train[["label", "label_text", "text"]]
    df_test = df_test[["label", "label_text", "text"]]

    # inspect the data
    logging.info(df_train.head())
    logging.info(df_test.head())

    # Load a Transformer
    # load a model and its tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_for_training_finetune, use_fast=True, model_max_length=512
    )
    logging.info(f"Base model_name for training :  {model_for_training_finetune} \n")

    # The label maps are already created above
    # link the numeric labels to the label texts
    # label_text = np.sort(df_test.label_text.unique()).tolist() # No longer needed here
    # label2id = dict(                             # No longer needed here
    #     zip(                                     # No longer needed here
    #         np.sort(label_text),                 # No longer needed here
    #         np.sort(pd.factorize(label_text, sort=True)[0]).tolist(), # No longer needed here
    #     )                                        # No longer needed here
    # )                                            # No longer needed here
    # id2label = dict(                             # No longer needed here
    #     zip(                                     # No longer needed here
    #         np.sort(pd.factorize(label_text, sort=True)[0]).tolist(), # No longer needed here
    #         np.sort(label_text),                 # No longer needed here
    #     )                                        # No longer needed here
    # )                                            # No longer needed here
    config = AutoConfig.from_pretrained(
        model_for_training_finetune,
        label2id=label2id,  # Use the maps created earlier
        id2label=id2label,  # Use the maps created earlier
        num_labels=len(label2id),
    )
    logging.info(f"\n Model config updated with label maps: {label2id} \n")

    # load model with config
    model = AutoModelForSequenceClassification.from_pretrained(
        model_for_training_finetune,
        config=config,
        # ignore_mismatched_sizes=True allows loading a pre-trained model even if
        # its classification head (last layer) doesn't match the new number of labels.
        # The head weights will be randomly initialized.
        ignore_mismatched_sizes=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    model.to(device)

    # Tokenize data
    # convert pandas dataframes to Hugging Face dataset object to facilitate pre-processing

    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_pandas(df_train),
            "test": datasets.Dataset.from_pandas(df_test),
        }
    )

    # tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Align with tokenizer loading
        )  # max_length can be reduced to e.g. 256 to increase speed, but long texts will be cut off

    dataset = dataset.map(tokenize, batched=True)

    # remove unnecessary columns for model training doesnt expect
    dataset = dataset.remove_columns(
        [
            "label_text",
        ]
    )

    # **Inspect processed data**
    logging.info("The overall structure of the pre-processed train and test sets:\n")
    logging.info(dataset)
    logging.info(dataset["train"].to_pandas().head())
    logging.info(dataset["test"].to_pandas().head())

    logging.info("\n\nAn example for a row in the tokenized dataset:\n")
    [logging.info(f"{key}:    {value}") for key, value in dataset["train"][0].items()]

    # logging with wandb

    # logging
    wandb.login(key=wandb_access_token)
    project_name = f"zeroshot-{DATE}"
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_for_training_finetune.split('/')[-1]}-zeroshot-{now}"
    # run_name = f"model-{hf_model_name}-{SEED_GLOBAL}"
    wandb.init(project=project_name, name=run_name)
    # if updating config here, HF trainer does not seem to log info to config anymore
    # wandb.config.update({"dataset_name_heldout": dataset_name_heldout}, allow_val_change=True)

    # Metrics
    # Function to calculate metrics

    def compute_metrics_standard(eval_pred):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            labels = eval_pred.label_ids
            pred_logits = eval_pred.predictions
            preds_max = np.argmax(
                pred_logits, axis=1
            )  # argmax on each row (axis=1) in the tensor

            # metrics
            precision_macro, recall_macro, f1_macro, _ = (
                precision_recall_fscore_support(labels, preds_max, average="macro")
            )  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
            precision_micro, recall_micro, f1_micro, _ = (
                precision_recall_fscore_support(labels, preds_max, average="micro")
            )  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
            acc_balanced = balanced_accuracy_score(labels, preds_max)
            acc_not_balanced = accuracy_score(labels, preds_max)

            metrics = {
                "accuracy": acc_not_balanced,
                "f1_macro": f1_macro,
                "accuracy_balanced": acc_balanced,
                "f1_micro": f1_micro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
            }

            return metrics

    # Setting training arguments / hyperparameters

    # Set the directory to write the fine-tuned model and training logs to.
    training_directory = f"./results/{hf_model_name.split('/')[-1]}-tanh-{now}"
    # training_directory = f'./results/{model_name.split("/")[-1]}-zeroshot-{args.dataset_name_heldout}-{now}'

    # FP16 is a hyperparameter which can increase training speed and reduce memory
    # consumption, but only on GPU and if batch-size > 8, see here:
    # https://huggingface.co/transformers/performance.html?#fp16
    fp16_bool = True if torch.cuda.is_available() else False

    # Overview of all training arguments:
    # https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    # Hugging Face tipps to increase training speed and decrease out-of-memory (OOM) issues:
    # https://huggingface.co/transformers/performance.html?
    # ---- Removed obsolete commented-out TrainingArguments block ----

    # copy of finetune
    eval_batch = 64 if "large" in model_for_training_finetune else 64 * 2  # 40
    per_device_train_batch_size = (
        16 if "large" in model_for_training_finetune else 32
    )  # 8
    # gradient_accumulation_steps = 4 if "large" in model_name else 1

    hub_model_id = hf_model_name

    logging.info(f"Hub model id: ==================> {hub_model_id}")
    # ---- Removed obsolete commented-out TrainingArguments block ----
    # end of copied from other

    # Fine-tuning and evaluation

    train_args = TrainingArguments(
        output_dir=training_directory,
        logging_dir=f"{training_directory}/logs",
        # deepspeed="ds_config_zero3.json",  # if using deepspeed
        # lr_scheduler_type="linear",
        # can increase speed with dynamic padding, by grouping similar length texts
        # https://huggingface.co/transformers/main_classes/trainer.html
        # group_by_length=False,
        learning_rate=9e-6 if "large" in model_for_training_finetune else 2e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=eval_batch,
        # (!adapt/halve batch size accordingly). accumulates gradients over X steps,
        # only then backward/update. decreases memory usage, but also slightly speed
        # gradient_accumulation_steps=gradient_accumulation_steps,
        # eval_accumulation_steps=2,
        num_train_epochs=2,
        # max_steps=400,ll
        # warmup_steps=0,  # 1000,
        warmup_ratio=0.06,  # 0.1, 0.06
        weight_decay=0.01,  # 0.1,
        # ! only makes sense at batch-size > 8. loads two copies of
        # model weights, which creates overhead. https://huggingface.co/transformers/performance.html?#fp16
        # fp16=fp16_bool,
        bf16=fp16_bool,
        fp16_full_eval=fp16_bool,
        evaluation_strategy="epoch",
        seed=SEED_GLOBAL,
        # metric_for_best_model="accuracy",
        metric_for_best_model="f1_macro",
        # eval_steps=300,  # evaluate after n steps if evaluation_strategy!='steps'.
        # defaults to logging_steps
        save_strategy="epoch",  # options: "no"/"steps"/"epoch"
        # save_steps=1_000_000,  # Number of updates steps before two checkpoint saves.
        # If a value is passed, will limit the total amount of checkpoints. Deletes
        # the older checkpoints in output_dir
        # save_total_limit=1,
        # logging_strategy="epoch",
        load_best_model_at_end=True,
        report_to="all",  # "all"
        run_name=run_name,
        push_to_hub=True,  # does not seem to work if save_strategy="no"
        hub_model_id=hf_model_name,
        hub_token=hf_access_token,
        hub_strategy="end",
        hub_private_repo=True,
    )
    # If you get an 'out-of-memory' error, reduce the 'per_device_train_batch_size'
    # to 8 or 4 in the TrainingArguments above and restart the runtime. If you
    # don't restart your runtime (menu to the to left 'Runtime' > 'Restart runtime')
    # and rerun the entire script, the 'out-of-memory' error will probably not go away.

    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics_standard,
    )

    if device == "cuda":
        # free memory
        flush()
        release_memory(model)
        # del (model, trainer)

    trainer.train()

    if device == "cuda":
        # free memory
        flush()
        release_memory(model)
        # del (model, trainer)

    # Evaluate the fine-tuned model on the held-out test set
    results = trainer.evaluate()
    logging.info(results)

    # =================end============================

    # Push to Hugging Face hub

    # if args.upload_to_hub and args.do_train:
    # push to hugging face hub from save .safetensor to pytorch_model.bin
    trainer.push_to_hub(commit_message="End of training")

    # trainer.push_to_hub(commit_message="End of training")

    # tokenizer needs to be uploaded separately to create tokenizer.json
    # otherwise only tokenizer_config.json is created and pip install sentencepiece is required
    tokenizer.push_to_hub(
        repo_id=hf_model_name,
        use_temp_dir=True,
        private=True,
        use_auth_token=hf_access_token,
    )

    # create wandb report
    others = {"Train Datset": len(df_train)}
    logging.info(f"Creating wandb report : {wandb.run.dir}")
    report_generator = WandbMetricsReport(wandb.run.dir)
    report_generator.create_report(
        location=WANDB_REPORT_DIR,
        filename=WANDB_REPORT_FILE_NAME,
        others=others,
    )

    # log results to wandb
    wandb.finish()
