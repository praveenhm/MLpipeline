# Data Format

## File Format Schema

### Input Files

All input files must have the following fields/columns:

- `text`: the actual text snippet / chunk
- `label`: the input classification or subject of the `text`
- `input_src`: a string identifier by which we can identify where this `text` snippet is coming from
- `entity_id`: a stable unique identifier of `text`. In this case this is the hexadecimal string representation of a SHA256 sum of `text`.

```json
{
  "text": "I was recently offered a position contingent upon passing the background check. I entered what I thought to be the start/end dates based off my resume to the third party background check. I thought I started in January 2013 and ended in December 2015. After calling my former employer (which is something I should've done for all of my past jobs), I was informed I started in July 2013 and ended in March 2016. I immediately called the third-party background check company, who said they would have their researcher update this, but my resume still has the wrong dates. Overall I worked for the company for 33 months versus 33, but the start/end dates are off. \n\nWill my background check for employment verification be flagged? Please help!",
  "label": "human_resource",
  "input_src": "reddit_filtered.jsonl",
  "entity_id": "527dbb0800a214ef3b5648b82807beb4c8c5433efe4f0c0c3fa8e5bbaf6e0e6d"
}
```

### Output Files

All processed files must have the additional fields/columns:

- `id`: a 64bit running integer representing an ID within this set. **NOTE:** compared to `entity_id` this is not a stable ID, and only unique within the file.
- `MODEL_labels`: per model there will be a list of strings of classifications / labels
- `MODEL_verdict`: per model there will be a string of either: "correct", "partially correct" or "incorrect"
    - correct: `label == MODEL_labels[0]`
    - partially correct: `label in MODEL_labels`
    - incorrect: all others

```json
{
  "text": "I was recently offered a position contingent upon passing the background check. I entered what I thought to be the start/end dates based off my resume to the third party background check. I thought I started in January 2013 and ended in December 2015. After calling my former employer (which is something I should've done for all of my past jobs), I was informed I started in July 2013 and ended in March 2016. I immediately called the third-party background check company, who said they would have their researcher update this, but my resume still has the wrong dates. Overall I worked for the company for 33 months versus 33, but the start/end dates are off. \n\nWill my background check for employment verification be flagged? Please help!",
  "label": "human_resource",
  "input_src": "reddit_filtered.jsonl",
  "entity_id": "527dbb0800a214ef3b5648b82807beb4c8c5433efe4f0c0c3fa8e5bbaf6e0e6d",
  "deberta_labels": [
    "human_resource",
    "legal"
  ],
  "deberta_verdict": "correct",
  "mistral_labels": [
    "risk_and_compliance"
  ],
  "mistral_verdict": "incorrect",
  "zephyr_labels": [
    "human_resource"
  ],
  "zephyr_verdict": "correct",
  "id": 1
}
```

## Processing Folder Structure

### Local Folder Structure

Our main folder for processing sits locally at this top-level folder (the root being the repository):

./data/src

    Per dataset we are going to add a dedicated folder (like reddit, quora, PDFs, etc.pp.)

    ./data/src/reddit/

        In every dataset we are going to have some top-level files:

        ./data/src/reddit/reddit.none.input.jsonl

            The main input file: must follow the input data format as explained before

        ./data/src/reddit/reddit.model-verdict.jsonl

            The main output file: must follow the ouptut data format as explained before

        ./data/src/reddit/reddit.model-verdict.discovered.jsonl

            Contains all discovered labels by all models.

        ./data/src/reddit/reddit.model-verdict.html
        ./data/src/reddit/reddit.model-verdict.ydata.html (TODO)
        ./data/src/reddit/reddit.model-verdict.png

            Data analytics for the main output file

        Per model, we are going to have a dedicated folder

        ./data/src/reddit/deberta

            ./data/src/reddit/deberta/reddit.none.input.deberta.jsonl

                The generated file after running through deberta.
                It is going to have all the columns of the main input file + `deberta_labels`

            ./data/src/reddit/deberta/reddit.none.input.deberta.correct_data.jsonl

                A subset of the generated file with only correct data: `label == deberta_labels[0]`

            ./data/src/reddit/deberta/reddit.none.input.deberta.partly_correct_data.jsonl

                A subset of the generated file with only partially correct data: `label in deberta_labels`

            ./data/src/reddit/deberta/reddit.none.input.deberta.incorrect_data.jsonl

                A subset of the generated file with only incorrect data: is not correct or partially correct

    Apart from a folder per dataset there are also globally concatenated and merged files from all datasets:

    ./data/src/model-verdict.jsonl

        The main combined output file: must follow the ouptut data format as explained before

    ./data/src/model-verdict.discovered.jsonl

        Contains all discovered labels by all models for all datsets.

    ./data/src/model-verdict.html
    ./data/src/model-verdict.ydata.html (TODO)
    ./data/src/model-verdict.png

        Data analytics for the main combined output file

### GCS Bucket Folder Structure

We are uploading all these files to the following GCS bucket.

Project: development-398309
Bucket: docprocessor

Our top-level folder here will be `src`.
So locally `./data/src` becomes `gs://docprocessor/src/`.
The rest of the folder structure will be preserved as it is locally.
