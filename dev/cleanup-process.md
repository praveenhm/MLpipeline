# Cleanup Process

Tasks at hand:

- `irrelevant`/`conversation` analysis
- manual cleanup
- input label verdict: we might got the input label wrong if all models disagree

## Manual Cleanup Process

- index lists
- author lists
- appendix lists
- URLs
- titles

### Collaboration Session on 20240229

We went through an exercise of trying to clean up our dataset even more.
The starting point was the file from: `gs://docprocessor/data/20240228-new-data-format/model-verdict.jsonl`

Here is the (cleaned) bash history of what we did.

```bash
# select all rows where the first label of the models labels match
jq -c 'select(.deberta_labels[0] == .mistral_labels[0] and .deberta_labels[0] == .zephyr_labels[0])' model-verdict.jsonl > models_agree.jsonl

# remove all occurrences of http or HTTP
cat models_agree.jsonl | grep -v 'http' | grep -v 'HTTP' > models_agree2.jsonl

# we're sorting the rows here
cat models_agree2.jsonl | sort > models_agree3.jsonl

# remove all lines where the text starts with a '$' sign
cat models_agree3.jsonl | grep -v -E '^{"text":"\$' > models_agree4.jsonl

# keep only the lines where the text starts with a letter
cat models_agree4.jsonl | grep -E '^\{"text":"[AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz]+' > models_agree5.jsonl

# remove all lines that contain the text "chapter" as they are most likely chapter headings
cat models_agree5.jsonl | grep -v -E '^\{"text":"[Cc]?(hapter|HAPTER)' > models_agree6.jsonl

# remove all lines that contain "in: " strings as they are most likely refering to references
cat models_agree6.jsonl | grep -v -E '[Ii][Nn]\:[ ]?' > models_agree7.jsonl

# remove all lines that have a year number in parantheses or brackets like '(2017)' or '[2017]' as they are most likely references
cat models_agree7.jsonl | grep -v -E '\([12][[:digit:]]{3}\)' > models_agree8.jsonl
cat models_agree8.jsonl | grep -v -E '\[[12][[:digit:]]{3}\]' > models_agree9.jsonl

# remove all lines which have duplicate entity_id strings because the chunks are essentially the same
jq -s -c 'group_by(.entity_id) | map(.[0])[]' models_agree9.jsonl > models_agree10.jsonl

# we are uploading all the above test data to this GCS bucket
gsutil cp models_agree* gs://docprocessor/data/20240229/
gsutil ls gs://docprocessor/data/20240229/
```
