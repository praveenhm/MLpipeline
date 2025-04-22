# 🚀 Fine‑Tuning Journey: From Raw Text to 🎯 Accurate Model

Ever wondered **what actually happens** when we say *"we fine‑tuned the model"*?  Below is a friendly, emoji‑powered walk‑through of the entire pipeline so you can share it at stand‑ups or in a PR description.

---
## 🗺️ High‑Level Map

| 🏁 Step | 📂 File | 🛠️ What It Does |
|---------|---------|-----------------|
| 1️⃣  | `cleaner_finetune.py` | 🧹 Cleans & balances raw JSONL, then 🚚 uploads it as a HF dataset (`train`/`test`). |
| 2️⃣  | `libdocs/finetune/finetune.py` | 🏋️‍♂️ Fine‑tunes a base model on the `train` split, pushes the 🎯 new model to HF Hub. |
| 3️⃣  | `libdocs/finetune/run.py` | 🔬 Evaluates the pushed model on the held‑out `test` split, logs top‑k accuracy. |
| 4️⃣  | `classifier_model.py` | 🎛️ Command‑line orchestrator that wires everything together (GPU check, args, train, eval). |

---
## 🪄 Step‑by‑Step Magic

### 1️⃣ Data Preparation  🧹
* **Load** raw JSONL (you tell us which columns are `text` ✍️ & `label` 🏷️).
* **Clean** with TF‑IDF similarity – goodbye duplicates & noise 👋.
* **Balance** the classes ⚖️ (optional label filters & down‑sampling).
* **Split** into 80 % train / 20 % test ✂️, keeping label proportions.
* **Upload** the two CSVs to a brand‑new 🤩 Hugging Face dataset repo.

> 📦 Outcome: `username/dataset‑name` with perfectly prepped `train`/`test` splits.

### 2️⃣ Model Fine‑Tuning  🏋️‍♂️
* **Download** your freshly minted dataset.
* **Map** labels to integers (`label2id`/`id2label`) ➡️ ensures the model's head matches your classes.
* **Tokenize** every text up to 512 tokens ✂️📏.
* **Train** for 2 epochs using the HF `Trainer` (accuracy + macro/micro F1 tracked on WandB 📊).
* **Push** the best checkpoint & tokenizer to `username/model‑name` on HF Hub 🚀.

### 3️⃣ Evaluation  🔍
* **Instantiate** a `DebertaZeroShot` classifier with the pushed model.
* **Batch‑predict** the held‑out test set (configurable batch size).⚙️
* **Compute** 🎯 Top‑1, 🎯🎯 Top‑2, 🎯🎯🎯 Top‑3 accuracy (fixed bug only checks top‑k list). 
* **Log** results to console *and* append per‑example JSONL with predictions, scores & correctness flags 📄.

### 4️⃣ Orchestration  🎛️
* Single CLI command handles 💻 GPU check, 🔑 tokens, file paths and flags.
* Defaults to *train ➜ evaluate* workflow; add `--examples` to skip training & test an existing model instead.
* Produces everything you need: 📚 dataset on Hub, 🤗 model on Hub, 📈 WandB run, and a detailed results file.

---
## ✨ Why This Rocks

* **Reproducible** – every artefact lives on the Hugging Face Hub.
* **Balanced Data** – no more skewed labels ruining your metrics.
* **Real‑world Metrics** – top‑k accuracy shows if the right answer is "close" even when not rank‑1.
* **Modular** – swap the base model, tweak hyper‑params, or plug in a new cleaner.
* **Hands‑free** – one command and the pipeline handles the rest. ☕️

Happy fine‑tuning! 🎉 