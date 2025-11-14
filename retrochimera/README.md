This README describes the steps and scripts involved in our full model pipeline, starting from training data, and ending at ensembling. First two steps can be skipped if the data is already cleaned up and/or split.

# Deduplication

- Input: `*.smi` file with mapped reaction SMILES
- Output: `*.smi` file with deduplicated mapped reaction SMILES
- Script: `./retrochimera/cli/deduplicate_dataset.py`

This light preprocessing step removes duplicated reaction SMILES that map to the same canonical SMILES. Output format is same as input format, and the output is formed using still the original (non-canonical) SMILES (the canonicalization is just used for grouping).

# Filtering and splitting

- Input: `*.smi` file with mapped reaction SMILES
- Output: `{train, val, test}.smi` files with mapped reaction SMILES
- Script: `./retrochimera/cli/split_dataset.py`

This step filters and normalizes the data (e.g. removes side products, filters out cases where there are too many reactants or massive reactant-product size imbalance, etc). It then split it into folds (groups reactions by products, then splits randomly).

# Template extraction

- Input: `{train, val, test}.smi` or `{raw_train, raw_val, raw_test}.csv` files with mapped reaction SMILES
- Output: `{train, val, test}.jsonl` files with processed datapoints and `template_lib.json` with templates
- Script: `./retrochimera/cli/extract_templates.py`

This step extracts templates, and converts the data into our internal JSONL format.

Note that template extraction is currently done with standard rdchiral. We have preliminary evidence that improved template extraction can lead to better results, but this needs further investigation.

# Data augmentation for De-Novo Model

- Input: `{train, val, test}.jsonl` files with mapped reaction SMILES
- Output: `{train, val, test}.jsonl` files with augmented datapoints using R-SMILES algorithm
- Script: `./retrochimera/cli/augment_rsmiles.py`

This step augments the original dataset (in JSONL format, containing mapped reaction SMILES information) by using R-SMILES to align and augment the original smiles of products and reactants. This augmentation aims to improve the performance of the RetroChimera De-Novo model.

# Tokenizer construction for De-Novo Model

- Input: `{train, val, test}.jsonl` files with augmented datapoints using R-SMILES algorithm
- Output: `vocab.txt` file
- Script: `./retrochimera/cli/build_tokenizer.py`

This step constructs a tokenizer and generates the corresponding vocabulary file based on the training dataset for the De-Novo model.

# Data featurization

- Input: `{train, val, test}.jsonl` and `template_lib.json` produced by previous step
- Output: `data.h5` file containing all featurized folds and `template_lib.json` (copied from input)
- Script: `./retrochimera/cli/preprocess.py`

This step converts the data further into a featurized form ready for training. This is different depending on the downstream model, e.g. for graph-based model, this is when the molecules are turned into graphs. Due to this, in this step one needs to set the model class (and potentially other hyperparameters that affect data preprocessing).

# Training

- Input: `data.h5` and `template_lib.json` (the latter is ignored for SMILES-based models)
- Output: `*.ckpt` model checkpoint(s) and `template_lib.json` (copied from input)
- Script: `./retrochimera/cli/train.py`

In this step one trains a model on the featurized data from previous step. Model class and other hyperparameters set here must match those set during preprocessing. By default training will proceed for several epochs, saving checkpoints along the way, and then average out a set of best checkpoints to form `combined.ckpt`.
Settings used for the provided checkpoints can be found under `retrochimera/cli/config`, and are accessed by setting the `preset` argument.

# Ensembling

- Input: `*.json` files containing dumped model outputs on validation and train set
- Output: `ensembles_optimized.json` file with ensembling weights and validation/test accuracies for the ensembles
- Script: `./retrochimera/cli/optimize_ensembles.py`

Finally, in this step one can ensemble several models into a single stronger one, e.g. combine NeuralLoc and R-SMILES 2 into RetroChimera. Instead of taking checkpoints as input, this script uses pre-generated model outputs produced by running `cli/eval.py` on validation and test data.
