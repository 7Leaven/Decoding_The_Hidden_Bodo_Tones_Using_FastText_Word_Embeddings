# Bodo Tone Identification

A research project exploring **High (H)** and **Low (L)** tone classification for Bodo words in sentence context.

## Overview

The project takes a Bodo sentence and a target word, combines them with a `[TONAL]` marker, and predicts the target word's tone using a scikit-learn text-classification pipeline.

> **Research note:** This repository documents the supplied artifacts. The included CSV contains 493 data rows when read without a header (the first row is data, not column names). Reported performance must be recalculated from a reproducible evaluation before being presented as a final result.

## Repository structure

```text
bodo-tone-identification/
├── data/
│   ├── example_with_tone.csv
│   └── DATASET_CARD.md
├── docs/
│   └── METHODOLOGY.md
├── models/
│   └── bodo_tone_classifier.pkl
├── notebooks/
│   └── original_experiments.ipynb
├── src/
│   ├── predict.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset format

The CSV has three fields and **does not include a header row**:

1. `sentence`: Bodo sentence in Devanagari script
2. `tonal_word`: target word found in the sentence
3. `label`: tone label (`H` or `L`)

The repository copy is provided as an example dataset. Confirm that you have permission to publish it and that its source permits redistribution before making the repository public.

## Setup

Use Python 3.10 for compatibility with the supplied pickle model.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Predict a tone

```bash
python src/predict.py --sentence "बासननि आखाय बायनायजों बेखौ हमनायाव गोब्राब जायो" --word "आखाय"
```

Or run interactively:

```bash
python src/predict.py
```

The supplied model was serialized with scikit-learn 1.1.3. Loading scikit-learn pickle files across versions is not guaranteed to work; use the pinned environment and only load pickle files you trust.

## Train a model

```bash
python src/train.py --data data/example_with_tone.csv --output models/retrained_pipeline.joblib
```

The training script uses a stratified train/test split and a TF-IDF + Logistic Regression pipeline. It prints accuracy, a classification report, and a confusion matrix. These results are a new evaluation run and may differ from earlier notebook results.

## Method summary

1. Read the dataset as a headerless CSV.
2. Drop incomplete rows and normalize field types.
3. Construct the input as `sentence + " [TONAL] " + tonal_word`.
4. Fit a TF-IDF vectorizer and Logistic Regression classifier.
5. Evaluate on a held-out test split.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for details and limitations.

## Limitations

- Tone labels and contextual examples are limited in size.
- Text-only classification cannot directly observe acoustic/prosodic signals such as pitch.
- Results depend on dataset quality, label consistency, and train/test splitting.
- The supplied notebook contains multiple experimental cells and should be treated as a record of experiments, not a single clean end-to-end pipeline.
- Do not claim a particular accuracy as the definitive result until it has been reproduced from the documented training/evaluation script.

## Academic attribution

**Project title:** Decoding the Hidden Bodo Tones: Identification Using FastText Word Embeddings  
**Institution:** Central Institute of Technology, Kokrajhar  
**Author:** Hrittik Kumar Sah

Update the author, team members, supervisor, dataset citations, and license details as appropriate before publication. Do not add a license unless you have the right to license all included code and data.
