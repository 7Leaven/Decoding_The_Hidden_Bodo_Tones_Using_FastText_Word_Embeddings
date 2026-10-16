# Methodology

## Input representation

Each training example is represented as:

`<sentence> [TONAL] <target_word>`

The `[TONAL]` marker separates the sentence from the word being classified.

## Baseline pipeline

The included reproducible baseline uses:
1. TF-IDF features over word unigrams and bigrams.
2. Logistic Regression classification.
3. A stratified train/test split with a fixed random seed.
4. Accuracy, per-class precision/recall/F1, and a confusion matrix for evaluation.

## Relationship to the project title

The project title refers to FastText word embeddings. The supplied notebook also contains TF-IDF and Logistic Regression experiments, and the supplied `.pkl` loads as a scikit-learn Pipeline. Verify the actual training procedure used for each model before describing the baseline as a FastText model. The included `src/train.py` is a clean TF-IDF baseline; it does not train FastText embeddings.

## Research limitations

- Small labeled datasets can produce unstable estimates.
- Sentence context alone does not provide direct acoustic pitch measurements.
- Ensure the test split does not contain duplicated or near-duplicated examples from training.
- Compare against a majority-class baseline and consider grouped splits if repeated sentences or words appear across examples.
