"""Train and evaluate a baseline Bodo tone text classifier."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "example_with_tone.csv"
DEFAULT_OUTPUT = ROOT / "models" / "retrained_pipeline.joblib"


def load_dataset(path: Path) -> pd.DataFrame:
    # The supplied CSV has no header row; the first line is an actual example.
    df = pd.read_csv(path, header=None, names=["sentence", "tonal_word", "label"])
    df = df.dropna(subset=["sentence", "tonal_word", "label"]).copy()
    for col in ["sentence", "tonal_word", "label"]:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["sentence"] != "") & (df["tonal_word"] != "") & (df["label"] != "")]
    df["label"] = df["label"].str.upper()
    df = df[df["label"].isin(["H", "L"])]
    if df.empty:
        raise ValueError("No valid rows found. Expected sentence, tonal_word, label columns and H/L labels.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"Valid examples: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts().sort_index().to_string())

    X = df["sentence"] + " [TONAL] " + df["tonal_word"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (label order H, L):")
    print(confusion_matrix(y_test, predictions, labels=["H", "L"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"\nSaved retrained pipeline to: {args.output}")


if __name__ == "__main__":
    main()
