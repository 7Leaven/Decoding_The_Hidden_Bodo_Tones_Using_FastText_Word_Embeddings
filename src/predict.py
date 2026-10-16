"""Predict the tone of a target Bodo word in a sentence."""
import argparse
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "bodo_tone_classifier.pkl"


def load_model(model_path: Path):
    # Only load pickle files from trusted sources.
    with model_path.open("rb") as file:
        return pickle.load(file)


def predict(sentence: str, word: str, model):
    sentence = sentence.strip()
    word = word.strip()
    if not sentence or not word:
        raise ValueError("Both sentence and target word must be non-empty.")
    model_input = f"{sentence} [TONAL] {word}"
    return model.predict([model_input])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", help="Bodo sentence containing the target word")
    parser.add_argument("--word", help="Target word whose tone should be predicted")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to trusted model pickle")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    sentence = args.sentence if args.sentence is not None else input("Enter a sentence in Bodo: ")
    word = args.word if args.word is not None else input("Enter the target word: ")

    model = load_model(args.model)
    result = predict(sentence, word, model)
    print(f"Predicted tone for '{word.strip()}': {result}")


if __name__ == "__main__":
    main()
