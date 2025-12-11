import os
import json
from pathlib import Path
import pandas as pd
from joblib import load

MODEL_DIR = Path("Experiment_B/outputs/iemocap_emotion_experiment")
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "svm_model.joblib"  

DATA_DIR = Path("data_json")  
OUTPUT_DIR = Path("Experiment_B/outputs/json_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "summary_accuracy.csv"

print("Loading vectorizer & classifier...")
vectorizer = load(VECTORIZER_PATH)
clf = load(MODEL_PATH)

def predict_emotion(text: str) -> str:
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

def process_one_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dialogue_turns = data.get("dialogue_turns", [])

    total = 0
    correct = 0

    for turn in dialogue_turns:

        speaker = turn.get("speaker", "").lower()

        if speaker == "interviewer":
            continue

        if "text" in turn:
            text = turn["text"].strip()
            if text:

                pred = predict_emotion(text)

                turn["predicted_emotions"] = [pred]

                if "emotions" in turn and turn["emotions"]:
                    gold = turn["emotions"][0]  
                    total += 1
                    if gold == pred:
                        correct += 1

        if "sentences" in turn:
            for sent in turn["sentences"]:
                text = sent["text"].strip()
                if not text:
                    continue

                pred = predict_emotion(text)
                sent["predicted_emotions"] = [pred]

                if "emotions" in sent and sent["emotions"]:
                    gold = sent["emotions"][0]
                    total += 1
                    if gold == pred:
                        correct += 1

    accuracy = correct / total if total > 0 else None

    out_path = OUTPUT_DIR / path.name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved → {out_path}")

    return {
        "file": path.name,
        "total_sentences": total,
        "correct_predictions": correct,
        "accuracy": accuracy
    }

def main():
    json_files = sorted(DATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotated dialogues in {DATA_DIR}")

    summary_rows = []

    for file in json_files:
        print(f"\nProcessing {file.name} ...")
        row = process_one_file(file)
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nAccuracy summary saved → {SUMMARY_CSV}")
    print(df)


if __name__ == "__main__":
    main()
